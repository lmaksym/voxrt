//! FastConformer encoder, scalar fp32 (PLAN 2.7–2.9). Mirrors HF `ParakeetEncoder` (transformers 5.18);
//! numerics documented in `docs/golden.md`. Ternary weights are dequantised to f32 at load in this phase.
use crate::format::ModelFile;
use crate::subsampling::Subsampling;
use crate::tensor::{dot, layer_norm, linear, load_f32_shaped, sigmoid, silu};
use crate::trits::dequant_matrix;
use crate::Result;

/// Dequantised ternary matrix `[out, in]`.
fn load_ternary(m: &ModelFile, name: &str) -> Result<(Vec<f32>, usize, usize)> {
    let q = m
        .ternary
        .quantized_modules
        .iter()
        .find(|q| q.name == name)
        .ok_or_else(|| crate::Error::Format(format!("{name} is not a ternary module")))?;
    let (_, qw) = m.tensor(&format!("{name}.qweight"))?;
    let (_, sc) = m.tensor(&format!("{name}.scales"))?;
    Ok((
        dequant_matrix(qw, sc, q.out_features, q.in_features, q.group_size),
        q.out_features,
        q.in_features,
    ))
}

pub struct FeedForward {
    w1: Vec<f32>, // [4d, d]
    w2: Vec<f32>, // [d, 4d]
    d: usize,
    h: usize,
}

impl FeedForward {
    fn load(m: &ModelFile, p: &str) -> Result<Self> {
        let (w1, h, d) = load_ternary(m, &format!("{p}.linear1"))?;
        let (w2, _, _) = load_ternary(m, &format!("{p}.linear2"))?;
        Ok(Self { w1, w2, d, h })
    }
    pub fn forward(&self, x: &[f32], t: usize, y: &mut [f32]) {
        let mut hbuf = vec![0f32; t * self.h];
        linear(x, t, self.d, &self.w1, None, self.h, &mut hbuf);
        hbuf.iter_mut().for_each(|v| *v = silu(*v));
        linear(&hbuf, t, self.h, &self.w2, None, self.d, y);
    }
}

pub struct Attention {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    wp: Vec<f32>,     // relative_k_proj
    bias_u: Vec<f32>, // [H, Dh]
    bias_v: Vec<f32>,
    d: usize,
    heads: usize,
    dh: usize,
}

impl Attention {
    fn load(m: &ModelFile, p: &str, heads: usize) -> Result<Self> {
        let (wq, d, _) = load_ternary(m, &format!("{p}.q_proj"))?;
        let (wk, _, _) = load_ternary(m, &format!("{p}.k_proj"))?;
        let (wv, _, _) = load_ternary(m, &format!("{p}.v_proj"))?;
        let (wo, _, _) = load_ternary(m, &format!("{p}.o_proj"))?;
        let (wp, _, _) = load_ternary(m, &format!("{p}.relative_k_proj"))?;
        let dh = d / heads;
        let bias_u = load_f32_shaped(m, &format!("{p}.bias_u"), &[heads, dh])?;
        let bias_v = load_f32_shaped(m, &format!("{p}.bias_v"), &[heads, dh])?;
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            wp,
            bias_u,
            bias_v,
            d,
            heads,
            dh,
        })
    }

    /// `x`: normalised input `[t, d]`; `rel_k`: `relative_k_proj(pos_emb)` `[2t-1, d]` (shared across layers' shapes but per-layer weights).
    pub fn forward(&self, x: &[f32], t: usize, pos_emb: &[f32], y: &mut [f32]) {
        let (d, h, dh) = (self.d, self.heads, self.dh);
        let (mut q, mut k, mut v) = (vec![0f32; t * d], vec![0f32; t * d], vec![0f32; t * d]);
        linear(x, t, d, &self.wq, None, d, &mut q);
        linear(x, t, d, &self.wk, None, d, &mut k);
        linear(x, t, d, &self.wv, None, d, &mut v);
        let np = 2 * t - 1;
        let mut rel_k = vec![0f32; np * d];
        linear(pos_emb, np, d, &self.wp, None, d, &mut rel_k);
        let scaling = 1.0 / (dh as f32).sqrt();
        let mut ctx = vec![0f32; t * d]; // [t, h*dh]
                                         // per head
        let mut scores = vec![0f32; t];
        let mut qu = vec![0f32; dh];
        let mut qv = vec![0f32; dh];
        for head in 0..h {
            let (bu, bv) = (
                &self.bias_u[head * dh..(head + 1) * dh],
                &self.bias_v[head * dh..(head + 1) * dh],
            );
            for i in 0..t {
                let qi = &q[i * d + head * dh..i * d + (head + 1) * dh];
                for c in 0..dh {
                    qu[c] = qi[c] + bu[c];
                    qv[c] = qi[c] + bv[c];
                }
                let mut max = f32::NEG_INFINITY;
                for j in 0..t {
                    let kj = &k[j * d + head * dh..j * d + (head + 1) * dh];
                    let p = t - 1 - i + j; // rel shift: position index for relative distance i-j
                    let pj = &rel_k[p * d + head * dh..p * d + (head + 1) * dh];
                    let s = (dot(&qu, kj) + dot(&qv, pj)) * scaling;
                    scores[j] = s;
                    max = max.max(s);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                let out = &mut ctx[i * d + head * dh..i * d + (head + 1) * dh];
                out.iter_mut().for_each(|o| *o = 0.0);
                for j in 0..t {
                    let w = scores[j] * inv;
                    let vj = &v[j * d + head * dh..j * d + (head + 1) * dh];
                    for c in 0..dh {
                        out[c] += w * vj[c];
                    }
                }
            }
        }
        linear(&ctx, t, d, &self.wo, None, d, y);
    }
}

pub struct ConvModule {
    pw1: Vec<f32>,     // [2d, d]
    dw: Vec<f32>,      // [d, k] with BatchNorm folded in (scale)
    dw_bias: Vec<f32>, // folded BN shift
    pw2: Vec<f32>,     // [d, d]
    d: usize,
    k: usize,
}

impl ConvModule {
    fn load(m: &ModelFile, p: &str, k: usize) -> Result<Self> {
        let (pw1, _, d) = load_ternary(m, &format!("{p}.pointwise_conv1"))?;
        let (pw2, _, _) = load_ternary(m, &format!("{p}.pointwise_conv2"))?;
        let mut dw = load_f32_shaped(m, &format!("{p}.depthwise_conv.weight"), &[d, 1, k])?;
        let gamma = load_f32_shaped(m, &format!("{p}.norm.weight"), &[d])?;
        let beta = load_f32_shaped(m, &format!("{p}.norm.bias"), &[d])?;
        let mean = load_f32_shaped(m, &format!("{p}.norm.running_mean"), &[d])?;
        let var = load_f32_shaped(m, &format!("{p}.norm.running_var"), &[d])?;
        // BN(eval): y = (x - mean) / sqrt(var + 1e-5) * gamma + beta  →  fold scale into the (bias-free) depthwise conv
        let mut dw_bias = vec![0f32; d];
        for c in 0..d {
            let s = gamma[c] / (var[c] + 1e-5).sqrt();
            for kk in 0..k {
                dw[c * k + kk] *= s;
            }
            dw_bias[c] = beta[c] - mean[c] * s;
        }
        Ok(Self {
            pw1,
            dw,
            dw_bias,
            pw2,
            d,
            k,
        })
    }

    pub fn forward(&self, x: &[f32], t: usize, y: &mut [f32]) {
        let (d, k) = (self.d, self.k);
        let pad = (k - 1) / 2;
        let mut g = vec![0f32; t * 2 * d];
        linear(x, t, d, &self.pw1, None, 2 * d, &mut g);
        // GLU over channels: a * sigmoid(b), a = first half
        let mut a = vec![0f32; t * d];
        for i in 0..t {
            for c in 0..d {
                a[i * d + c] = g[i * 2 * d + c] * sigmoid(g[i * 2 * d + d + c]);
            }
        }
        // depthwise conv over time, folded BN, SiLU
        let mut hbuf = vec![0f32; t * d];
        for i in 0..t {
            for c in 0..d {
                let mut acc = self.dw_bias[c];
                for kk in 0..k {
                    let j = i as isize + kk as isize - pad as isize;
                    if j >= 0 && (j as usize) < t {
                        acc += self.dw[c * k + kk] * a[j as usize * d + c];
                    }
                }
                hbuf[i * d + c] = silu(acc);
            }
        }
        linear(&hbuf, t, d, &self.pw2, None, d, y);
    }
}

pub struct LayerNormP {
    g: Vec<f32>,
    b: Vec<f32>,
}
impl LayerNormP {
    fn load(m: &ModelFile, p: &str, d: usize) -> Result<Self> {
        Ok(Self {
            g: load_f32_shaped(m, &format!("{p}.weight"), &[d])?,
            b: load_f32_shaped(m, &format!("{p}.bias"), &[d])?,
        })
    }
    pub fn forward(&self, x: &[f32], t: usize, y: &mut [f32]) {
        layer_norm(x, t, self.g.len(), &self.g, &self.b, y)
    }
}

pub struct ConformerLayer {
    pub ff1: FeedForward,
    pub attn: Attention,
    pub conv: ConvModule,
    pub ff2: FeedForward,
    pub norm_ff1: LayerNormP,
    pub norm_attn: LayerNormP,
    pub norm_conv: LayerNormP,
    pub norm_ff2: LayerNormP,
    pub norm_out: LayerNormP,
    d: usize,
}

/// Intermediate outputs of one layer (for golden comparison).
#[derive(Default)]
pub struct LayerTrace {
    pub norm_ff1: Vec<f32>,
    pub ff1: Vec<f32>,
    pub norm_attn: Vec<f32>,
    pub attn: Vec<f32>,
    pub norm_conv: Vec<f32>,
    pub conv: Vec<f32>,
    pub ff2: Vec<f32>,
}

impl ConformerLayer {
    pub fn load(m: &ModelFile, i: usize) -> Result<Self> {
        let e = &m.config.encoder_config;
        let p = format!("encoder.layers.{i}");
        Ok(Self {
            ff1: FeedForward::load(m, &format!("{p}.feed_forward1"))?,
            attn: Attention::load(m, &format!("{p}.self_attn"), e.num_attention_heads)?,
            conv: ConvModule::load(m, &format!("{p}.conv"), e.conv_kernel_size)?,
            ff2: FeedForward::load(m, &format!("{p}.feed_forward2"))?,
            norm_ff1: LayerNormP::load(m, &format!("{p}.norm_feed_forward1"), e.hidden_size)?,
            norm_attn: LayerNormP::load(m, &format!("{p}.norm_self_att"), e.hidden_size)?,
            norm_conv: LayerNormP::load(m, &format!("{p}.norm_conv"), e.hidden_size)?,
            norm_ff2: LayerNormP::load(m, &format!("{p}.norm_feed_forward2"), e.hidden_size)?,
            norm_out: LayerNormP::load(m, &format!("{p}.norm_out"), e.hidden_size)?,
            d: e.hidden_size,
        })
    }

    /// In place on `x` (`[t, d]`); optionally records sublayer outputs.
    pub fn forward(
        &self,
        x: &mut [f32],
        t: usize,
        pos_emb: &[f32],
        trace: Option<&mut LayerTrace>,
    ) {
        let d = self.d;
        let mut n = vec![0f32; t * d];
        let mut o = vec![0f32; t * d];
        let mut tr: Option<&mut LayerTrace> = trace;
        self.norm_ff1.forward(x, t, &mut n);
        self.ff1.forward(&n, t, &mut o);
        if let Some(tr) = tr.as_mut() {
            tr.norm_ff1 = n.clone();
            tr.ff1 = o.clone();
        }
        for (xv, ov) in x.iter_mut().zip(&o) {
            *xv += 0.5 * ov;
        }
        self.norm_attn.forward(x, t, &mut n);
        self.attn.forward(&n, t, pos_emb, &mut o);
        if let Some(tr) = tr.as_mut() {
            tr.norm_attn = n.clone();
            tr.attn = o.clone();
        }
        for (xv, ov) in x.iter_mut().zip(&o) {
            *xv += ov;
        }
        self.norm_conv.forward(x, t, &mut n);
        self.conv.forward(&n, t, &mut o);
        if let Some(tr) = tr.as_mut() {
            tr.norm_conv = n.clone();
            tr.conv = o.clone();
        }
        for (xv, ov) in x.iter_mut().zip(&o) {
            *xv += ov;
        }
        self.norm_ff2.forward(x, t, &mut n);
        self.ff2.forward(&n, t, &mut o);
        if let Some(tr) = tr.as_mut() {
            tr.ff2 = o.clone();
        }
        for (xv, ov) in x.iter_mut().zip(&o) {
            *xv += 0.5 * ov;
        }
        self.norm_out.forward(x, t, &mut n);
        x.copy_from_slice(&n);
    }
}

/// Relative positional table for `t` frames: `[2t-1, d]`, positions `t-1 … -(t-1)`, interleaved (sin, cos) per frequency.
pub fn rel_pos_table(t: usize, d: usize) -> Vec<f32> {
    let np = 2 * t - 1;
    let half = d / 2;
    let mut out = vec![0f32; np * d];
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / d as f32))
        .collect();
    for (r, row) in out.chunks_exact_mut(d).enumerate() {
        let pos = (t as i64 - 1 - r as i64) as f32;
        for i in 0..half {
            let f = inv[i] * pos;
            row[2 * i] = f.sin();
            row[2 * i + 1] = f.cos();
        }
    }
    out
}

pub struct Encoder {
    pub subsampling: Subsampling,
    pub layers: Vec<ConformerLayer>,
    pub d: usize,
}

impl Encoder {
    pub fn load(m: &ModelFile) -> Result<Self> {
        let e = &m.config.encoder_config;
        let subsampling = Subsampling::load(m)?;
        let layers = (0..e.num_hidden_layers)
            .map(|i| ConformerLayer::load(m, i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            subsampling,
            layers,
            d: e.hidden_size,
        })
    }

    /// `feats` `[t, n_mels]` → `(encoder_out [t_e, d], t_e)`; `layer_outs` receives every layer output if given.
    pub fn forward(
        &self,
        feats: &[f32],
        t: usize,
        mut layer_outs: Option<&mut Vec<Vec<f32>>>,
        layer0_trace: Option<&mut LayerTrace>,
    ) -> (Vec<f32>, usize) {
        let mut x = self.subsampling.forward(feats, t);
        let te = self.subsampling.out_frames(t);
        let pos = rel_pos_table(te, self.d);
        let mut l0 = layer0_trace;
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(&mut x, te, &pos, if i == 0 { l0.take() } else { None });
            if let Some(lo) = layer_outs.as_deref_mut() {
                lo.push(x.clone());
            }
        }
        (x, te)
    }
}
