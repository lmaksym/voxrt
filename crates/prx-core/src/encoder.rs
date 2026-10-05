//! FastConformer encoder (PLAN 2.7–2.9, 3.x). Mirrors HF `ParakeetEncoder` (transformers 5.18);
//! numerics documented in `docs/golden.md`. Ternary linears run in f32 (reference) or int8 (kernels), see `Precision`.
use crate::format::ModelFile;
use crate::subsampling::Subsampling;
use crate::tensor::{dot, layer_norm, load_f32_shaped, sigmoid, silu};
use crate::ternary_linear::{Precision, TernaryLinear};
use crate::Result;

pub struct FeedForward {
    w1: TernaryLinear, // [4d, d]
    w2: TernaryLinear, // [d, 4d]
    h: usize,
}

impl FeedForward {
    fn load(m: &ModelFile, p: &str, prec: Precision) -> Result<Self> {
        let w1 = TernaryLinear::load(m, &format!("{p}.linear1"), prec)?;
        let w2 = TernaryLinear::load(m, &format!("{p}.linear2"), prec)?;
        let h = w1.out_features();
        Ok(Self { w1, w2, h })
    }
    pub fn forward(&self, x: &[f32], t: usize, y: &mut [f32]) {
        let mut hbuf = vec![0f32; t * self.h];
        self.w1.forward(x, t, &mut hbuf);
        hbuf.iter_mut().for_each(|v| *v = silu(*v));
        self.w2.forward(&hbuf, t, y);
    }
}

pub struct Attention {
    wq: TernaryLinear,
    wk: TernaryLinear,
    wv: TernaryLinear,
    wo: TernaryLinear,
    wp: TernaryLinear, // relative_k_proj
    bias_u: Vec<f32>,  // [H, Dh]
    bias_v: Vec<f32>,
    d: usize,
    heads: usize,
    dh: usize,
}

impl Attention {
    fn load(m: &ModelFile, p: &str, heads: usize, prec: Precision) -> Result<Self> {
        let wq = TernaryLinear::load(m, &format!("{p}.q_proj"), prec)?;
        let wk = TernaryLinear::load(m, &format!("{p}.k_proj"), prec)?;
        let wv = TernaryLinear::load(m, &format!("{p}.v_proj"), prec)?;
        let wo = TernaryLinear::load(m, &format!("{p}.o_proj"), prec)?;
        let wp = TernaryLinear::load(m, &format!("{p}.relative_k_proj"), prec)?;
        let d = wq.out_features();
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

    /// `x`: normalised input `[t, d]`; `pos_emb`: relative positional table `[2t-1, d]`.
    pub fn forward(&self, x: &[f32], t: usize, pos_emb: &[f32], y: &mut [f32]) {
        let (d, h, dh) = (self.d, self.heads, self.dh);
        let (mut q, mut k, mut v) = (vec![0f32; t * d], vec![0f32; t * d], vec![0f32; t * d]);
        let xin = self.wq.prepare(x, t); // q, k, v share the same (quantised) input
        self.wq.forward_prepared(&xin, t, &mut q);
        self.wk.forward_prepared(&xin, t, &mut k);
        self.wv.forward_prepared(&xin, t, &mut v);
        let np = 2 * t - 1;
        let mut rel_k = vec![0f32; np * d];
        self.wp.forward(pos_emb, np, &mut rel_k);
        let scaling = 1.0 / (dh as f32).sqrt();
        let mut ctx = vec![0f32; t * d]; // [t, h*dh]
                                         // work items: (head, row range); each writes a disjoint ctx[i, head*dh..] block
        use rayon::prelude::*;
        let rows_per = 16usize;
        let n_chunks = t.div_ceil(rows_per);
        let ctxp = ctx.as_mut_ptr() as usize;
        let (q, k, v, rel_k) = (&q, &k, &v, &rel_k);
        let (bias_u, bias_v) = (&self.bias_u, &self.bias_v);
        (0..h * n_chunks).into_par_iter().for_each(|item| {
            let (head, ci) = (item / n_chunks, item % n_chunks);
            let (i0, i1) = (ci * rows_per, ((ci + 1) * rows_per).min(t));
            // SAFETY: this task writes only ctx[i, head*dh..(head+1)*dh] for i in [i0, i1); disjoint across tasks.
            let ctx = unsafe { std::slice::from_raw_parts_mut(ctxp as *mut f32, t * d) };
            let (bu, bv) = (
                &bias_u[head * dh..(head + 1) * dh],
                &bias_v[head * dh..(head + 1) * dh],
            );
            let mut scores = vec![0f32; t];
            let mut qu = vec![0f32; dh];
            let mut qv = vec![0f32; dh];
            for i in i0..i1 {
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
                    let sc = (dot(&qu, kj) + dot(&qv, pj)) * scaling;
                    scores[j] = sc;
                    max = max.max(sc);
                }
                let mut sum = 0f32;
                for sv in scores.iter_mut() {
                    *sv = (*sv - max).exp();
                    sum += *sv;
                }
                let inv = 1.0 / sum;
                let out = &mut ctx[i * d + head * dh..i * d + (head + 1) * dh];
                out.iter_mut().for_each(|o| *o = 0.0);
                for j in 0..t {
                    let wgt = scores[j] * inv;
                    let vj = &v[j * d + head * dh..j * d + (head + 1) * dh];
                    for c in 0..dh {
                        out[c] += wgt * vj[c];
                    }
                }
            }
        });
        self.wo.forward(&ctx, t, y);
    }
}

pub struct ConvModule {
    pw1: TernaryLinear, // [2d, d]
    dw: Vec<f32>,       // [d, k] with BatchNorm folded in (scale)
    dw_bias: Vec<f32>,  // folded BN shift
    pw2: TernaryLinear, // [d, d]
    d: usize,
    k: usize,
}

impl ConvModule {
    fn load(m: &ModelFile, p: &str, k: usize, prec: Precision) -> Result<Self> {
        let pw1 = TernaryLinear::load(m, &format!("{p}.pointwise_conv1"), prec)?;
        let pw2 = TernaryLinear::load(m, &format!("{p}.pointwise_conv2"), prec)?;
        let d = pw1.in_features();
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
        self.pw1.forward(x, t, &mut g);
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
        self.pw2.forward(&hbuf, t, y);
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

/// Wall-clock per sublayer type, summed over layers (enable with `PRX_PROFILE=1`).
#[derive(Default, Debug, Clone, Copy)]
pub struct Profile {
    pub subsampling: f64,
    pub ff: f64,
    pub attn: f64,
    pub conv: f64,
    pub norm: f64,
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
    pub fn load(m: &ModelFile, i: usize, prec: Precision) -> Result<Self> {
        let e = &m.config.encoder_config;
        let p = format!("encoder.layers.{i}");
        Ok(Self {
            ff1: FeedForward::load(m, &format!("{p}.feed_forward1"), prec)?,
            attn: Attention::load(m, &format!("{p}.self_attn"), e.num_attention_heads, prec)?,
            conv: ConvModule::load(m, &format!("{p}.conv"), e.conv_kernel_size, prec)?,
            ff2: FeedForward::load(m, &format!("{p}.feed_forward2"), prec)?,
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
        self.forward_profiled(x, t, pos_emb, trace, None)
    }

    pub fn forward_profiled(
        &self,
        x: &mut [f32],
        t: usize,
        pos_emb: &[f32],
        trace: Option<&mut LayerTrace>,
        mut prof: Option<&mut Profile>,
    ) {
        let d = self.d;
        let mut n = vec![0f32; t * d];
        let mut o = vec![0f32; t * d];
        let mut tr: Option<&mut LayerTrace> = trace;
        let mut tick = std::time::Instant::now();
        let mut lap = |field: fn(&mut Profile) -> &mut f64, prof: &mut Option<&mut Profile>| {
            if let Some(p) = prof.as_deref_mut() {
                *field(p) += tick.elapsed().as_secs_f64();
                tick = std::time::Instant::now();
            }
        };
        self.norm_ff1.forward(x, t, &mut n);
        lap(|p| &mut p.norm, &mut prof);
        self.ff1.forward(&n, t, &mut o);
        lap(|p| &mut p.ff, &mut prof);
        if let Some(tr) = tr.as_mut() {
            tr.norm_ff1 = n.clone();
            tr.ff1 = o.clone();
        }
        for (xv, ov) in x.iter_mut().zip(&o) {
            *xv += 0.5 * ov;
        }
        self.norm_attn.forward(x, t, &mut n);
        lap(|p| &mut p.norm, &mut prof);
        self.attn.forward(&n, t, pos_emb, &mut o);
        lap(|p| &mut p.attn, &mut prof);
        if let Some(tr) = tr.as_mut() {
            tr.norm_attn = n.clone();
            tr.attn = o.clone();
        }
        for (xv, ov) in x.iter_mut().zip(&o) {
            *xv += ov;
        }
        self.norm_conv.forward(x, t, &mut n);
        lap(|p| &mut p.norm, &mut prof);
        self.conv.forward(&n, t, &mut o);
        lap(|p| &mut p.conv, &mut prof);
        if let Some(tr) = tr.as_mut() {
            tr.norm_conv = n.clone();
            tr.conv = o.clone();
        }
        for (xv, ov) in x.iter_mut().zip(&o) {
            *xv += ov;
        }
        self.norm_ff2.forward(x, t, &mut n);
        lap(|p| &mut p.norm, &mut prof);
        self.ff2.forward(&n, t, &mut o);
        lap(|p| &mut p.ff, &mut prof);
        if let Some(tr) = tr.as_mut() {
            tr.ff2 = o.clone();
        }
        for (xv, ov) in x.iter_mut().zip(&o) {
            *xv += 0.5 * ov;
        }
        self.norm_out.forward(x, t, &mut n);
        x.copy_from_slice(&n);
        lap(|p| &mut p.norm, &mut prof);
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
    pub precision: Precision,
}

impl Encoder {
    /// Reference path: dequantised f32 ternary weights.
    pub fn load(m: &ModelFile) -> Result<Self> {
        Self::load_with(m, Precision::F32)
    }

    pub fn load_with(m: &ModelFile, precision: Precision) -> Result<Self> {
        let e = &m.config.encoder_config;
        let subsampling = Subsampling::load(m)?;
        let layers = (0..e.num_hidden_layers)
            .map(|i| ConformerLayer::load(m, i, precision))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            subsampling,
            layers,
            d: e.hidden_size,
            precision,
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
        let profiling = std::env::var_os("PRX_PROFILE").is_some();
        let mut prof = Profile::default();
        let t0 = std::time::Instant::now();
        let mut x = self.subsampling.forward(feats, t);
        prof.subsampling = t0.elapsed().as_secs_f64();
        let te = self.subsampling.out_frames(t);
        let pos = rel_pos_table(te, self.d);
        let mut l0 = layer0_trace;
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_profiled(
                &mut x,
                te,
                &pos,
                if i == 0 { l0.take() } else { None },
                if profiling { Some(&mut prof) } else { None },
            );
            if let Some(lo) = layer_outs.as_deref_mut() {
                lo.push(x.clone());
            }
        }
        if profiling {
            let total = t0.elapsed().as_secs_f64();
            eprintln!("encoder profile ({:?}, T_e={te}): total {total:.3}s · subsampling {:.3} · ff {:.3} · attn {:.3} · conv {:.3} · norm {:.3}", self.precision, prof.subsampling, prof.ff, prof.attn, prof.conv, prof.norm);
        }
        (x, te)
    }
}
