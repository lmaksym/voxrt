//! FastConformer encoder. Mirrors HF `ParakeetEncoder` (transformers 5.18); numerics in `docs/golden.md`.
//! Ternary linears run in f32 (reference) or int8 (kernels), see `Precision`. Phase 3c: fused q/k/v GEMM,
//! LayerNorm→quantise fusion, GEMM epilogues (SiLU, residual adds), one workspace per utterance.
use crate::format::ModelFile;
use crate::kernels::Epilogue;
use crate::packed::{QuantAct, TernaryMatrix};
use crate::subsampling::Subsampling;
use crate::tensor::{load_f32_shaped, par_rows, par_rows_cost};
use crate::ternary_linear::{Precision, TernaryLinear};
use crate::Result;
use std::sync::atomic::{AtomicU64, Ordering};

/// Buffers reused across layers within one encoder forward (sized for `t` frames).
pub struct Workspace {
    t: usize,
    hbuf: Vec<f32>,    // [t, 4d]
    qkv: Vec<f32>,     // [t, 3d]
    ctx: Vec<f32>,     // [t, d]
    g: Vec<f32>,       // [t, 2d]
    a: Vec<f32>,       // [t, d]
    hb: Vec<f32>,      // [t, d]
    heads: Vec<f32>,   // per-head operands: qu, qv, kh (3·h·t·dh) + rel (h·np·dh) + vᵀ (h·dh·t)
    s1: Vec<f32>,      // int8 attention core: content scores [h][n_pad][t2]
    s2: Vec<f32>, // int8 attention core: position scores [h][n_pad][np2] (only the band is written)
    scratch: Vec<f32>, // [t, d] for traces / f32 path
}

impl Workspace {
    pub fn new(t: usize, d: usize, ff: usize, heads: usize) -> Self {
        let np = 2 * t - 1;
        let (n_pad, t2, np2) = (t.div_ceil(8) * 8, t.div_ceil(2) * 2, np.div_ceil(2) * 2);
        Self {
            t,
            hbuf: vec![0f32; t * ff],
            qkv: vec![0f32; t * 3 * d],
            ctx: vec![0f32; t * d],
            g: vec![0f32; t * 2 * d],
            a: vec![0f32; t * d],
            hb: vec![0f32; t * d],
            heads: vec![0f32; 3 * t * d + np * d + d * t.div_ceil(16) * 16],
            s1: vec![0f32; heads * n_pad * t2],
            s2: vec![0f32; heads * n_pad * np2],
            scratch: vec![0f32; t * d],
        }
    }
}

pub struct LayerNormP {
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}
impl LayerNormP {
    fn load(m: &ModelFile, p: &str, d: usize) -> Result<Self> {
        Ok(Self {
            g: load_f32_shaped(m, &format!("{p}.weight"), &[d])?,
            b: load_f32_shaped(m, &format!("{p}.bias"), &[d])?,
        })
    }
    pub fn forward(&self, x: &[f32], t: usize, y: &mut [f32]) {
        crate::tensor::layer_norm(x, t, self.g.len(), &self.g, &self.b, y)
    }
}

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

    /// `x += 0.5 · linear2(silu(linear1(norm(x))))`, fused. Returns the norm output for tracing (f32 path only).
    fn forward_fused(
        &self,
        norm: &LayerNormP,
        x: &mut [f32],
        t: usize,
        ws: &mut Workspace,
    ) -> Option<Vec<f32>> {
        let a = self.w1.prepare_normed(x, t, &norm.g, &norm.b);
        let traced = a.as_f32().map(|v| v.to_vec());
        let hbuf = &mut ws.hbuf[..t * self.h];
        self.w1.forward_prepared_ep(&a, t, hbuf, Epilogue::Silu);
        let a2 = self.w2.prepare(hbuf, t);
        self.w2
            .forward_prepared_ep(&a2, t, x, Epilogue::AddScaled(0.5));
        traced
    }
}

pub struct Attention {
    wqkv: TernaryLinear, // [3d, d]: q | k | v stacked
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
        let wqkv = TernaryLinear::load_stacked(
            m,
            &[
                &format!("{p}.q_proj"),
                &format!("{p}.k_proj"),
                &format!("{p}.v_proj"),
            ],
            prec,
        )?;
        let wo = TernaryLinear::load(m, &format!("{p}.o_proj"), prec)?;
        let wp = TernaryLinear::load(m, &format!("{p}.relative_k_proj"), prec)?;
        let d = wo.out_features();
        let dh = d / heads;
        let bias_u = load_f32_shaped(m, &format!("{p}.bias_u"), &[heads, dh])?;
        let bias_v = load_f32_shaped(m, &format!("{p}.bias_v"), &[heads, dh])?;
        Ok(Self {
            wqkv,
            wo,
            wp,
            bias_u,
            bias_v,
            d,
            heads,
            dh,
        })
    }

    /// `x += o_proj(attention(norm(x)))`, fused. Returns the norm output for tracing (f32 path only).
    fn forward_fused(
        &self,
        norm: &LayerNormP,
        x: &mut [f32],
        t: usize,
        pos_emb: &[f32],
        ws: &mut Workspace,
    ) -> Option<Vec<f32>> {
        let (d, h, dh) = (self.d, self.heads, self.dh);
        let a = self.wqkv.prepare_normed(x, t, &norm.g, &norm.b);
        let traced = a.as_f32().map(|v| v.to_vec());
        let qkv = &mut ws.qkv[..t * 3 * d];
        self.wqkv.forward_prepared_ep(&a, t, qkv, Epilogue::Store);
        let np = 2 * t - 1;
        let mut rel_k = vec![0f32; np * d];
        self.wp.forward(pos_emb, np, &mut rel_k);
        let scaling = 1.0 / (dh as f32).sqrt();
        if self.wqkv.is_int8() && attn_int8() {
            let Workspace {
                qkv,
                heads,
                s1,
                s2,
                ctx,
                ..
            } = ws;
            let t0 = std::time::Instant::now();
            let t16 = t.div_ceil(16) * 16;
            let vh_t = &mut heads[3 * t * d + np * d..3 * t * d + np * d + d * t16];
            attn_core_int8(
                &qkv[..t * 3 * d],
                &rel_k,
                t,
                d,
                h,
                dh,
                &self.bias_u,
                &self.bias_v,
                scaling,
                vh_t,
                s1,
                s2,
                &mut ctx[..t * d],
            );
            ATTN_CORE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let a_o = self.wo.prepare(&ctx[..t * d], t);
            self.wo
                .forward_prepared_ep(&a_o, t, x, Epilogue::AddScaled(1.0));
            return traced;
        }
        let t0 = std::time::Instant::now();
        // per-head contiguous operands in the workspace: qu/qv/kh [h][t,dh], rel [h][np,dh], vᵀ [h][dh,t]
        let (qu, rest) = ws.heads.split_at_mut(t * d);
        let (qv, rest) = rest.split_at_mut(t * d);
        let (kh, rest) = rest.split_at_mut(t * d);
        let (rel_h, vh_t) = rest.split_at_mut(np * d);
        let vh_t = &mut vh_t[..d * t];
        for head in 0..h {
            let (bu, bv) = (
                &self.bias_u[head * dh..(head + 1) * dh],
                &self.bias_v[head * dh..(head + 1) * dh],
            );
            let base = head * t * dh;
            for i in 0..t {
                let row = &qkv[i * 3 * d..(i + 1) * 3 * d];
                let (qi, ki, vi) = (
                    &row[head * dh..(head + 1) * dh],
                    &row[d + head * dh..d + (head + 1) * dh],
                    &row[2 * d + head * dh..2 * d + (head + 1) * dh],
                );
                for c in 0..dh {
                    qu[base + i * dh + c] = qi[c] + bu[c];
                    qv[base + i * dh + c] = qi[c] + bv[c];
                    kh[base + i * dh + c] = ki[c];
                    vh_t[head * dh * t + c * t + i] = vi[c];
                }
            }
            for p in 0..np {
                rel_h[head * np * dh + p * dh..head * np * dh + (p + 1) * dh]
                    .copy_from_slice(&rel_k[p * d + head * dh..p * d + (head + 1) * dh]);
            }
        }
        // one task per (head, block of 4 query rows): content scores, banded position scores, softmax, context
        use rayon::prelude::*;
        let n_blocks = t.div_ceil(4);
        let ctx = &mut ws.ctx[..t * d];
        let ctxp = ctx.as_mut_ptr() as usize;
        let (qu, qv, kh, rel_h, vh_t) = (&*qu, &*qv, &*kh, &*rel_h, &*vh_t);
        (0..h * n_blocks).into_par_iter().for_each(|item| {
            let (head, blk) = (item / n_blocks, item % n_blocks);
            let (i0, rows) = (blk * 4, (t - blk * 4).min(4));
            let base = head * t * dh;
            let mut s1 = vec![0f32; 4 * t];
            crate::tensor::tile_f32(
                &qu[base + i0 * dh..],
                rows,
                dh,
                &kh[base..base + t * dh],
                None,
                t,
                &mut s1,
            );
            let p0 = t - 1 - (i0 + rows - 1);
            let pn = (2 * t - 2 - i0) - p0 + 1;
            let mut s2 = vec![0f32; 4 * pn];
            crate::tensor::tile_f32(
                &qv[base + i0 * dh..],
                rows,
                dh,
                &rel_h[head * np * dh + p0 * dh..head * np * dh + (p0 + pn) * dh],
                None,
                pn,
                &mut s2,
            );
            let mut probs = vec![0f32; 4 * t];
            for r in 0..rows {
                let i = i0 + r;
                let row = &mut probs[r * t..(r + 1) * t];
                let mut max = f32::NEG_INFINITY;
                for j in 0..t {
                    let p = t - 1 - i + j - p0;
                    let sc = (s1[r * t + j] + s2[r * pn + p]) * scaling;
                    row[j] = sc;
                    max = max.max(sc);
                }
                let mut sum = 0f32;
                for sv in row.iter_mut() {
                    *sv = (*sv - max).exp();
                    sum += *sv;
                }
                let inv = 1.0 / sum;
                row.iter_mut().for_each(|sv| *sv *= inv);
            }
            let mut out = vec![0f32; 4 * dh];
            crate::tensor::tile_f32(
                &probs,
                rows,
                t,
                &vh_t[head * dh * t..(head + 1) * dh * t],
                None,
                dh,
                &mut out,
            );
            // SAFETY: this task writes only ctx[i, head*dh..(head+1)*dh] for its rows; tasks are disjoint.
            let ctx = unsafe { std::slice::from_raw_parts_mut(ctxp as *mut f32, t * d) };
            for r in 0..rows {
                ctx[(i0 + r) * d + head * dh..(i0 + r) * d + (head + 1) * dh]
                    .copy_from_slice(&out[r * dh..(r + 1) * dh]);
            }
        });
        ATTN_CORE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let a_o = self.wo.prepare(ctx, t);
        self.wo
            .forward_prepared_ep(&a_o, t, x, Epilogue::AddScaled(1.0));
        traced
    }
}

/// Time spent in the attention core (scores, softmax, context), for the `PRX_PROFILE` line.
pub static ATTN_CORE_NS: AtomicU64 = AtomicU64::new(0);

/// `PRX_ATTN_CTX=f32` keeps the context product (probabilities × v) in f32 inside the int8 core.
fn attn_ctx_int8() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !matches!(std::env::var("PRX_ATTN_CTX").as_deref(), Ok("f32")))
}

/// `PRX_ATTN=f32` keeps the f32 attention core on the int8 path (A/B and diagnostics).
fn attn_int8() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !matches!(std::env::var("PRX_ATTN").as_deref(), Ok("f32")))
}

/// Attention core on int8 scores (PLAN 3c.8). Per head, the query operands `q + bias_u` / `q + bias_v` are quantised
/// per token (one group = the head dim) and the keys / relative positions per row as the *weight* operand, so the
/// content scores and the banded position scores run on the int8 GEMM tiles (`gemm_rows_range`). Softmax and the
/// context product stay f32. One task per (head, block of 8 query rows); the full-width score planes `s1`/`s2`
/// live in the workspace and each task writes only its own token rows.
#[allow(clippy::too_many_arguments)]
fn attn_core_int8(
    qkv: &[f32],
    rel_k: &[f32],
    t: usize,
    d: usize,
    h: usize,
    dh: usize,
    bias_u: &[f32],
    bias_v: &[f32],
    scaling: f32,
    vh_t: &mut [f32],
    s1: &mut [f32],
    s2: &mut [f32],
    ctx: &mut [f32],
) {
    use crate::kernels::gemm_rows_range;
    use rayon::prelude::*;
    let layout = crate::kernels::kernel().int8_layout();
    let np = 2 * t - 1;
    let (t2, np2) = (t.div_ceil(2) * 2, np.div_ceil(2) * 2);
    let n_pad = t.div_ceil(8) * 8;
    let d3 = 3 * d;
    let ctx_int8 = attn_ctx_int8();
    // keys of the context product are padded to a multiple of 16 (one int8 group per probability row)
    let t16 = t.div_ceil(16) * 16;
    enum Op {
        Act(QuantAct),
        Mat(TernaryMatrix),
    }
    // one task per (head, operand): q+bias_u, q+bias_v, k, rel, and v (int8 vᵀ, or the f32 transpose)
    let vtp = vh_t.as_mut_ptr() as usize;
    let mut built: Vec<Option<Op>> = (0..h * 5)
        .into_par_iter()
        .map(|item| {
            let (head, kind) = (item / 5, item % 5);
            let q_at = |i: usize| &qkv[i * d3 + head * dh..i * d3 + (head + 1) * dh];
            let with_bias = |b: &[f32]| {
                QuantAct::from_rows(t, dh, dh, layout, |i, row| {
                    for ((o, q), bb) in row.iter_mut().zip(q_at(i)).zip(b) {
                        *o = q + bb;
                    }
                })
            };
            match kind {
                0 => Some(Op::Act(with_bias(&bias_u[head * dh..(head + 1) * dh]))),
                1 => Some(Op::Act(with_bias(&bias_v[head * dh..(head + 1) * dh]))),
                2 => Some(Op::Mat(TernaryMatrix::from_f32_rows(
                    t2,
                    dh,
                    dh,
                    layout,
                    |j, row| {
                        if j < t {
                            row.copy_from_slice(
                                &qkv[j * d3 + d + head * dh..j * d3 + d + (head + 1) * dh],
                            );
                        } else {
                            row.fill(0.0);
                        }
                    },
                ))),
                3 => Some(Op::Mat(TernaryMatrix::from_f32_rows(
                    np2,
                    dh,
                    dh,
                    layout,
                    |p, row| {
                        if p < np {
                            row.copy_from_slice(&rel_k[p * d + head * dh..p * d + (head + 1) * dh]);
                        } else {
                            row.fill(0.0);
                        }
                    },
                ))),
                _ if ctx_int8 => Some(Op::Mat(TernaryMatrix::from_f32_rows(
                    dh,
                    t16,
                    t16,
                    layout,
                    |c, row| {
                        for (i, slot) in row[..t].iter_mut().enumerate() {
                            *slot = qkv[i * d3 + 2 * d + head * dh + c];
                        }
                        row[t..].fill(0.0);
                    },
                ))),
                _ => {
                    // SAFETY: head `head` owns vh_t[head·dh·t16 .. (head+1)·dh·t16]; tasks are disjoint.
                    let vt = unsafe {
                        std::slice::from_raw_parts_mut(
                            (vtp as *mut f32).add(head * dh * t16),
                            dh * t16,
                        )
                    };
                    for c in 0..dh {
                        vt[c * t16 + t..(c + 1) * t16].fill(0.0);
                    }
                    for i in 0..t {
                        let v = &qkv[i * d3 + 2 * d + head * dh..i * d3 + 2 * d + (head + 1) * dh];
                        for (c, vc) in v.iter().enumerate() {
                            vt[c * t16 + i] = *vc;
                        }
                    }
                    None
                }
            }
        })
        .collect();
    struct HeadOps {
        qu: QuantAct,
        qv: QuantAct,
        k: TernaryMatrix,
        rel: TernaryMatrix,
        v: Option<TernaryMatrix>,
    }
    let ops: Vec<HeadOps> = (0..h)
        .map(|head| {
            let mut take = |kind: usize| built[head * 5 + kind].take();
            let (Some(Op::Act(qu)), Some(Op::Act(qv)), Some(Op::Mat(k)), Some(Op::Mat(rel))) =
                (take(0), take(1), take(2), take(3))
            else {
                unreachable!()
            };
            let v = match take(4) {
                Some(Op::Mat(v)) => Some(v),
                _ => None,
            };
            HeadOps { qu, qv, k, rel, v }
        })
        .collect();
    let vh_t = &*vh_t;
    let nb = t.div_ceil(8);
    let (s1p, s2p, ctxp) = (
        s1.as_mut_ptr() as usize,
        s2.as_mut_ptr() as usize,
        ctx.as_mut_ptr() as usize,
    );
    thread_local! {
        static SCR: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    (0..h * nb).into_par_iter().for_each(|item| {
        let (head, blk) = (item / nb, item % nb);
        let (i0, rows) = (blk * 8, (t - blk * 8).min(8));
        let op = &ops[head];
        // SAFETY: a task writes only token rows [i0, i0+rows) of its head's s1/s2 planes and of ctx (columns of its
        // head); tasks are disjoint. The planes are sized [h][n_pad][t2] / [h][n_pad][np2] by `Workspace::new`.
        let s1h = unsafe {
            std::slice::from_raw_parts_mut((s1p as *mut f32).add(head * n_pad * t2), n_pad * t2)
        };
        let s2h = unsafe {
            std::slice::from_raw_parts_mut((s2p as *mut f32).add(head * n_pad * np2), n_pad * np2)
        };
        gemm_rows_range(&op.k, &op.qu, s1h, 0, t2, i0, i0 + rows, Epilogue::Store);
        // position band for rows [i0, i0+rows): p = t-1-i+j ∈ [t-i0-rows, 2t-1-i0), widened to even row pairs
        let p0 = (t - i0 - rows) & !1;
        let p1 = (2 * t - 1 - i0).div_ceil(2) * 2;
        gemm_rows_range(&op.rel, &op.qv, s2h, p0, p1, i0, i0 + rows, Epilogue::Store);
        SCR.with(|c| {
            let mut buf = c.borrow_mut();
            if buf.len() < 8 * t16 + 8 * dh {
                buf.resize(8 * t16 + 8 * dh, 0.0);
            }
            let (probs, out) = buf.split_at_mut(8 * t16);
            // unnormalised exp(score − max) per row (max element is exactly 1) and the row sums
            let mut inv_sum = [0f32; 8];
            for r in 0..rows {
                let i = i0 + r;
                let s1r = &s1h[i * t2..i * t2 + t];
                let s2r = &s2h[i * np2 + (t - 1 - i)..i * np2 + (t - 1 - i) + t];
                let row = &mut probs[r * t16..r * t16 + t];
                for ((sc, a), b) in row.iter_mut().zip(s1r).zip(s2r) {
                    *sc = a + b;
                }
                inv_sum[r] = 1.0 / crate::tensor::exp_row_scaled(row, scaling);
                probs[r * t16 + t..(r + 1) * t16].fill(0.0);
            }
            if let Some(v) = &op.v {
                // int8 probabilities (scale 1/127 per row, 1/Σ folded into the scale) × int8 vᵀ [dh, t16]
                let mut qp = QuantAct::from_rows(rows, t16, t16, layout, |r, row| {
                    row.copy_from_slice(&probs[r * t16..(r + 1) * t16])
                });
                for r in 0..rows {
                    qp.scales[r] *= inv_sum[r];
                    qp.scales_t[r] *= inv_sum[r];
                }
                gemm_rows_range(v, &qp, out, 0, dh, 0, rows, Epilogue::Store);
            } else {
                for r in 0..rows {
                    probs[r * t16..r * t16 + t]
                        .iter_mut()
                        .for_each(|p| *p *= inv_sum[r]);
                }
                let vt = &vh_t[head * dh * t16..(head + 1) * dh * t16];
                let mut r = 0;
                while r < rows {
                    let k = (rows - r).min(4);
                    // rows of `probs` are t16 apart; the tile reads `in_f`-strided rows, so hand it t16 with zero tails
                    crate::tensor::tile_f32(
                        &probs[r * t16..],
                        k,
                        t16,
                        vt,
                        None,
                        dh,
                        &mut out[r * dh..],
                    );
                    r += k;
                }
            }
            // SAFETY: see above.
            let ctx = unsafe { std::slice::from_raw_parts_mut(ctxp as *mut f32, t * d) };
            for r in 0..rows {
                ctx[(i0 + r) * d + head * dh..(i0 + r) * d + (head + 1) * dh]
                    .copy_from_slice(&out[r * dh..(r + 1) * dh]);
            }
        });
    });
}

pub struct ConvModule {
    pw1: TernaryLinear, // [2d, d]
    dw: Vec<f32>,       // [k, d] (tap-major) with BatchNorm folded in (scale)
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
        // tap-major [k, d]: the time loop then runs one contiguous axpy over channels per tap
        let mut dw_t = vec![0f32; k * d];
        for c in 0..d {
            for kk in 0..k {
                dw_t[kk * d + c] = dw[c * k + kk];
            }
        }
        Ok(Self {
            pw1,
            dw: dw_t,
            dw_bias,
            pw2,
            d,
            k,
        })
    }

    /// `x += pw2(silu(bn(dw(glu(pw1(norm(x)))))))`, fused. Returns the norm output for tracing (f32 path only).
    fn forward_fused(
        &self,
        norm: &LayerNormP,
        x: &mut [f32],
        t: usize,
        ws: &mut Workspace,
    ) -> Option<Vec<f32>> {
        let (d, k) = (self.d, self.k);
        let pad = (k - 1) / 2;
        let a_in = self.pw1.prepare_normed(x, t, &norm.g, &norm.b);
        let traced = a_in.as_f32().map(|v| v.to_vec());
        let g = &mut ws.g[..t * 2 * d];
        self.pw1.forward_prepared_ep(&a_in, t, g, Epilogue::Store);
        // GLU over channels: a * sigmoid(b), a = first half — parallel over time rows
        let a = &mut ws.a[..t * d];
        {
            let g = &*g;
            par_rows(a, d, |i, row| {
                let gi = &g[i * 2 * d..(i + 1) * 2 * d];
                crate::tensor::glu_row(&gi[..d], &gi[d..], row);
            });
        }
        // depthwise conv over time (k taps), folded BN, SiLU — parallel over output rows
        let hb = &mut ws.hb[..t * d];
        {
            let (dw, dw_bias, a) = (&self.dw, &self.dw_bias, &*a);
            par_rows_cost(hb, d, k, |i, row| {
                row.copy_from_slice(dw_bias);
                for kk in 0..k {
                    let j = i + kk;
                    if j >= pad && j - pad < t {
                        let (w, aj) = (
                            &dw[kk * d..(kk + 1) * d],
                            &a[(j - pad) * d..(j - pad + 1) * d],
                        );
                        for ((r, w), x) in row.iter_mut().zip(w).zip(aj) {
                            *r += w * x;
                        }
                    }
                }
                crate::tensor::silu_row(row);
            });
        }
        let a2 = self.pw2.prepare(hb, t);
        self.pw2
            .forward_prepared_ep(&a2, t, x, Epilogue::AddScaled(1.0));
        traced
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

/// Intermediate outputs of one layer (for golden comparison, f32 path). Sublayer outputs are reconstructed from the
/// residual stream (`out = x_after − x_before`, ÷0.5 for the feed-forwards), since the fused path never stores them.
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

fn diff(after: &[f32], before: &[f32], scale: f32) -> Vec<f32> {
    after
        .iter()
        .zip(before)
        .map(|(a, b)| (a - b) * scale)
        .collect()
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
        let mut ws = Workspace::new(t, self.d, self.ff1.h, self.attn.heads);
        self.forward_profiled(x, t, pos_emb, trace, None, &mut ws)
    }

    pub fn forward_profiled(
        &self,
        x: &mut [f32],
        t: usize,
        pos_emb: &[f32],
        trace: Option<&mut LayerTrace>,
        mut prof: Option<&mut Profile>,
        ws: &mut Workspace,
    ) {
        debug_assert!(ws.t >= t);
        let d = self.d;
        let mut tr: Option<&mut LayerTrace> = trace;
        let mut tick = std::time::Instant::now();
        let mut lap = |field: fn(&mut Profile) -> &mut f64, prof: &mut Option<&mut Profile>| {
            if let Some(p) = prof.as_deref_mut() {
                *field(p) += tick.elapsed().as_secs_f64();
                tick = std::time::Instant::now();
            }
        };
        let before = tr.as_ref().map(|_| x.to_vec());
        let n = self.ff1.forward_fused(&self.norm_ff1, x, t, ws);
        if let (Some(tr), Some(b)) = (tr.as_mut(), before.as_ref()) {
            tr.norm_ff1 = n.unwrap_or_default();
            tr.ff1 = diff(x, b, 2.0);
        }
        lap(|p| &mut p.ff, &mut prof);
        let before = tr.as_ref().map(|_| x.to_vec());
        let n = self.attn.forward_fused(&self.norm_attn, x, t, pos_emb, ws);
        if let (Some(tr), Some(b)) = (tr.as_mut(), before.as_ref()) {
            tr.norm_attn = n.unwrap_or_default();
            tr.attn = diff(x, b, 1.0);
        }
        lap(|p| &mut p.attn, &mut prof);
        let before = tr.as_ref().map(|_| x.to_vec());
        let n = self.conv.forward_fused(&self.norm_conv, x, t, ws);
        if let (Some(tr), Some(b)) = (tr.as_mut(), before.as_ref()) {
            tr.norm_conv = n.unwrap_or_default();
            tr.conv = diff(x, b, 1.0);
        }
        lap(|p| &mut p.conv, &mut prof);
        let before = tr.as_ref().map(|_| x.to_vec());
        self.ff2.forward_fused(&self.norm_ff2, x, t, ws);
        if let (Some(tr), Some(b)) = (tr.as_mut(), before.as_ref()) {
            tr.ff2 = diff(x, b, 2.0);
        }
        lap(|p| &mut p.ff, &mut prof);
        let s = &mut ws.scratch[..t * d];
        self.norm_out.forward(x, t, s);
        x.copy_from_slice(s);
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
        let mut ws = Workspace::new(te, self.d, self.layers[0].ff1.h, self.layers[0].attn.heads);
        let mut l0 = layer0_trace;
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_profiled(
                &mut x,
                te,
                &pos,
                if i == 0 { l0.take() } else { None },
                if profiling { Some(&mut prof) } else { None },
                &mut ws,
            );
            if let Some(lo) = layer_outs.as_deref_mut() {
                lo.push(x.clone());
            }
        }
        if profiling {
            let total = t0.elapsed().as_secs_f64();
            use crate::ternary_linear::{GEMM_NS, QUANT_NS};
            let (q, g) = (
                QUANT_NS.swap(0, std::sync::atomic::Ordering::Relaxed),
                GEMM_NS.swap(0, std::sync::atomic::Ordering::Relaxed),
            );
            let core = ATTN_CORE_NS.swap(0, Ordering::Relaxed);
            eprintln!("encoder profile ({:?}, T_e={te}): total {total:.3}s · subsampling {:.3} · ff {:.3} · attn {:.3} (core {:.3}) · conv {:.3} · norm {:.3} · [ternary gemm {:.3}, quantize {:.3}]", self.precision, prof.subsampling, prof.ff, prof.attn, core as f64 * 1e-9, prof.conv, prof.norm, g as f64 * 1e-9, q as f64 * 1e-9);
        }
        (x, te)
    }
}
