//! Dense tensor helpers: load F16/F32 safetensors into f32 vectors; row-major ops with a scalar
//! (auto-vectorisable) inner loop and coarse std::thread parallelism. Real kernels come in Phase 3.
use half::f16;
use safetensors::Dtype;

use crate::format::ModelFile;
use crate::{Error, Result};

pub fn load_f32(m: &ModelFile, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let (info, bytes) = m.tensor(name)?;
    let v = match info.dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        d => return Err(Error::Format(format!("{name}: cannot load {d:?} as f32"))),
    };
    Ok((info.shape, v))
}

pub fn load_f32_shaped(m: &ModelFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let (s, v) = load_f32(m, name)?;
    if s != shape {
        return Err(Error::Format(format!(
            "{name}: shape {s:?}, expected {shape:?}"
        )));
    }
    Ok(v)
}

pub fn threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `y[n, out] = x[n, in] · W[out, in]^T + b[out]`. Parallel (rayon) when the work is large enough; single-threaded for
/// tiny calls. On aarch64 the inner product uses a 4 rows × 4 tokens NEON register tile.
pub fn linear(
    x: &[f32],
    n: usize,
    in_f: usize,
    w: &[f32],
    b: Option<&[f32]>,
    out_f: usize,
    y: &mut [f32],
) {
    use rayon::prelude::*;
    debug_assert_eq!(x.len(), n * in_f);
    debug_assert_eq!(w.len(), out_f * in_f);
    debug_assert_eq!(y.len(), n * out_f);
    if n == 1 {
        gemv(x, in_f, w, b, out_f, y);
        return;
    }
    let small = n * out_f * in_f < 1 << 20;
    // tiles of 4 tokens; each task owns 4 token rows of y (all output columns)
    let tile = |xi: &[f32], yi: &mut [f32]| tile_f32(xi, xi.len() / in_f, in_f, w, b, out_f, yi);
    if small || n < 8 {
        for (xi, yi) in x.chunks(4 * in_f).zip(y.chunks_mut(4 * out_f)) {
            tile(xi, yi);
        }
    } else {
        y.par_chunks_mut(4 * out_f)
            .zip(x.par_chunks(4 * in_f))
            .for_each(|(yi, xi)| tile(xi, yi));
    }
}

/// `y[toks, out] = x[toks, in] · W[out, in]ᵀ (+ b)` for 1..=4 token rows: the f32 register tile (NEON on aarch64).
pub fn tile_f32(
    x: &[f32],
    toks: usize,
    in_f: usize,
    w: &[f32],
    b: Option<&[f32]>,
    out_f: usize,
    y: &mut [f32],
) {
    debug_assert!((1..=4).contains(&toks) && x.len() >= toks * in_f && y.len() >= toks * out_f);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; the tile reads whole 4-float groups then a scalar tail.
        unsafe { tile_rows_neon(x, toks, in_f, w, b, out_f, y) };
        return;
    }
    #[allow(unreachable_code)]
    tile_rows_portable(x, toks, in_f, w, b, out_f, y)
}

// Portable mirrors of the NEON reductions below: same lane split, fused multiply-adds, `vaddvq_f32` order
// ((l0 + l1) + (l2 + l3)) and tails, so x86 and aarch64 give the same bits (checked by `portable_tests`).

#[inline]
fn hsum4(a: [f32; 4]) -> f32 {
    (a[0] + a[1]) + (a[2] + a[3])
}

/// Mirror of `tile_rows_neon`: per (row, token) four fused lanes over 4-column steps, a separate scalar tail.
pub(crate) fn tile_rows_portable(
    x: &[f32],
    toks: usize,
    in_f: usize,
    w: &[f32],
    b: Option<&[f32]>,
    out_f: usize,
    y: &mut [f32],
) {
    let c4 = in_f / 4 * 4;
    for o in 0..out_f {
        let wr = &w[o * in_f..(o + 1) * in_f];
        for tk in 0..toks {
            let xr = &x[tk * in_f..(tk + 1) * in_f];
            let mut acc = [0f32; 4];
            for c in (0..c4).step_by(4) {
                for l in 0..4 {
                    acc[l] = wr[c + l].mul_add(xr[c + l], acc[l]);
                }
            }
            let mut tail = 0f32;
            for c in c4..in_f {
                tail += wr[c] * xr[c];
            }
            y[tk * out_f + o] = hsum4(acc) + tail + b.map_or(0.0, |b| b[o]);
        }
    }
}

/// Mirror of the 2 × 4-lane dot in `gemv_neon` / `gemv_f16_neon`: fused lanes over 8-column steps, lane-wise
/// `a + b`, horizontal add, then the tail added in order.
#[inline]
#[allow(clippy::needless_range_loop)]
fn dot_lanes8(x: &[f32], w: impl Fn(usize) -> f32) -> f32 {
    let n8 = x.len() / 8 * 8;
    let (mut a, mut bb) = ([0f32; 4], [0f32; 4]);
    for c in (0..n8).step_by(8) {
        for l in 0..4 {
            a[l] = x[c + l].mul_add(w(c + l), a[l]);
            bb[l] = x[c + 4 + l].mul_add(w(c + 4 + l), bb[l]);
        }
    }
    let mut r = hsum4(std::array::from_fn(|l| a[l] + bb[l]));
    for c in n8..x.len() {
        r += x[c] * w(c);
    }
    r
}

/// Mirror of `gemv_neon` (4-row blocks; leftover rows use `dot`).
pub(crate) fn gemv_portable(
    x: &[f32],
    in_f: usize,
    w: &[f32],
    b: Option<&[f32]>,
    out_f: usize,
    y: &mut [f32],
) {
    let o4 = out_f / 4 * 4;
    for o in 0..out_f {
        let wr = &w[o * in_f..(o + 1) * in_f];
        let r = if o < o4 {
            dot_lanes8(x, |c| wr[c])
        } else {
            dot(x, wr)
        };
        y[o] = r + b.map_or(0.0, |b| b[o]);
    }
}

/// Mirror of `gemv_f16_neon` (`in_f % 8 == 0`; leftover rows are a plain in-order sum).
pub(crate) fn gemv_f16_portable(
    x: &[f32],
    in_f: usize,
    w: &[u16],
    b: Option<&[f32]>,
    out_f: usize,
    y: &mut [f32],
) {
    let o4 = out_f / 4 * 4;
    for o in 0..out_f {
        let wr = &w[o * in_f..(o + 1) * in_f];
        let r = if o < o4 {
            dot_lanes8(x, |c| f16::from_bits(wr[c]).to_f32())
        } else {
            x.iter().zip(wr).fold(0f32, |acc, (xv, &wv)| {
                acc + xv * f16::from_bits(wv).to_f32()
            })
        };
        y[o] = r + b.map_or(0.0, |b| b[o]);
    }
}

/// Mirror of `row_mean_var_neon`.
pub(crate) fn row_mean_var_portable(xi: &[f32]) -> (f32, f32) {
    let d = xi.len();
    let n8 = d / 8 * 8;
    let (mut s0, mut s1) = ([0f32; 4], [0f32; 4]);
    for c in (0..n8).step_by(8) {
        for l in 0..4 {
            s0[l] += xi[c + l];
            s1[l] += xi[c + 4 + l];
        }
    }
    let mut sum = hsum4(std::array::from_fn(|l| s0[l] + s1[l]));
    for v in &xi[n8..] {
        sum += v;
    }
    let mean = sum / d as f32;
    let (mut q0, mut q1) = ([0f32; 4], [0f32; 4]);
    for c in (0..n8).step_by(8) {
        for l in 0..4 {
            let (a, e) = (xi[c + l] - mean, xi[c + 4 + l] - mean);
            q0[l] = a.mul_add(a, q0[l]);
            q1[l] = e.mul_add(e, q1[l]);
        }
    }
    let mut sq = hsum4(std::array::from_fn(|l| q0[l] + q1[l]));
    for v in &xi[n8..] {
        let a = v - mean;
        sq += a * a;
    }
    (mean, sq / d as f32)
}

/// Up to 4 token rows × all outputs, 4 outputs at a time: 16 float32x4 accumulators, loads shared across the tile.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)]
unsafe fn tile_rows_neon(
    x: &[f32],
    toks: usize,
    in_f: usize,
    w: &[f32],
    b: Option<&[f32]>,
    out_f: usize,
    y: &mut [f32],
) {
    use std::arch::aarch64::*;
    let xp: [*const f32; 4] = std::array::from_fn(|c| x.as_ptr().add(c.min(toks - 1) * in_f));
    let mut o = 0;
    while o < out_f {
        let rows = (out_f - o).min(4);
        let wp: [*const f32; 4] =
            std::array::from_fn(|r| w.as_ptr().add((o + r.min(rows - 1)) * in_f));
        let z = vdupq_n_f32(0.0);
        let mut acc = [[z; 4]; 4]; // [row][token]
        let mut c = 0;
        while c + 4 <= in_f {
            let xv = [
                vld1q_f32(xp[0].add(c)),
                vld1q_f32(xp[1].add(c)),
                vld1q_f32(xp[2].add(c)),
                vld1q_f32(xp[3].add(c)),
            ];
            for r in 0..4 {
                let wv = vld1q_f32(wp[r].add(c));
                acc[r][0] = vfmaq_f32(acc[r][0], wv, xv[0]);
                acc[r][1] = vfmaq_f32(acc[r][1], wv, xv[1]);
                acc[r][2] = vfmaq_f32(acc[r][2], wv, xv[2]);
                acc[r][3] = vfmaq_f32(acc[r][3], wv, xv[3]);
            }
            c += 4;
        }
        let mut tail = [[0f32; 4]; 4];
        while c < in_f {
            for r in 0..4 {
                let wv = *wp[r].add(c);
                for tk in 0..4 {
                    tail[r][tk] += wv * *xp[tk].add(c);
                }
            }
            c += 1;
        }
        for r in 0..rows {
            let bias = b.map_or(0.0, |b| b[o + r]);
            for tk in 0..toks {
                *y.get_unchecked_mut(tk * out_f + o + r) =
                    vaddvq_f32(acc[r][tk]) + tail[r][tk] + bias;
            }
        }
        o += rows;
    }
}

/// Single-token matrix-vector product with 4-row register blocking (decoder steps: LSTM gates, joint head).
pub fn gemv(x: &[f32], in_f: usize, w: &[f32], b: Option<&[f32]>, out_f: usize, y: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64.
        unsafe { gemv_neon(x, in_f, w, b, out_f, y) };
        return;
    }
    #[allow(unreachable_code)]
    gemv_portable(x, in_f, w, b, out_f, y)
}

#[cfg(target_arch = "aarch64")]
unsafe fn gemv_neon(
    x: &[f32],
    in_f: usize,
    w: &[f32],
    b: Option<&[f32]>,
    out_f: usize,
    y: &mut [f32],
) {
    use std::arch::aarch64::*;
    debug_assert_eq!(in_f % 4, 0);
    let mut o = 0;
    while o + 4 <= out_f {
        let (w0, w1, w2, w3) = (
            w.as_ptr().add(o * in_f),
            w.as_ptr().add((o + 1) * in_f),
            w.as_ptr().add((o + 2) * in_f),
            w.as_ptr().add((o + 3) * in_f),
        );
        let (mut a0, mut a1, mut a2, mut a3) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
        let (mut b0, mut b1, mut b2, mut b3) = (
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
            vdupq_n_f32(0.0),
        );
        let mut c = 0;
        while c + 8 <= in_f {
            let xa = vld1q_f32(x.as_ptr().add(c));
            let xb = vld1q_f32(x.as_ptr().add(c + 4));
            a0 = vfmaq_f32(a0, xa, vld1q_f32(w0.add(c)));
            b0 = vfmaq_f32(b0, xb, vld1q_f32(w0.add(c + 4)));
            a1 = vfmaq_f32(a1, xa, vld1q_f32(w1.add(c)));
            b1 = vfmaq_f32(b1, xb, vld1q_f32(w1.add(c + 4)));
            a2 = vfmaq_f32(a2, xa, vld1q_f32(w2.add(c)));
            b2 = vfmaq_f32(b2, xb, vld1q_f32(w2.add(c + 4)));
            a3 = vfmaq_f32(a3, xa, vld1q_f32(w3.add(c)));
            b3 = vfmaq_f32(b3, xb, vld1q_f32(w3.add(c + 4)));
            c += 8;
        }
        let mut r = [
            vaddvq_f32(vaddq_f32(a0, b0)),
            vaddvq_f32(vaddq_f32(a1, b1)),
            vaddvq_f32(vaddq_f32(a2, b2)),
            vaddvq_f32(vaddq_f32(a3, b3)),
        ];
        while c < in_f {
            for k in 0..4 {
                r[k] += x[c] * w[(o + k) * in_f + c];
            }
            c += 1;
        }
        for k in 0..4 {
            y[o + k] = r[k] + b.map_or(0.0, |b| b[o + k]);
        }
        o += 4;
    }
    while o < out_f {
        y[o] = dot(x, &w[o * in_f..(o + 1) * in_f]) + b.map_or(0.0, |b| b[o]);
        o += 1;
    }
}

/// Load an F16 tensor as raw half bits (for memory-bound single-token paths).
pub fn load_f16_bits(m: &ModelFile, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
    let (info, bytes) = m.tensor(name)?;
    if info.dtype != Dtype::F16 {
        return Err(Error::Format(format!(
            "{name}: expected F16, got {:?}",
            info.dtype
        )));
    }
    if info.shape != shape {
        return Err(Error::Format(format!(
            "{name}: shape {:?}, expected {shape:?}",
            info.shape
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Single-token GEMV with f16 weights (half the memory traffic of f32). `w`: `[out, in]` half bits.
/// Memory-bound per decoder step; splitting it across the pool costs more in dispatch than it saves (measured).
pub fn gemv_f16(x: &[f32], in_f: usize, w: &[u16], b: Option<&[f32]>, out_f: usize, y: &mut [f32]) {
    gemv_f16_part(x, in_f, w, b, out_f, y);
}

fn gemv_f16_part(
    x: &[f32],
    in_f: usize,
    w: &[u16],
    b: Option<&[f32]>,
    out_f: usize,
    y: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { gemv_f16_neon(x, in_f, w, b, out_f, y) };
        return;
    }
    #[allow(unreachable_code)]
    gemv_f16_portable(x, in_f, w, b, out_f, y)
}

#[cfg(target_arch = "aarch64")]
unsafe fn gemv_f16_neon(
    x: &[f32],
    in_f: usize,
    w: &[u16],
    b: Option<&[f32]>,
    out_f: usize,
    y: &mut [f32],
) {
    use std::arch::aarch64::*;
    use std::arch::asm;
    #[inline(always)]
    unsafe fn cvt_lo(h: uint16x8_t) -> float32x4_t {
        let out: float32x4_t;
        asm!("fcvtl {o:v}.4s, {i:v}.4h", o = out(vreg) out, i = in(vreg) h, options(pure, nomem, nostack));
        out
    }
    #[inline(always)]
    unsafe fn cvt_hi(h: uint16x8_t) -> float32x4_t {
        let out: float32x4_t;
        asm!("fcvtl2 {o:v}.4s, {i:v}.8h", o = out(vreg) out, i = in(vreg) h, options(pure, nomem, nostack));
        out
    }
    debug_assert_eq!(in_f % 8, 0);
    let mut o = 0;
    while o + 4 <= out_f {
        let wp = [
            w.as_ptr().add(o * in_f),
            w.as_ptr().add((o + 1) * in_f),
            w.as_ptr().add((o + 2) * in_f),
            w.as_ptr().add((o + 3) * in_f),
        ];
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut acc2 = [vdupq_n_f32(0.0); 4];
        let mut c = 0;
        while c < in_f {
            let xa = vld1q_f32(x.as_ptr().add(c));
            let xb = vld1q_f32(x.as_ptr().add(c + 4));
            for k in 0..4 {
                let h = vld1q_u16(wp[k].add(c));
                acc[k] = vfmaq_f32(acc[k], xa, cvt_lo(h));
                acc2[k] = vfmaq_f32(acc2[k], xb, cvt_hi(h));
            }
            c += 8;
        }
        for k in 0..4 {
            y[o + k] = vaddvq_f32(vaddq_f32(acc[k], acc2[k])) + b.map_or(0.0, |b| b[o + k]);
        }
        o += 4;
    }
    while o < out_f {
        let row = &w[o * in_f..(o + 1) * in_f];
        let mut acc = 0f32;
        for (xv, &wv) in x.iter().zip(row) {
            acc += xv * f16::from_bits(wv).to_f32();
        }
        y[o] = acc + b.map_or(0.0, |b| b[o]);
        o += 1;
    }
}

/// Minimum work (elements × cost) for row-parallel helpers to use the pool; below it, dispatch dominates.
pub const PAR_MIN_ELEMS: usize = 64 * 1024;

/// Effective threshold (`PRX_PAR_MIN` overrides, for tuning).
pub fn par_min_elems() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PRX_PAR_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PAR_MIN_ELEMS)
    })
}

/// Apply `f` to each row of a `[n, d]` buffer: in parallel when `elements × cost` is large enough, else sequentially.
/// `cost` ≈ operations per element relative to a LayerNorm (1 for norms/activations, 9 for the depthwise conv).
pub fn par_rows_cost<F: Fn(usize, &mut [f32]) + Sync>(
    buf: &mut [f32],
    d: usize,
    cost: usize,
    f: F,
) {
    use rayon::prelude::*;
    let n = buf.len() / d;
    if n >= 4 && buf.len() * cost >= par_min_elems() {
        buf.par_chunks_mut(d)
            .enumerate()
            .for_each(|(i, row)| f(i, row));
    } else {
        buf.chunks_mut(d).enumerate().for_each(|(i, row)| f(i, row));
    }
}

/// `par_rows_cost` with cost 1.
pub fn par_rows<F: Fn(usize, &mut [f32]) + Sync>(buf: &mut [f32], d: usize, f: F) {
    par_rows_cost(buf, d, 1, f)
}

/// LayerNorm over the last dim, eps 1e-5 (nn.LayerNorm default), biased variance. Parallel over rows.
pub fn layer_norm(x: &[f32], n: usize, d: usize, g: &[f32], b: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), n * d);
    par_rows(y, d, |i, yr| {
        layer_norm_row(&x[i * d..(i + 1) * d], g, b, yr)
    });
}

/// One LayerNorm row (`eps = 1e-5`, biased variance): `y = (x − mean) · rsqrt(var + eps) · g + b`. The two
/// reductions run on four NEON lanes (a scalar float sum is a latency-bound dependency chain, which made the
/// fused norm→quantise pass the slowest per-element loop of the encoder); the normalise loop auto-vectorises.
#[inline]
pub fn layer_norm_row(xi: &[f32], g: &[f32], b: &[f32], y: &mut [f32]) {
    let d = xi.len();
    let (mean, var) = row_mean_var(xi);
    let inv = 1.0 / (var + 1e-5).sqrt();
    for k in 0..d {
        y[k] = (xi[k] - mean) * inv * g[k] + b[k];
    }
}

/// Mean and biased variance of a row (two passes, 4-lane partial sums).
#[inline]
pub fn row_mean_var(xi: &[f32]) -> (f32, f32) {
    let d = xi.len();
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; the tails are scalar.
        return unsafe { row_mean_var_neon(xi, d) };
    }
    #[allow(unreachable_code)]
    {
        let _ = d;
        row_mean_var_portable(xi)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn row_mean_var_neon(xi: &[f32], d: usize) -> (f32, f32) {
    use std::arch::aarch64::*;
    let p = xi.as_ptr();
    let (mut s0, mut s1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let mut i = 0;
    while i + 8 <= d {
        s0 = vaddq_f32(s0, vld1q_f32(p.add(i)));
        s1 = vaddq_f32(s1, vld1q_f32(p.add(i + 4)));
        i += 8;
    }
    let mut sum = vaddvq_f32(vaddq_f32(s0, s1));
    while i < d {
        sum += *p.add(i);
        i += 1;
    }
    let mean = sum / d as f32;
    let vm = vdupq_n_f32(mean);
    let (mut q0, mut q1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    i = 0;
    while i + 8 <= d {
        let a = vsubq_f32(vld1q_f32(p.add(i)), vm);
        let c = vsubq_f32(vld1q_f32(p.add(i + 4)), vm);
        q0 = vfmaq_f32(q0, a, a);
        q1 = vfmaq_f32(q1, c, c);
        i += 8;
    }
    let mut sq = vaddvq_f32(vaddq_f32(q0, q1));
    while i < d {
        let a = *p.add(i) - mean;
        sq += a * a;
        i += 1;
    }
    (mean, sq / d as f32)
}

/// In-place SiLU over a `[n, d]` buffer, parallel over rows.
pub fn silu_inplace(buf: &mut [f32], d: usize) {
    par_rows(buf, d, |_, row| row.iter_mut().for_each(|v| *v = silu(*v)));
}

#[inline]
pub fn silu(v: f32) -> f32 {
    v / (1.0 + libm::expf(-v))
}

#[inline]
pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-v))
}

/// In-place softmax over one row: `row[j] = exp(row[j]·scale − max) / Σ`.
pub fn softmax_row_scaled(row: &mut [f32], scale: f32) {
    let inv = 1.0 / exp_row_scaled(row, scale);
    row.iter_mut().for_each(|v| *v *= inv);
}

/// In-place `row[j] = exp(row[j]·scale − max)` (the maximum element becomes exactly 1); returns the row sum.
/// NEON `exp` (range-reduced degree-6 polynomial, ~2 ulp) on aarch64, its scalar mirror `exp_poly` elsewhere and
/// in the tails, so every platform gets the same bits. Used by the int8 attention
/// core (PLAN 3c.8); the f32 reference path keeps scalar libm `exp`.
pub fn exp_row_scaled(row: &mut [f32], scale: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; tails are scalar.
        return unsafe { exp_row_neon(row, scale) };
    }
    #[allow(unreachable_code)]
    exp_row_scaled_portable(row, scale)
}

/// Mirror of `exp_row_neon`: `exp_poly` per element, 4 lane sums, (l0 + l1) + (l2 + l3), then the tail in order.
pub(crate) fn exp_row_scaled_portable(row: &mut [f32], scale: f32) -> f32 {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, v| m.max(v * scale));
    let mut lanes = [0f32; 4];
    let mut quads = row.chunks_exact_mut(4);
    for q in &mut quads {
        for (v, l) in q.iter_mut().zip(&mut lanes) {
            *v = exp_poly(*v * scale - max);
            *l += *v;
        }
    }
    let mut sum = hsum4(lanes);
    for v in quads.into_remainder() {
        *v = exp_poly(*v * scale - max);
        sum += *v;
    }
    sum
}

const EXP_LN2_HI: f32 = 0.693_145_75;
const EXP_LN2_LO: f32 = 1.428_606_8e-6;
const EXP_POLY: [f32; 6] = [1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0];

/// `exp(x)` by range reduction (`x = n·ln2 + r`, hi/lo split) and a degree-6 Horner polynomial, ~2 ulp; inputs
/// clamped to [−87, 88]. Scalar mirror of `exp_v`: the same operations (fused multiply-adds, round-to-even, exponent
/// construction) in the same order, so both give bit-identical results — the kernels' scalar == SIMD contract
/// extends to the SiLU epilogue.
#[inline]
pub fn exp_poly(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round_ties_even();
    let r = (-n).mul_add(EXP_LN2_HI, x);
    let r = (-n).mul_add(EXP_LN2_LO, r);
    let mut p: f32 = 1.0 / 720.0;
    for c in EXP_POLY {
        p = p.mul_add(r, c);
    }
    p * f32::from_bits(((n as i32 + 127) << 23) as u32)
}

/// `x / (1 + exp(−x))` with `exp_poly`; the SiLU used by the GEMM epilogue (scalar and SIMD paths agree bit for bit).
#[inline]
pub fn silu_fast(x: f32) -> f32 {
    x / (1.0 + exp_poly(-x))
}

/// `1 / (1 + exp(−x))` with `exp_poly` (conv-module GLU).
#[inline]
pub fn sigmoid_fast(x: f32) -> f32 {
    1.0 / (1.0 + exp_poly(-x))
}

/// Vector `exp_poly` (see there).
///
/// # Safety
/// NEON only (baseline on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn exp_v(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0)), vdupq_n_f32(88.0));
    let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)));
    // vfmsq(a, b, c) = a − b·c fused = fma(−b, c, a): the scalar mirror uses (−n).mul_add(ln2, x)
    let r = vfmsq_f32(x, n, vdupq_n_f32(EXP_LN2_HI));
    let r = vfmsq_f32(r, n, vdupq_n_f32(EXP_LN2_LO));
    let mut p = vdupq_n_f32(1.0 / 720.0);
    for c in EXP_POLY {
        p = vfmaq_f32(vdupq_n_f32(c), p, r);
    }
    let e = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)));
    vmulq_f32(p, vreinterpretq_f32_s32(e))
}

/// Vector SiLU, bit-identical to `silu_fast`.
///
/// # Safety
/// NEON only (baseline on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn silu_v(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let one = vdupq_n_f32(1.0);
    vdivq_f32(x, vaddq_f32(one, exp_v(vnegq_f32(x))))
}

/// Vector sigmoid, bit-identical to `sigmoid_fast`.
///
/// # Safety
/// NEON only (baseline on aarch64).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn sigmoid_v(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let one = vdupq_n_f32(1.0);
    vdivq_f32(one, vaddq_f32(one, exp_v(vnegq_f32(x))))
}

/// GLU over one time row: `out[c] = a[c] · sigmoid(b[c])`.
pub fn glu_row(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = out.len();
        let mut i = 0;
        // SAFETY: NEON baseline; bounds by the loop condition.
        unsafe {
            while i + 4 <= n {
                let g = sigmoid_v(vld1q_f32(b.as_ptr().add(i)));
                vst1q_f32(
                    out.as_mut_ptr().add(i),
                    vmulq_f32(vld1q_f32(a.as_ptr().add(i)), g),
                );
                i += 4;
            }
        }
        for c in i..n {
            out[c] = a[c] * sigmoid_fast(b[c]);
        }
        return;
    }
    #[allow(unreachable_code)]
    for c in 0..out.len() {
        out[c] = a[c] * sigmoid_fast(b[c]);
    }
}

/// In-place SiLU over a row with `silu_fast` / `silu_v`.
pub fn silu_row(row: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let n = row.len();
        let mut i = 0;
        // SAFETY: NEON baseline; bounds by the loop condition.
        unsafe {
            while i + 4 <= n {
                let p = row.as_mut_ptr().add(i);
                vst1q_f32(p, silu_v(vld1q_f32(p)));
                i += 4;
            }
        }
        for v in &mut row[i..] {
            *v = silu_fast(*v);
        }
        return;
    }
    #[allow(unreachable_code)]
    for v in row.iter_mut() {
        *v = silu_fast(*v);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn exp_row_neon(row: &mut [f32], scale: f32) -> f32 {
    use std::arch::aarch64::*;
    let n = row.len();
    let p = row.as_mut_ptr();
    let vs = vdupq_n_f32(scale);
    let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 4 <= n {
        let v = vmulq_f32(vld1q_f32(p.add(i)), vs);
        vst1q_f32(p.add(i), v);
        vmax = vmaxq_f32(vmax, v);
        i += 4;
    }
    let mut max = vmaxvq_f32(vmax);
    while i < n {
        *p.add(i) *= scale;
        max = max.max(*p.add(i));
        i += 1;
    }
    let vm = vdupq_n_f32(max);
    let mut vsum = vdupq_n_f32(0.0);
    i = 0;
    while i + 4 <= n {
        let e = exp_v(vsubq_f32(vld1q_f32(p.add(i)), vm));
        vst1q_f32(p.add(i), e);
        vsum = vaddq_f32(vsum, e);
        i += 4;
    }
    let mut sum = vaddvq_f32(vsum);
    while i < n {
        let e = exp_poly(*p.add(i) - max);
        *p.add(i) = e;
        sum += e;
        i += 1;
    }
    sum
}

#[cfg(test)]
mod exp_tests {
    #[test]
    fn exp_poly_close_to_libm_and_scalar_equals_vector() {
        let xs: Vec<f32> = (-2000..2000)
            .map(|i| i as f32 * 0.05)
            .chain([-87.0, -100.0, 88.0, 1e3, 0.0, -0.0])
            .collect();
        for &x in &xs {
            let got = super::exp_poly(x);
            let want = (x.clamp(-87.0, 88.0) as f64).exp();
            assert!(
                ((got as f64 - want) / want).abs() < 3e-7,
                "x={x}: {got} vs {want}"
            );
        }
        #[cfg(target_arch = "aarch64")]
        for ch in xs.chunks_exact(4) {
            let mut out = [0f32; 4];
            unsafe {
                use std::arch::aarch64::*;
                vst1q_f32(out.as_mut_ptr(), super::silu_v(vld1q_f32(ch.as_ptr())));
            }
            for (x, o) in ch.iter().zip(out) {
                assert_eq!(
                    o.to_bits(),
                    super::silu_fast(*x).to_bits(),
                    "silu scalar != vector at {x}"
                );
            }
        }
    }
}

#[cfg(test)]
mod softmax_tests {
    #[test]
    fn neon_softmax_matches_libm() {
        let mut seed = 1u64;
        for n in [1usize, 3, 4, 7, 64, 333] {
            let x: Vec<f32> = (0..n)
                .map(|_| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((seed >> 33) % 20001) as f32 / 100.0 - 100.0
                })
                .collect();
            let mut a = x.clone();
            super::softmax_row_scaled(&mut a, 0.0884);
            let max = x.iter().fold(f32::NEG_INFINITY, |m, v| m.max(v * 0.0884));
            let e: Vec<f64> = x
                .iter()
                .map(|v| ((v * 0.0884 - max) as f64).exp())
                .collect();
            let s: f64 = e.iter().sum();
            for (j, (got, want)) in a.iter().zip(&e).enumerate() {
                let want = (want / s) as f32;
                assert!(
                    (got - want).abs() <= 2e-6 + want.abs() * 4e-6,
                    "n={n} j={j}: {got} vs {want}"
                );
            }
        }
    }
}

/// aarch64: every NEON reduction equals its portable mirror bit for bit, so x86 (which runs the mirrors) matches.
#[cfg(all(test, target_arch = "aarch64"))]
mod portable_tests {
    use super::*;

    fn rnd(seed: &mut u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0
            })
            .collect()
    }

    #[test]
    fn neon_reductions_equal_portable_mirrors() {
        let mut seed = 11u64;
        for &(in_f, out_f) in &[
            (640usize, 2560usize),
            (1024, 13),
            (4, 4),
            (12, 7),
            (1000, 9),
        ] {
            let x = rnd(&mut seed, 4 * in_f);
            let w = rnd(&mut seed, out_f * in_f);
            let bias = rnd(&mut seed, out_f);
            for toks in 1..=4 {
                let (mut a, mut p) = (vec![0f32; toks * out_f], vec![0f32; toks * out_f]);
                unsafe { tile_rows_neon(&x, toks, in_f, &w, Some(&bias), out_f, &mut a) };
                tile_rows_portable(&x, toks, in_f, &w, Some(&bias), out_f, &mut p);
                assert!(
                    a.iter().zip(&p).all(|(u, v)| u.to_bits() == v.to_bits()),
                    "tile {in_f}x{out_f} t{toks}"
                );
            }
            if in_f % 4 == 0 {
                let (mut a, mut p) = (vec![0f32; out_f], vec![0f32; out_f]);
                unsafe { gemv_neon(&x[..in_f], in_f, &w, Some(&bias), out_f, &mut a) };
                gemv_portable(&x[..in_f], in_f, &w, Some(&bias), out_f, &mut p);
                assert!(
                    a.iter().zip(&p).all(|(u, v)| u.to_bits() == v.to_bits()),
                    "gemv {in_f}x{out_f}"
                );
            }
            if in_f % 8 == 0 {
                let wh: Vec<u16> = w.iter().map(|v| f16::from_f32(*v).to_bits()).collect();
                let (mut a, mut p) = (vec![0f32; out_f], vec![0f32; out_f]);
                unsafe { gemv_f16_neon(&x[..in_f], in_f, &wh, Some(&bias), out_f, &mut a) };
                gemv_f16_portable(&x[..in_f], in_f, &wh, Some(&bias), out_f, &mut p);
                assert!(
                    a.iter().zip(&p).all(|(u, v)| u.to_bits() == v.to_bits()),
                    "gemv_f16 {in_f}x{out_f}"
                );
            }
        }
        for n in [1usize, 3, 4, 7, 8, 13, 64, 1023, 1024] {
            let x = rnd(&mut seed, n);
            let (m1, v1) = unsafe { row_mean_var_neon(&x, n) };
            let (m2, v2) = row_mean_var_portable(&x);
            assert_eq!(
                (m1.to_bits(), v1.to_bits()),
                (m2.to_bits(), v2.to_bits()),
                "mean_var {n}"
            );
            let (mut a, mut p) = (x.clone(), x.clone());
            let s1 = unsafe { exp_row_neon(&mut a, 0.0884) };
            let s2 = exp_row_scaled_portable(&mut p, 0.0884);
            assert_eq!(s1.to_bits(), s2.to_bits(), "exp sum {n}");
            assert!(
                a.iter().zip(&p).all(|(u, v)| u.to_bits() == v.to_bits()),
                "exp row {n}"
            );
        }
    }
}
