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
    for (ti, row) in x.chunks_exact(in_f).take(toks).enumerate() {
        for o in 0..out_f {
            y[ti * out_f + o] = dot(row, &w[o * in_f..(o + 1) * in_f]) + b.map_or(0.0, |b| b[o]);
        }
    }
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
    for o in 0..out_f {
        y[o] = dot(x, &w[o * in_f..(o + 1) * in_f]) + b.map_or(0.0, |b| b[o]);
    }
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
    for o in 0..out_f {
        let row = &w[o * in_f..(o + 1) * in_f];
        let mut acc = 0f32;
        for (xv, &wv) in x.iter().zip(row) {
            acc += xv * f16::from_bits(wv).to_f32();
        }
        y[o] = acc + b.map_or(0.0, |b| b[o]);
    }
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
        let xi = &x[i * d..(i + 1) * d];
        let mean = xi.iter().sum::<f32>() / d as f32;
        let var = xi.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for k in 0..d {
            yr[k] = (xi[k] - mean) * inv * g[k] + b[k];
        }
    });
}

/// In-place SiLU over a `[n, d]` buffer, parallel over rows.
pub fn silu_inplace(buf: &mut [f32], d: usize) {
    par_rows(buf, d, |_, row| row.iter_mut().for_each(|v| *v = silu(*v)));
}

#[inline]
pub fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

#[inline]
pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}
