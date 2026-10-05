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

/// `y[n, out] = x[n, in] · W[out, in]^T + b[out]`. Parallel over output columns (rayon pool) when the work is
/// large enough; single-threaded for tiny calls (decoder steps).
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
    let row = |xi: &[f32], yrow: &mut [f32], o0: usize| {
        for (o, yo) in yrow.iter_mut().enumerate() {
            let oo = o0 + o;
            *yo = dot(xi, &w[oo * in_f..(oo + 1) * in_f]) + b.map_or(0.0, |b| b[oo]);
        }
    };
    if n * out_f * in_f < 1 << 20 {
        for (xi, yrow) in x.chunks_exact(in_f).zip(y.chunks_exact_mut(out_f)) {
            row(xi, yrow, 0);
        }
        return;
    }
    if n >= 8 {
        y.par_chunks_mut(out_f)
            .zip(x.par_chunks(in_f))
            .for_each(|(yrow, xi)| row(xi, yrow, 0));
    } else {
        // few tokens: split the output columns instead
        let cols = out_f.div_ceil(threads()).max(16);
        for (xi, yrow) in x.chunks_exact(in_f).zip(y.chunks_exact_mut(out_f)) {
            yrow.par_chunks_mut(cols)
                .enumerate()
                .for_each(|(ci, yc)| row(xi, yc, ci * cols));
        }
    }
}

/// LayerNorm over the last dim, eps 1e-5 (nn.LayerNorm default), biased variance.
pub fn layer_norm(x: &[f32], n: usize, d: usize, g: &[f32], b: &[f32], y: &mut [f32]) {
    for i in 0..n {
        let xi = &x[i * d..(i + 1) * d];
        let mean = xi.iter().sum::<f32>() / d as f32;
        let var = xi.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for k in 0..d {
            y[i * d + k] = (xi[k] - mean) * inv * g[k] + b[k];
        }
    }
}

#[inline]
pub fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

#[inline]
pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}
