//! Ternary × int8 GEMM kernels (PLAN 3.4+). `y[t, o] = Σ_g s_w[o,g] · s_x[t,g] · Σ_{c∈g} w[o,c] · x[t,c]`.
//! Scalar reference + runtime-dispatched SIMD. All variants must produce identical i32 group sums.
use crate::packed::{QuantAct, TernaryMatrix};
use crate::tensor::threads;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    NeonDotprod,
}

pub fn detect() -> Kernel {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return Kernel::NeonDotprod;
        }
    }
    Kernel::Scalar
}

static KERNEL: std::sync::OnceLock<Kernel> = std::sync::OnceLock::new();

pub fn kernel() -> Kernel {
    *KERNEL.get_or_init(|| match std::env::var("PRX_KERNEL").as_deref() {
        Ok("scalar") => Kernel::Scalar,
        _ => detect(),
    })
}

/// Scalar reference: rows `[r0, r1)` of `w`, all tokens of `x`, writes `y[t, o]` (y is the full `[n, out]` buffer).
pub fn gemm_scalar(w: &TernaryMatrix, x: &QuantAct, y: &mut [f32], r0: usize, r1: usize) {
    let (inp, g) = (w.inp, w.group);
    let ng = inp / g;
    for o in r0..r1 {
        let wr = &w.w[o * inp..(o + 1) * inp];
        for t in 0..x.n {
            let xr = &x.x[t * inp..(t + 1) * inp];
            let mut acc = 0f32;
            for gi in 0..ng {
                let mut s = 0i32;
                for c in gi * g..(gi + 1) * g {
                    s += wr[c] as i32 * xr[c] as i32;
                }
                // same float association and FMA as the SIMD kernels: acc = fma(sum, s_x · s_w, acc)
                acc = (s as f32).mul_add(x.scales[t * ng + gi] * w.scales[o * ng + gi], acc);
            }
            y[t * w.out + o] = acc;
        }
    }
}

fn gemm_rows(k: Kernel, w: &TernaryMatrix, x: &QuantAct, y: &mut [f32], r0: usize, r1: usize) {
    match k {
        Kernel::Scalar => gemm_scalar(w, x, y, r0, r1),
        #[cfg(target_arch = "aarch64")]
        Kernel::NeonDotprod => unsafe { neon::gemm_dotprod(w, x, y, r0, r1) },
        #[cfg(not(target_arch = "aarch64"))]
        Kernel::NeonDotprod => gemm_scalar(w, x, y, r0, r1),
    }
}

/// `y[n, out]` = quantised `x` × `w`ᵀ, parallel over output-row blocks (rayon pool). `bias` optional.
pub fn gemm(w: &TernaryMatrix, x: &QuantAct, y: &mut [f32], bias: Option<&[f32]>) {
    use rayon::prelude::*;
    debug_assert_eq!(x.inp, w.inp);
    debug_assert_eq!(x.group, w.group);
    debug_assert_eq!(y.len(), x.n * w.out);
    let k = kernel();
    let blocks = (threads() * 2).min(w.out / 4).max(1);
    let chunk = w.out.div_ceil(blocks).div_ceil(4) * 4;
    let yp = y.as_mut_ptr() as usize;
    let len = y.len();
    let n_blocks = w.out.div_ceil(chunk);
    (0..n_blocks).into_par_iter().for_each(|bi| {
        let r0 = bi * chunk;
        let r1 = (r0 + chunk).min(w.out);
        // SAFETY: each block writes only columns [r0, r1) of every row of y; blocks are disjoint.
        let y = unsafe { std::slice::from_raw_parts_mut(yp as *mut f32, len) };
        gemm_rows(k, w, x, y, r0, r1);
    });
    if let Some(b) = bias {
        for t in 0..x.n {
            for (o, v) in y[t * w.out..(t + 1) * w.out].iter_mut().enumerate() {
                *v += b[o];
            }
        }
    }
}
