//! Ternary × int8 GEMM kernels (PLAN 3.4+). `y[t, o] = Σ_g s_w[o,g] · s_x[t,g] · Σ_{c∈g} w[o,c] · x[t,c]`.
//! Scalar reference + runtime-dispatched SIMD. All variants produce identical i32 group sums and use the same
//! float association `acc = fma(sum, s_x·s_w, acc)`, so results are bit-identical across kernels.
use crate::packed::{Layout, QuantAct, TernaryMatrix};
use crate::tensor::threads;

#[cfg(target_arch = "aarch64")]
pub mod neon;

/// What the kernel does at each output store. Applied identically by every kernel, so results stay bit-identical.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Epilogue {
    /// `y = v`
    Store,
    /// `y = silu(v)`
    Silu,
    /// `y = y + s · v` (residual add into a pre-filled buffer)
    AddScaled(f32),
}

impl Epilogue {
    /// Vector pre-pass: elementwise epilogues (SiLU) are applied on the whole register before the scatter, which
    /// then stores plainly; `AddScaled` needs the old values and stays per lane.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub unsafe fn apply_v(
        self,
        v: std::arch::aarch64::float32x4_t,
    ) -> (std::arch::aarch64::float32x4_t, Epilogue) {
        match self {
            Epilogue::Silu => (crate::tensor::silu_v(v), Epilogue::Store),
            other => (v, other),
        }
    }

    #[inline(always)]
    pub fn apply(self, old: f32, v: f32) -> f32 {
        match self {
            Epilogue::Store => v,
            Epilogue::Silu => crate::tensor::silu_fast(v),
            Epilogue::AddScaled(s) => s.mul_add(v, old),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    NeonDotprod,
    NeonI8mm,
}

impl Kernel {
    /// Weight layout this kernel consumes for ternary matrices (`PRX_WLAYOUT=pairs|2bit` overrides for I8MM).
    pub fn layout(self) -> Layout {
        match self {
            Kernel::NeonI8mm => match std::env::var("PRX_WLAYOUT").as_deref() {
                Ok("pairs") => Layout::Pairs,
                _ => Layout::Packed2,
            },
            _ => Layout::Rows,
        }
    }

    /// Layout for non-ternary int8 matrices (decoder weights): never 2-bit.
    pub fn int8_layout(self) -> Layout {
        match self {
            Kernel::NeonI8mm => Layout::Pairs,
            _ => Layout::Rows,
        }
    }
}

pub fn detect() -> Kernel {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            return Kernel::NeonI8mm;
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return Kernel::NeonDotprod;
        }
    }
    Kernel::Scalar
}

static KERNEL: std::sync::OnceLock<Kernel> = std::sync::OnceLock::new();

/// Row-block multiplier for GEMM dispatch (`PRX_GEMM_BLOCKS`: 0 = adaptive (default), k = k·threads blocks).
fn gemm_blocks_mult() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PRX_GEMM_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    })
}

/// Selected kernel (`PRX_KERNEL=scalar|sdot|i8mm` overrides detection).
pub fn kernel() -> Kernel {
    *KERNEL.get_or_init(|| match std::env::var("PRX_KERNEL").as_deref() {
        Ok("scalar") => Kernel::Scalar,
        Ok("sdot") | Ok("dotprod") => Kernel::NeonDotprod,
        Ok("i8mm") => Kernel::NeonI8mm,
        _ => detect(),
    })
}

/// Scalar reference: rows `[r0, r1)` of `w`, all tokens of `x`, writes `y[t, o]` (y is the full `[n, out]` buffer).
#[allow(clippy::needless_range_loop)]
pub fn gemm_scalar(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    r0: usize,
    r1: usize,
    ep: Epilogue,
) {
    gemm_scalar_range(w, x, y, r0, r1, 0, x.n, ep)
}

/// `gemm_scalar` restricted to tokens `[t0, t1)`.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn gemm_scalar_range(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    r0: usize,
    r1: usize,
    t0: usize,
    t1: usize,
    ep: Epilogue,
) {
    let (inp, g) = (w.inp, w.group);
    let ng = inp / g;
    for o in r0..r1 {
        for t in t0..t1 {
            let mut acc = 0f32;
            for gi in 0..ng {
                let mut s = 0i32;
                for c in gi * g..(gi + 1) * g {
                    s += w.get(o, c) as i32 * x.get(t, c) as i32;
                }
                // same float association and FMA as the SIMD kernels: acc = fma(sum, s_x · s_w, acc)
                acc = (s as f32).mul_add(x.scales[t * ng + gi] * w.scales[o * ng + gi], acc);
            }
            let slot = &mut y[t * w.out + o];
            *slot = ep.apply(*slot, acc);
        }
    }
}

fn gemm_rows(
    k: Kernel,
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    r0: usize,
    r1: usize,
    ep: Epilogue,
) {
    match k {
        Kernel::Scalar => gemm_scalar(w, x, y, r0, r1, ep),
        #[cfg(target_arch = "aarch64")]
        Kernel::NeonDotprod => unsafe { neon::gemm_dotprod(w, x, y, r0, r1, ep) },
        #[cfg(target_arch = "aarch64")]
        Kernel::NeonI8mm => unsafe {
            if w.layout == Layout::Packed2 {
                neon::gemm_i8mm_2bit(w, x, y, r0, r1, ep)
            } else {
                neon::gemm_i8mm(w, x, y, r0, r1, ep)
            }
        },
        #[cfg(not(target_arch = "aarch64"))]
        Kernel::NeonDotprod | Kernel::NeonI8mm => gemm_scalar(w, x, y, r0, r1, ep),
    }
}

/// Kernel for a (weight layout, activation layout) pair; degrades gracefully to the scalar reference.
fn select_kernel(w: &TernaryMatrix, x: &QuantAct) -> Kernel {
    match (kernel(), w.layout, x.layout) {
        (Kernel::NeonI8mm, Layout::Pairs | Layout::Packed2, Layout::Pairs) => Kernel::NeonI8mm,
        (Kernel::NeonI8mm, Layout::Rows, Layout::Rows)
        | (Kernel::NeonDotprod, Layout::Rows, Layout::Rows) => Kernel::NeonDotprod,
        _ => Kernel::Scalar,
    }
}

/// Single-threaded tile of the GEMM: rows `[r0, r1)` × tokens `[t0, t1)` of `y[n, out]` (row stride `w.out`).
/// Used by the attention core (PLAN 3c.8), where each task owns a block of query rows; `r0` and `r1` must be even
/// on the I8MM path (row pairs). Not for `Packed2` weights.
#[allow(clippy::too_many_arguments)]
pub fn gemm_rows_range(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    r0: usize,
    r1: usize,
    t0: usize,
    t1: usize,
    ep: Epilogue,
) {
    debug_assert!(w.layout != Layout::Packed2);
    debug_assert!(t1 <= x.n && r1 <= w.out);
    match select_kernel(w, x) {
        #[cfg(target_arch = "aarch64")]
        Kernel::NeonDotprod => unsafe { neon::gemm_dotprod_range(w, x, y, r0, r1, t0, t1, ep) },
        #[cfg(target_arch = "aarch64")]
        Kernel::NeonI8mm if w.layout == Layout::Pairs => unsafe {
            neon::gemm_i8mm_range(w, x, y, r0, r1, t0, t1, ep)
        },
        _ => gemm_scalar_range(w, x, y, r0, r1, t0, t1, ep),
    }
}

/// Single-threaded `y[n, out]` = quantised `x` × `w`ᵀ (+ bias), for per-step decoder work where dispatch would dominate.
pub fn gemm_single(w: &TernaryMatrix, x: &QuantAct, y: &mut [f32], bias: Option<&[f32]>) {
    let k = match (kernel(), w.layout, x.layout) {
        (Kernel::NeonI8mm, Layout::Pairs | Layout::Packed2, Layout::Pairs) => Kernel::NeonI8mm,
        (Kernel::NeonI8mm, Layout::Rows, Layout::Rows)
        | (Kernel::NeonDotprod, Layout::Rows, Layout::Rows) => Kernel::NeonDotprod,
        _ => Kernel::Scalar,
    };
    gemm_rows(k, w, x, y, 0, w.out, Epilogue::Store);
    if let Some(b) = bias {
        for t in 0..x.n {
            for (o, v) in y[t * w.out..(t + 1) * w.out].iter_mut().enumerate() {
                *v += b[o];
            }
        }
    }
}

/// `y[n, out]` = quantised `x` × `w`ᵀ, parallel over output-row blocks (rayon pool). `bias` optional.
pub fn gemm(w: &TernaryMatrix, x: &QuantAct, y: &mut [f32], bias: Option<&[f32]>) {
    gemm_ep(w, x, y, bias, Epilogue::Store)
}

/// `gemm` with an epilogue applied at every store (SiLU, or residual add into `y`).
pub fn gemm_ep(w: &TernaryMatrix, x: &QuantAct, y: &mut [f32], bias: Option<&[f32]>, ep: Epilogue) {
    use rayon::prelude::*;
    debug_assert_eq!(x.inp, w.inp);
    debug_assert_eq!(x.group, w.group);
    debug_assert_eq!(y.len(), x.n * w.out);
    // the kernel must match the weight layout; degrade gracefully otherwise
    let k = match (kernel(), w.layout, x.layout) {
        (Kernel::NeonI8mm, Layout::Pairs | Layout::Packed2, Layout::Pairs) => Kernel::NeonI8mm,
        (Kernel::NeonI8mm, Layout::Rows, Layout::Rows) => Kernel::NeonDotprod,
        (Kernel::NeonDotprod, Layout::Rows, Layout::Rows) => Kernel::NeonDotprod,
        _ => Kernel::Scalar,
    };
    // block count adapts to the work: enough small blocks for work-stealing to balance P- and E-cores on big GEMMs,
    // fewer on small ones so dispatch and tail effects stay low (≈ 4 MMAC per block, within [threads, 8·threads])
    let work = x.n * w.out * w.inp;
    let mult = gemm_blocks_mult();
    let blocks = if mult == 0 {
        (work / (4 << 20)).clamp(threads(), threads() * 8)
    } else {
        threads() * mult
    }
    .min(w.out / 8)
    .max(1);
    let chunk = w.out.div_ceil(blocks).div_ceil(8) * 8;
    let yp = y.as_mut_ptr() as usize;
    let len = y.len();
    let n_blocks = w.out.div_ceil(chunk);
    (0..n_blocks).into_par_iter().for_each(|bi| {
        let r0 = bi * chunk;
        let r1 = (r0 + chunk).min(w.out);
        // SAFETY: each block writes only columns [r0, r1) of every row of y; blocks are disjoint.
        let y = unsafe { std::slice::from_raw_parts_mut(yp as *mut f32, len) };
        gemm_rows(k, w, x, y, r0, r1, ep);
    });
    if let Some(b) = bias {
        for t in 0..x.n {
            for (o, v) in y[t * w.out..(t + 1) * w.out].iter_mut().enumerate() {
                *v += b[o];
            }
        }
    }
}
