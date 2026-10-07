//! x86-64 kernels on the `Quad16` layout (16 rows × 4 columns per 64-byte block, PLAN 3c.9).
//!
//! A SIMD lane owns one output row: each step loads 4 columns of 16 rows and multiplies them with 4 activation bytes
//! broadcast to every lane, so a lane accumulates its row's exact i32 group sum without horizontal adds. Both
//! instruction sets multiply an *unsigned* byte by a signed one, so the weights are offset to unsigned on load:
//!
//! - AVX-512 VNNI (`vpdpbusd`): `u = w ^ 0x80 = w + 128` for any int8 `w`; `Σ u·x = Σ w·x + 128·Σ x`.
//! - AVX2 (`vpmaddubsw`, saturating i16 pairs): ternary `u = w + 1 ∈ {0, 1, 2}` (pairs ≤ 508, accumulated in i16
//!   across a group of ≤ 256), corrected by `Σ x`; arbitrary int8 uses `|w| · sign(x, w)` (pairs ≤ 32 258, widened
//!   to i32 every step), which needs no correction.
//!
//! `Σ x` per (token, group) comes from `QuantAct::gsum`. Group sums are exact integers and the float step is
//! `acc = fma(sum, s_x · s_w, acc)` per lane in group order, so results are bit-identical to `gemm_scalar_range`.
use std::arch::x86_64::*;

use super::Epilogue;
use crate::packed::{Layout, QuantAct, TernaryMatrix};

/// Store a 16-row lane block for token `t`: only rows in `[r0, r1)` (partial blocks at range edges).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn store_block(
    vals: &[f32; 16],
    y: &mut [f32],
    out: usize,
    rbase: usize,
    r0: usize,
    r1: usize,
    t: usize,
    ep: Epilogue,
) {
    let lo = r0.max(rbase);
    let hi = r1.min(rbase + 16);
    for o in lo..hi {
        let slot = &mut y[t * out + o];
        *slot = ep.apply(*slot, vals[o - rbase]);
    }
}

/// AVX-512 VNNI: rows `[r0, r1)` × tokens `[t0, t1)` of `y[n, out]`.
///
/// # Safety
/// The CPU must support AVX-512 F/BW/VNNI and FMA; `w` in `Quad16`, `x` with `gsum` (layout `Quad16`).
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_vnni_range(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    r0: usize,
    r1: usize,
    t0: usize,
    t1: usize,
    ep: Epilogue,
) {
    debug_assert!(w.layout == Layout::Quad16 && x.layout == Layout::Quad16);
    // with few tokens a single `vpdpbusd` chain per token is latency-bound (decoder GEMVs): split the group's steps
    // over K independent accumulators (integer sums, so the order does not matter)
    let split = (w.group / 4).is_multiple_of(4);
    for rb in r0 / 16..r1.div_ceil(16) {
        let mut t = t0;
        while t < t1 {
            let left = t1 - t;
            if left >= 8 {
                vnni_tile::<8, 1>(w, x, y, rb, r0, r1, t, ep);
                t += 8;
            } else if left >= 4 {
                vnni_tile::<4, 1>(w, x, y, rb, r0, r1, t, ep);
                t += 4;
            } else if left >= 2 {
                if split {
                    vnni_tile::<2, 2>(w, x, y, rb, r0, r1, t, ep);
                } else {
                    vnni_tile::<2, 1>(w, x, y, rb, r0, r1, t, ep);
                }
                t += 2;
            } else {
                if split {
                    vnni_tile::<1, 4>(w, x, y, rb, r0, r1, t, ep);
                } else {
                    vnni_tile::<1, 1>(w, x, y, rb, r0, r1, t, ep);
                }
                t += 1;
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vnni,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn vnni_tile<const T: usize, const K: usize>(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    rb: usize,
    r0: usize,
    r1: usize,
    t: usize,
    ep: Epilogue,
) {
    let (inp, g) = (w.inp, w.group);
    let (ng, qpg) = (inp / g, g / 4);
    let wb = w.w.as_ptr().add(rb * (inp / 4) * 64);
    let sq = w.scales_q.as_ptr().add(rb * ng * 16);
    let xp = x.x.as_ptr();
    let flip = _mm512_set1_epi8(0x80u8 as i8);
    let mut accf = [_mm512_setzero_ps(); T];
    for gi in 0..ng {
        let mut acc = [[_mm512_setzero_si512(); K]; T];
        let mut q = gi * qpg;
        while q < (gi + 1) * qpg {
            for k in 0..K {
                let wv = _mm512_xor_si512(
                    std::ptr::read_unaligned(wb.add((q + k) * 64) as *const __m512i),
                    flip,
                );
                for (j, a) in acc.iter_mut().enumerate() {
                    let xb =
                        std::ptr::read_unaligned(xp.add((t + j) * inp + (q + k) * 4) as *const i32);
                    a[k] = _mm512_dpbusd_epi32(a[k], wv, _mm512_set1_epi32(xb));
                }
            }
            q += K;
        }
        let sw = _mm512_loadu_ps(sq.add(gi * 16));
        for (j, (a, f)) in acc.iter().zip(accf.iter_mut()).enumerate() {
            let tok = t + j;
            let a = a
                .iter()
                .fold(_mm512_setzero_si512(), |s, v| _mm512_add_epi32(s, *v));
            let corr = _mm512_set1_epi32(128 * x.gsum[tok * ng + gi]);
            let s = _mm512_cvtepi32_ps(_mm512_sub_epi32(a, corr));
            let p = _mm512_mul_ps(_mm512_set1_ps(x.scales[tok * ng + gi]), sw);
            *f = _mm512_fmadd_ps(s, p, *f);
        }
    }
    let full = rb * 16 >= r0 && rb * 16 + 16 <= r1;
    let mut vals = [0f32; 16];
    for (j, f) in accf.iter().enumerate() {
        let dst = y.as_mut_ptr().add((t + j) * w.out + rb * 16);
        match ep {
            // whole block inside the range: vector store; the residual add is the same per-lane fma(s, v, old)
            Epilogue::Store if full => _mm512_storeu_ps(dst, *f),
            Epilogue::AddScaled(sc) if full => _mm512_storeu_ps(
                dst,
                _mm512_fmadd_ps(_mm512_set1_ps(sc), *f, _mm512_loadu_ps(dst)),
            ),
            _ => {
                _mm512_storeu_ps(vals.as_mut_ptr(), *f);
                store_block(&vals, y, w.out, rb * 16, r0, r1, t + j, ep);
            }
        }
    }
}

/// AVX2: rows `[r0, r1)` × tokens `[t0, t1)` of `y[n, out]`.
///
/// # Safety
/// The CPU must support AVX2 and FMA; `w` in `Quad16`, `x` with `gsum` (layout `Quad16`); ternary weights need
/// `group <= 256` (i16 accumulation).
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_avx2_range(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    r0: usize,
    r1: usize,
    t0: usize,
    t1: usize,
    ep: Epilogue,
) {
    debug_assert!(w.layout == Layout::Quad16 && x.layout == Layout::Quad16);
    debug_assert!(!w.ternary || w.group <= 256);
    for rb in r0 / 16..r1.div_ceil(16) {
        let mut t = t0;
        while t < t1 {
            let left = t1 - t;
            let step = if left >= 4 {
                4
            } else if left >= 2 {
                2
            } else {
                1
            };
            match (step, w.ternary) {
                (4, true) => avx2_tile::<4, true>(w, x, y, rb, r0, r1, t, ep),
                (4, false) => avx2_tile::<4, false>(w, x, y, rb, r0, r1, t, ep),
                (2, true) => avx2_tile::<2, true>(w, x, y, rb, r0, r1, t, ep),
                (2, false) => avx2_tile::<2, false>(w, x, y, rb, r0, r1, t, ep),
                (_, true) => avx2_tile::<1, true>(w, x, y, rb, r0, r1, t, ep),
                (_, false) => avx2_tile::<1, false>(w, x, y, rb, r0, r1, t, ep),
            }
            t += step;
        }
    }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn avx2_tile<const T: usize, const TERN: bool>(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    rb: usize,
    r0: usize,
    r1: usize,
    t: usize,
    ep: Epilogue,
) {
    let (inp, g) = (w.inp, w.group);
    let (ng, qpg) = (inp / g, g / 4);
    let wb = w.w.as_ptr().add(rb * (inp / 4) * 64);
    let sq = w.scales_q.as_ptr().add(rb * ng * 16);
    let xp = x.x.as_ptr();
    let ones8 = _mm256_set1_epi8(1);
    let ones16 = _mm256_set1_epi16(1);
    // [token][half]: rows 0..8 and 8..16 of the block
    let mut accf = [[_mm256_setzero_ps(); 2]; T];
    for gi in 0..ng {
        let mut acc = [[_mm256_setzero_si256(); 2]; T];
        for q in gi * qpg..(gi + 1) * qpg {
            let w0 = _mm256_loadu_si256(wb.add(q * 64) as *const __m256i);
            let w1 = _mm256_loadu_si256(wb.add(q * 64 + 32) as *const __m256i);
            if TERN {
                let (u0, u1) = (_mm256_add_epi8(w0, ones8), _mm256_add_epi8(w1, ones8));
                for (j, a) in acc.iter_mut().enumerate() {
                    let xb = _mm256_set1_epi32(std::ptr::read_unaligned(
                        xp.add((t + j) * inp + q * 4) as *const i32,
                    ));
                    a[0] = _mm256_add_epi16(a[0], _mm256_maddubs_epi16(u0, xb));
                    a[1] = _mm256_add_epi16(a[1], _mm256_maddubs_epi16(u1, xb));
                }
            } else {
                let (u0, u1) = (_mm256_abs_epi8(w0), _mm256_abs_epi8(w1));
                for (j, a) in acc.iter_mut().enumerate() {
                    let xb = _mm256_set1_epi32(std::ptr::read_unaligned(
                        xp.add((t + j) * inp + q * 4) as *const i32,
                    ));
                    let p0 = _mm256_maddubs_epi16(u0, _mm256_sign_epi8(xb, w0));
                    let p1 = _mm256_maddubs_epi16(u1, _mm256_sign_epi8(xb, w1));
                    a[0] = _mm256_add_epi32(a[0], _mm256_madd_epi16(p0, ones16));
                    a[1] = _mm256_add_epi32(a[1], _mm256_madd_epi16(p1, ones16));
                }
            }
        }
        let sw = [
            _mm256_loadu_ps(sq.add(gi * 16)),
            _mm256_loadu_ps(sq.add(gi * 16 + 8)),
        ];
        for (j, (a, f)) in acc.iter().zip(accf.iter_mut()).enumerate() {
            let tok = t + j;
            let sx = _mm256_set1_ps(x.scales[tok * ng + gi]);
            let corr = _mm256_set1_epi32(x.gsum[tok * ng + gi]);
            for h in 0..2 {
                let sum = if TERN {
                    _mm256_sub_epi32(_mm256_madd_epi16(a[h], ones16), corr)
                } else {
                    a[h]
                };
                let p = _mm256_mul_ps(sx, sw[h]);
                f[h] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), p, f[h]);
            }
        }
    }
    let full = rb * 16 >= r0 && rb * 16 + 16 <= r1;
    let mut vals = [0f32; 16];
    for (j, f) in accf.iter().enumerate() {
        let dst = y.as_mut_ptr().add((t + j) * w.out + rb * 16);
        match ep {
            Epilogue::Store if full => {
                _mm256_storeu_ps(dst, f[0]);
                _mm256_storeu_ps(dst.add(8), f[1]);
            }
            Epilogue::AddScaled(sc) if full => {
                let s = _mm256_set1_ps(sc);
                _mm256_storeu_ps(dst, _mm256_fmadd_ps(s, f[0], _mm256_loadu_ps(dst)));
                _mm256_storeu_ps(
                    dst.add(8),
                    _mm256_fmadd_ps(s, f[1], _mm256_loadu_ps(dst.add(8))),
                );
            }
            _ => {
                _mm256_storeu_ps(vals.as_mut_ptr(), f[0]);
                _mm256_storeu_ps(vals.as_mut_ptr().add(8), f[1]);
                store_block(&vals, y, w.out, rb * 16, r0, r1, t + j, ep);
            }
        }
    }
}
