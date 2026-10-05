//! NEON SDOT kernel (ARMv8.2 dotprod), PLAN 3.4. Register tile: 4 output rows × 4 tokens, 16 int32x4 accumulators,
//! one 128-wide group at a time (8 × 16-byte chunks), then a pairwise reduction and f32 FMA with the group scales.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
use std::arch::aarch64::*;
use std::arch::asm;

use crate::packed::{QuantAct, TernaryMatrix};

/// `acc += dot4(a, b)` per 32-bit lane (ARMv8.2 `sdot`). Inline asm because `vdotq_s32` is unstable on stable Rust.
#[inline(always)]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut r = acc;
    asm!("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) r, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
    r
}

/// Full 4×4 tile. `wp[r]` → row r of W at column 0, `xp[c]` → token c at column 0; `ng` groups of `g` columns.
#[inline(always)]
unsafe fn tile_4x4(
    wp: [*const i8; 4],
    xp: [*const i8; 4],
    sw: [*const f32; 4],
    sxt: *const f32,
    n: usize,
    ng: usize,
    g: usize,
    yp: [*mut f32; 4],
    out: usize,
) {
    let chunks = g / 16;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    for gi in 0..ng {
        let z = vdupq_n_s32(0);
        let (mut s00, mut s01, mut s02, mut s03) = (z, z, z, z);
        let (mut s10, mut s11, mut s12, mut s13) = (z, z, z, z);
        let (mut s20, mut s21, mut s22, mut s23) = (z, z, z, z);
        let (mut s30, mut s31, mut s32, mut s33) = (z, z, z, z);
        let base = gi * g;
        for ck in 0..chunks {
            let off = base + ck * 16;
            let w0 = vld1q_s8(wp[0].add(off));
            let w1 = vld1q_s8(wp[1].add(off));
            let w2 = vld1q_s8(wp[2].add(off));
            let w3 = vld1q_s8(wp[3].add(off));
            let x0 = vld1q_s8(xp[0].add(off));
            let x1 = vld1q_s8(xp[1].add(off));
            let x2 = vld1q_s8(xp[2].add(off));
            let x3 = vld1q_s8(xp[3].add(off));
            s00 = sdot(s00, w0, x0);
            s01 = sdot(s01, w0, x1);
            s02 = sdot(s02, w0, x2);
            s03 = sdot(s03, w0, x3);
            s10 = sdot(s10, w1, x0);
            s11 = sdot(s11, w1, x1);
            s12 = sdot(s12, w1, x2);
            s13 = sdot(s13, w1, x3);
            s20 = sdot(s20, w2, x0);
            s21 = sdot(s21, w2, x1);
            s22 = sdot(s22, w2, x2);
            s23 = sdot(s23, w2, x3);
            s30 = sdot(s30, w3, x0);
            s31 = sdot(s31, w3, x1);
            s32 = sdot(s32, w3, x2);
            s33 = sdot(s33, w3, x3);
        }
        // per row: 4 token sums into one vector [t0, t1, t2, t3]
        let r0 = vpaddq_s32(vpaddq_s32(s00, s01), vpaddq_s32(s02, s03));
        let r1 = vpaddq_s32(vpaddq_s32(s10, s11), vpaddq_s32(s12, s13));
        let r2 = vpaddq_s32(vpaddq_s32(s20, s21), vpaddq_s32(s22, s23));
        let r3 = vpaddq_s32(vpaddq_s32(s30, s31), vpaddq_s32(s32, s33));
        let sx = vld1q_f32(sxt.add(gi * n)); // 4 token scales for this group
        acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(r0), vmulq_n_f32(sx, *sw[0].add(gi)));
        acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(r1), vmulq_n_f32(sx, *sw[1].add(gi)));
        acc2 = vfmaq_f32(acc2, vcvtq_f32_s32(r2), vmulq_n_f32(sx, *sw[2].add(gi)));
        acc3 = vfmaq_f32(acc3, vcvtq_f32_s32(r3), vmulq_n_f32(sx, *sw[3].add(gi)));
    }
    // y is token-major: y[t, o]; write 4 tokens for each row with stride `out`
    let accs = [acc0, acc1, acc2, acc3];
    for r in 0..4 {
        let mut tmp = [0f32; 4];
        vst1q_f32(tmp.as_mut_ptr(), accs[r]);
        for c in 0..4 {
            *yp[r].add(c * out) = tmp[c];
        }
    }
}

/// Generic edge tile (rows ≤ 4, toks ≤ 4), same arithmetic and float association as the fast tile.
#[inline(always)]
unsafe fn tile_edge(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    o: usize,
    rows: usize,
    t: usize,
    toks: usize,
) {
    let (inp, g, out) = (w.inp, w.group, w.out);
    let ng = inp / g;
    let chunks = g / 16;
    let mut acc = [[0f32; 4]; 4];
    for gi in 0..ng {
        let mut s = [[vdupq_n_s32(0); 4]; 4];
        let base = gi * g;
        for ck in 0..chunks {
            let off = base + ck * 16;
            let mut wv = [vdupq_n_s8(0); 4];
            let mut xv = [vdupq_n_s8(0); 4];
            for r in 0..rows {
                wv[r] = vld1q_s8(w.w.as_ptr().add((o + r) * inp + off));
            }
            for c in 0..toks {
                xv[c] = vld1q_s8(x.x.as_ptr().add((t + c) * inp + off));
            }
            for r in 0..rows {
                for c in 0..toks {
                    s[r][c] = sdot(s[r][c], wv[r], xv[c]);
                }
            }
        }
        for r in 0..rows {
            let sw = *w.scales.get_unchecked((o + r) * ng + gi);
            for c in 0..toks {
                let sx = *x.scales_t.get_unchecked(gi * x.n_pad + t + c);
                acc[r][c] = (vaddvq_s32(s[r][c]) as f32).mul_add(sx * sw, acc[r][c]);
                // same FMA as the 4×4 tile
            }
        }
    }
    for r in 0..rows {
        for c in 0..toks {
            *y.get_unchecked_mut((t + c) * out + o + r) = acc[r][c];
        }
    }
}

/// # Safety
/// Caller must have verified `dotprod` support (see `kernels::detect`).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemm_dotprod(w: &TernaryMatrix, x: &QuantAct, y: &mut [f32], r0: usize, r1: usize) {
    let (inp, g, out, n) = (w.inp, w.group, w.out, x.n);
    let _ = n;
    let ng = inp / g;
    debug_assert_eq!(g % 16, 0);
    let mut o = r0;
    while o < r1 {
        let rows = (r1 - o).min(4);
        let mut t = 0;
        while t < n {
            let toks = (n - t).min(4);
            if rows == 4 && toks == 4 {
                let wp = [
                    w.w.as_ptr().add(o * inp),
                    w.w.as_ptr().add((o + 1) * inp),
                    w.w.as_ptr().add((o + 2) * inp),
                    w.w.as_ptr().add((o + 3) * inp),
                ];
                let xp = [
                    x.x.as_ptr().add(t * inp),
                    x.x.as_ptr().add((t + 1) * inp),
                    x.x.as_ptr().add((t + 2) * inp),
                    x.x.as_ptr().add((t + 3) * inp),
                ];
                let sw = [
                    w.scales.as_ptr().add(o * ng),
                    w.scales.as_ptr().add((o + 1) * ng),
                    w.scales.as_ptr().add((o + 2) * ng),
                    w.scales.as_ptr().add((o + 3) * ng),
                ];
                let yp = [
                    y.as_mut_ptr().add(t * out + o),
                    y.as_mut_ptr().add(t * out + o + 1),
                    y.as_mut_ptr().add(t * out + o + 2),
                    y.as_mut_ptr().add(t * out + o + 3),
                ];
                tile_4x4(
                    wp,
                    xp,
                    sw,
                    x.scales_t.as_ptr().add(t),
                    x.n_pad,
                    ng,
                    g,
                    yp,
                    out,
                );
            } else {
                tile_edge(w, x, y, o, rows, t, toks);
            }
            t += toks;
        }
        o += rows;
    }
}

/// `acc += A(2×8) · B(8×2)` (ARMv8.6 `smmla`): lanes `[a0·b0, a0·b1, a1·b0, a1·b1]`. Inline asm (intrinsic unstable).
#[inline(always)]
unsafe fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut r = acc;
    asm!("smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc = inout(vreg) r, a = in(vreg) a, b = in(vreg) b, options(pure, nomem, nostack));
    r
}

/// I8MM kernel (PLAN 3.5): weights in `Layout::Pairs`, activations in `x_pairs`; 4 rows × 4 tokens per tile as
/// four 2×2 `smmla` accumulators; no horizontal reductions. Same float association as the other kernels.
///
/// # Safety
/// Caller must have verified `i8mm` support and `w.layout == Layout::Pairs`.
#[target_feature(enable = "neon,i8mm")]
unsafe fn gemm_i8mm_4x4(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    r0: usize,
    r1: usize,
    t0: usize,
    t1: usize,
) {
    let (inp, g, out, n, n_pad) = (w.inp, w.group, w.out, t1, x.n_pad);
    let ng = inp / g;
    let slabs = inp / 8;
    let gslabs = g / 8;
    debug_assert!(r0 % 2 == 0 && r1 % 2 == 0);
    let mut o = r0;
    while o < r1 {
        let rows = (r1 - o).min(4);
        let rp0 = w.w.as_ptr().add((o / 2) * slabs * 16);
        let rp1 = if rows > 2 {
            w.w.as_ptr().add((o / 2 + 1) * slabs * 16)
        } else {
            rp0
        };
        let mut t = t0;
        while t < n {
            let toks = (n - t).min(4);
            let tp0 = x.x.as_ptr().add((t / 2) * slabs * 16);
            let tp1 = if toks > 2 {
                x.x.as_ptr().add((t / 2 + 1) * slabs * 16)
            } else {
                tp0
            };
            let z = vdupq_n_f32(0.0);
            let (mut f00, mut f01, mut f10, mut f11) = (z, z, z, z);
            for gi in 0..ng {
                let zi = vdupq_n_s32(0);
                let (mut a00, mut a01, mut a10, mut a11) = (zi, zi, zi, zi);
                let base = gi * gslabs * 16;
                for sl in 0..gslabs {
                    let off = base + sl * 16;
                    let wa = vld1q_s8(rp0.add(off));
                    let wb = vld1q_s8(rp1.add(off));
                    let xa = vld1q_s8(tp0.add(off));
                    let xb = vld1q_s8(tp1.add(off));
                    a00 = smmla(a00, wa, xa);
                    a01 = smmla(a01, wa, xb);
                    a10 = smmla(a10, wb, xa);
                    a11 = smmla(a11, wb, xb);
                }
                // scale lanes [r0t0, r0t1, r1t0, r1t1] = sx[t..] · sw[r..]  (same product as the other kernels)
                let sw = |r: usize| *w.scales.get_unchecked((o + r).min(out - 1) * ng + gi);
                let swa4 = vcombine_f32(vdup_n_f32(sw(0)), vdup_n_f32(sw(1))); // [sw0, sw0, sw1, sw1]
                let swb4 = vcombine_f32(vdup_n_f32(sw(2)), vdup_n_f32(sw(3)));
                let sxp = x.scales_t.as_ptr().add(gi * n_pad + t);
                let sx_a = vld1_f32(sxp); // [sx(t), sx(t+1)]
                let sx_b = vld1_f32(sxp.add(2)); // [sx(t+2), sx(t+3)]
                let sxa4 = vcombine_f32(sx_a, sx_a); // [sx0, sx1, sx0, sx1]
                let sxb4 = vcombine_f32(sx_b, sx_b);
                f00 = vfmaq_f32(f00, vcvtq_f32_s32(a00), vmulq_f32(sxa4, swa4));
                f01 = vfmaq_f32(f01, vcvtq_f32_s32(a01), vmulq_f32(sxb4, swa4));
                f10 = vfmaq_f32(f10, vcvtq_f32_s32(a10), vmulq_f32(sxa4, swb4));
                f11 = vfmaq_f32(f11, vcvtq_f32_s32(a11), vmulq_f32(sxb4, swb4));
            }
            // scatter: f[rp][tp] lane l → y[(t + 2tp + l%2) * out + o + 2rp + l/2]
            let fs = [[f00, f01], [f10, f11]];
            for rp in 0..2 {
                if 2 * rp >= rows {
                    break;
                }
                for tp in 0..2 {
                    if 2 * tp >= toks {
                        break;
                    }
                    let mut v = [0f32; 4];
                    vst1q_f32(v.as_mut_ptr(), fs[rp][tp]);
                    for (l, val) in v.iter().enumerate() {
                        let (rr, tt) = (2 * rp + l / 2, 2 * tp + l % 2);
                        if rr < rows && tt < toks {
                            *y.get_unchecked_mut((t + tt) * out + o + rr) = *val;
                        }
                    }
                }
            }
            t += toks;
        }
        o += rows;
    }
}

/// I8MM main kernel: 8 rows × 8 tokens per tile (4 row pairs × 4 token pairs = 16 `smmla` accumulators, 8 loads per
/// 8-column slab → 64 MACs per load). Edges (rows or tokens < 8) go through the 4×4 tile.
///
/// # Safety
/// Caller must have verified `i8mm` support and `w.layout == Layout::Pairs`.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn gemm_i8mm(w: &TernaryMatrix, x: &QuantAct, y: &mut [f32], r0: usize, r1: usize) {
    let (inp, g, out, n, n_pad) = (w.inp, w.group, w.out, x.n, x.n_pad);
    let ng = inp / g;
    let slabs = inp / 8;
    let gslabs = g / 8;
    let n8 = n / 8 * 8; // full token tiles
    let r8 = r0 + (r1 - r0) / 8 * 8; // full row tiles
    const TCHUNK: usize = 32; // tokens per L1-resident chunk (32 × 4 KB = 128 KB at inp = 4096)
    let mut tc = 0;
    while tc < n8 {
        let tc_end = (tc + TCHUNK).min(n8);
        let mut o = r0;
        while o < r8 {
            let wp: [*const i8; 4] =
                std::array::from_fn(|rp| w.w.as_ptr().add((o / 2 + rp) * slabs * 16));
            let mut t = tc;
            while t < tc_end {
                let xp: [*const i8; 4] =
                    std::array::from_fn(|tp| x.x.as_ptr().add((t / 2 + tp) * slabs * 16));
                let z = vdupq_n_f32(0.0);
                let mut f = [[z; 4]; 4]; // [row pair][token pair]
                for gi in 0..ng {
                    let zi = vdupq_n_s32(0);
                    let mut a = [[zi; 4]; 4];
                    let base = gi * gslabs * 16;
                    for sl in 0..gslabs {
                        let off = base + sl * 16;
                        let wv = [
                            vld1q_s8(wp[0].add(off)),
                            vld1q_s8(wp[1].add(off)),
                            vld1q_s8(wp[2].add(off)),
                            vld1q_s8(wp[3].add(off)),
                        ];
                        let xv = [
                            vld1q_s8(xp[0].add(off)),
                            vld1q_s8(xp[1].add(off)),
                            vld1q_s8(xp[2].add(off)),
                            vld1q_s8(xp[3].add(off)),
                        ];
                        for rp in 0..4 {
                            a[rp][0] = smmla(a[rp][0], wv[rp], xv[0]);
                            a[rp][1] = smmla(a[rp][1], wv[rp], xv[1]);
                            a[rp][2] = smmla(a[rp][2], wv[rp], xv[2]);
                            a[rp][3] = smmla(a[rp][3], wv[rp], xv[3]);
                        }
                    }
                    let sxp = x.scales_t.as_ptr().add(gi * n_pad + t);
                    let sx: [float32x4_t; 4] = std::array::from_fn(|tp| {
                        let v = vld1_f32(sxp.add(2 * tp));
                        vcombine_f32(v, v) // [sx0, sx1, sx0, sx1]
                    });
                    for rp in 0..4 {
                        let sw0 = *w.scales.get_unchecked((o + 2 * rp) * ng + gi);
                        let sw1 = *w.scales.get_unchecked((o + 2 * rp + 1) * ng + gi);
                        let swv = vcombine_f32(vdup_n_f32(sw0), vdup_n_f32(sw1)); // [sw0, sw0, sw1, sw1]
                        for tp in 0..4 {
                            f[rp][tp] = vfmaq_f32(
                                f[rp][tp],
                                vcvtq_f32_s32(a[rp][tp]),
                                vmulq_f32(sx[tp], swv),
                            );
                        }
                    }
                }
                for rp in 0..4 {
                    for tp in 0..4 {
                        let mut v = [0f32; 4];
                        vst1q_f32(v.as_mut_ptr(), f[rp][tp]);
                        for (l, val) in v.iter().enumerate() {
                            *y.get_unchecked_mut((t + 2 * tp + l % 2) * out + o + 2 * rp + l / 2) =
                                *val;
                        }
                    }
                }
                t += 8;
            }
            o += 8;
        }
        tc = tc_end;
    }
    if n8 < n {
        gemm_i8mm_4x4(w, x, y, r0, r8, n8, n);
    }
    if r8 < r1 {
        gemm_i8mm_4x4(w, x, y, r8, r1, 0, n);
    }
}
