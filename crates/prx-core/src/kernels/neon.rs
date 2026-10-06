//! NEON SDOT kernel (ARMv8.2 dotprod), PLAN 3.4. Register tile: 4 output rows × 4 tokens, 16 int32x4 accumulators,
//! one 128-wide group at a time (8 × 16-byte chunks), then a pairwise reduction and f32 FMA with the group scales.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
use std::arch::aarch64::*;
use std::arch::asm;

use super::Epilogue;
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
    ep: Epilogue,
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
            let p = yp[r].add(c * out);
            *p = ep.apply(*p, tmp[c]);
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
    ep: Epilogue,
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
            let p = y.get_unchecked_mut((t + c) * out + o + r);
            *p = ep.apply(*p, acc[r][c]);
        }
    }
}

/// # Safety
/// Caller must have verified `dotprod` support (see `kernels::detect`).
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn gemm_dotprod(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    r0: usize,
    r1: usize,
    ep: Epilogue,
) {
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
                    ep,
                );
            } else {
                tile_edge(w, x, y, o, rows, t, toks, ep);
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
    ep: Epilogue,
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
                            let p = y.get_unchecked_mut((t + tt) * out + o + rr);
                            *p = ep.apply(*p, *val);
                        }
                    }
                }
            }
            t += toks;
        }
        o += rows;
    }
}

/// 16 `smmla` updates of a 4×4 block of 2×2 accumulators: `$a[rp][tp] += W[rp] · X[tp]`.
macro_rules! mm16 {
    ($a00:ident,$a01:ident,$a02:ident,$a03:ident,$a10:ident,$a11:ident,$a12:ident,$a13:ident,
     $a20:ident,$a21:ident,$a22:ident,$a23:ident,$a30:ident,$a31:ident,$a32:ident,$a33:ident,
     $w0:expr,$w1:expr,$w2:expr,$w3:expr,$x0:expr,$x1:expr,$x2:expr,$x3:expr) => {{
        let (w0, w1, w2, w3) = ($w0, $w1, $w2, $w3);
        let (x0, x1, x2, x3) = ($x0, $x1, $x2, $x3);
        $a00 = smmla($a00, w0, x0);
        $a01 = smmla($a01, w0, x1);
        $a02 = smmla($a02, w0, x2);
        $a03 = smmla($a03, w0, x3);
        $a10 = smmla($a10, w1, x0);
        $a11 = smmla($a11, w1, x1);
        $a12 = smmla($a12, w1, x2);
        $a13 = smmla($a13, w1, x3);
        $a20 = smmla($a20, w2, x0);
        $a21 = smmla($a21, w2, x1);
        $a22 = smmla($a22, w2, x2);
        $a23 = smmla($a23, w2, x3);
        $a30 = smmla($a30, w3, x0);
        $a31 = smmla($a31, w3, x1);
        $a32 = smmla($a32, w3, x2);
        $a33 = smmla($a33, w3, x3);
    }};
}

/// Per-group epilogue: `f[rp][tp] += cvt(a[rp][tp]) · (sx[tp] · sw[rp])` for the 16 blocks, then zero the i32 accumulators.
macro_rules! epilogue16 {
    ($f:ident, $a00:ident,$a01:ident,$a02:ident,$a03:ident,$a10:ident,$a11:ident,$a12:ident,$a13:ident,
     $a20:ident,$a21:ident,$a22:ident,$a23:ident,$a30:ident,$a31:ident,$a32:ident,$a33:ident, $sx:expr, $sw:expr) => {{
        let sx: [float32x4_t; 4] = $sx;
        let sw: [float32x4_t; 4] = $sw;
        $f[0][0] = vfmaq_f32($f[0][0], vcvtq_f32_s32($a00), vmulq_f32(sx[0], sw[0]));
        $f[0][1] = vfmaq_f32($f[0][1], vcvtq_f32_s32($a01), vmulq_f32(sx[1], sw[0]));
        $f[0][2] = vfmaq_f32($f[0][2], vcvtq_f32_s32($a02), vmulq_f32(sx[2], sw[0]));
        $f[0][3] = vfmaq_f32($f[0][3], vcvtq_f32_s32($a03), vmulq_f32(sx[3], sw[0]));
        $f[1][0] = vfmaq_f32($f[1][0], vcvtq_f32_s32($a10), vmulq_f32(sx[0], sw[1]));
        $f[1][1] = vfmaq_f32($f[1][1], vcvtq_f32_s32($a11), vmulq_f32(sx[1], sw[1]));
        $f[1][2] = vfmaq_f32($f[1][2], vcvtq_f32_s32($a12), vmulq_f32(sx[2], sw[1]));
        $f[1][3] = vfmaq_f32($f[1][3], vcvtq_f32_s32($a13), vmulq_f32(sx[3], sw[1]));
        $f[2][0] = vfmaq_f32($f[2][0], vcvtq_f32_s32($a20), vmulq_f32(sx[0], sw[2]));
        $f[2][1] = vfmaq_f32($f[2][1], vcvtq_f32_s32($a21), vmulq_f32(sx[1], sw[2]));
        $f[2][2] = vfmaq_f32($f[2][2], vcvtq_f32_s32($a22), vmulq_f32(sx[2], sw[2]));
        $f[2][3] = vfmaq_f32($f[2][3], vcvtq_f32_s32($a23), vmulq_f32(sx[3], sw[2]));
        $f[3][0] = vfmaq_f32($f[3][0], vcvtq_f32_s32($a30), vmulq_f32(sx[0], sw[3]));
        $f[3][1] = vfmaq_f32($f[3][1], vcvtq_f32_s32($a31), vmulq_f32(sx[1], sw[3]));
        $f[3][2] = vfmaq_f32($f[3][2], vcvtq_f32_s32($a32), vmulq_f32(sx[2], sw[3]));
        $f[3][3] = vfmaq_f32($f[3][3], vcvtq_f32_s32($a33), vmulq_f32(sx[3], sw[3]));
        let z = vdupq_n_s32(0);
        $a00 = z;
        $a01 = z;
        $a02 = z;
        $a03 = z;
        $a10 = z;
        $a11 = z;
        $a12 = z;
        $a13 = z;
        $a20 = z;
        $a21 = z;
        $a22 = z;
        $a23 = z;
        $a30 = z;
        $a31 = z;
        $a32 = z;
        $a33 = z;
    }};
}

/// Scatter the 16 f32 2×2 blocks of a tile into `y` (token-major) through the epilogue.
#[inline(always)]
unsafe fn store_tile(
    f: &[[float32x4_t; 4]; 4],
    y: &mut [f32],
    t: usize,
    o: usize,
    out: usize,
    n: usize,
    ep: Epilogue,
) {
    for rp in 0..4 {
        for tp in 0..4 {
            let mut v = [0f32; 4];
            vst1q_f32(v.as_mut_ptr(), f[rp][tp]);
            for (l, val) in v.iter().enumerate() {
                let tt = t + 2 * tp + l % 2;
                if tt < n {
                    let p = y.get_unchecked_mut(tt * out + o + 2 * rp + l / 2);
                    *p = ep.apply(*p, *val);
                }
            }
        }
    }
}

#[inline(always)]
unsafe fn group_scales(
    x: &QuantAct,
    w: &TernaryMatrix,
    gi: usize,
    t: usize,
    o: usize,
    ng: usize,
) -> ([float32x4_t; 4], [float32x4_t; 4]) {
    let sxp = x.scales_t.as_ptr().add(gi * x.n_pad + t);
    let sx: [float32x4_t; 4] = std::array::from_fn(|tp| {
        let v = vld1_f32(sxp.add(2 * tp));
        vcombine_f32(v, v) // [sx0, sx1, sx0, sx1]
    });
    let sw: [float32x4_t; 4] = std::array::from_fn(|rp| {
        let sw0 = *w.scales.get_unchecked((o + 2 * rp) * ng + gi);
        let sw1 = *w.scales.get_unchecked((o + 2 * rp + 1) * ng + gi);
        vcombine_f32(vdup_n_f32(sw0), vdup_n_f32(sw1)) // [sw0, sw0, sw1, sw1]
    });
    (sx, sw)
}

/// I8MM main kernel on `Layout::Pairs`: 8 rows × 8 tokens per tile, 16 named `smmla` accumulators (fully unrolled —
/// indexed arrays spilled to the stack and halved throughput), per-group epilogue. Edges go through the 4×4 tile.
///
/// # Safety
/// Caller must have verified `i8mm` support and `w.layout == Layout::Pairs`.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn gemm_i8mm(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    r0: usize,
    r1: usize,
    ep: Epilogue,
) {
    let (inp, g, out, n, n_pad) = (w.inp, w.group, w.out, x.n, x.n_pad);
    let ng = inp / g;
    let slabs = inp / 8;
    let gslabs = g / 8;
    let n8 = n_pad; // tokens are zero-padded to a multiple of 8; padded outputs are not stored
    let r8 = r0 + (r1 - r0) / 8 * 8;
    const TCHUNK: usize = 32;
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
                let mut f = [[z; 4]; 4];
                let zi = vdupq_n_s32(0);
                let (mut a00, mut a01, mut a02, mut a03) = (zi, zi, zi, zi);
                let (mut a10, mut a11, mut a12, mut a13) = (zi, zi, zi, zi);
                let (mut a20, mut a21, mut a22, mut a23) = (zi, zi, zi, zi);
                let (mut a30, mut a31, mut a32, mut a33) = (zi, zi, zi, zi);
                for gi in 0..ng {
                    let base = gi * gslabs * 16;
                    for sl in 0..gslabs {
                        let off = base + sl * 16;
                        mm16!(
                            a00,
                            a01,
                            a02,
                            a03,
                            a10,
                            a11,
                            a12,
                            a13,
                            a20,
                            a21,
                            a22,
                            a23,
                            a30,
                            a31,
                            a32,
                            a33,
                            vld1q_s8(wp[0].add(off)),
                            vld1q_s8(wp[1].add(off)),
                            vld1q_s8(wp[2].add(off)),
                            vld1q_s8(wp[3].add(off)),
                            vld1q_s8(xp[0].add(off)),
                            vld1q_s8(xp[1].add(off)),
                            vld1q_s8(xp[2].add(off)),
                            vld1q_s8(xp[3].add(off))
                        );
                    }
                    let (sx, sw) = group_scales(x, w, gi, t, o, ng);
                    epilogue16!(
                        f, a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31,
                        a32, a33, sx, sw
                    );
                }
                store_tile(&f, y, t, o, out, n, ep);
                t += 8;
            }
            o += 8;
        }
        tc = tc_end;
    }
    if r8 < r1 {
        gemm_i8mm_4x4(w, x, y, r8, r1, 0, n, ep);
    }
}

/// I8MM kernel on `Layout::Packed2` weights (PLAN 3c.10): one 16-byte load per row pair per quad (4 slabs = 32 columns)
/// unpacked just in time into the four `smmla` operands. Fully unrolled; no edge tiles (tokens padded, rows multiples of 8).
///
/// # Safety
/// Caller must have verified `i8mm` support, `w.layout == Layout::Packed2`, `x.layout == Layout::Pairs`.
#[target_feature(enable = "neon,i8mm")]
pub unsafe fn gemm_i8mm_2bit(
    w: &TernaryMatrix,
    x: &QuantAct,
    y: &mut [f32],
    r0: usize,
    r1: usize,
    ep: Epilogue,
) {
    let (inp, g, out, n, n_pad) = (w.inp, w.group, w.out, x.n, x.n_pad);
    let ng = inp / g;
    let slabs = inp / 8;
    let nq = inp / 32;
    let gquads = g / 32;
    debug_assert!(r0 % 8 == 0 && (r1 - r0) % 8 == 0 && g % 32 == 0);
    const TCHUNK: usize = 32;
    let mask3 = vdupq_n_u8(3);
    let one = vdupq_n_s8(1);
    // operand j of a packed quad register
    macro_rules! unpack {
        ($c:expr, 0) => {
            vsubq_s8(vreinterpretq_s8_u8(vandq_u8($c, mask3)), one)
        };
        ($c:expr, 1) => {
            vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8($c, 2), mask3)), one)
        };
        ($c:expr, 2) => {
            vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8($c, 4), mask3)), one)
        };
        ($c:expr, 3) => {
            vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8($c, 6)), one)
        };
    }
    let mut tc = 0;
    while tc < n_pad {
        let tc_end = (tc + TCHUNK).min(n_pad);
        let mut o = r0;
        while o < r1 {
            let wp: [*const u8; 4] =
                std::array::from_fn(|rp| (w.w.as_ptr() as *const u8).add((o / 2 + rp) * nq * 16));
            let mut t = tc;
            while t < tc_end {
                let xp: [*const i8; 4] =
                    std::array::from_fn(|tp| x.x.as_ptr().add((t / 2 + tp) * slabs * 16));
                let z = vdupq_n_f32(0.0);
                let mut f = [[z; 4]; 4];
                let zi = vdupq_n_s32(0);
                let (mut a00, mut a01, mut a02, mut a03) = (zi, zi, zi, zi);
                let (mut a10, mut a11, mut a12, mut a13) = (zi, zi, zi, zi);
                let (mut a20, mut a21, mut a22, mut a23) = (zi, zi, zi, zi);
                let (mut a30, mut a31, mut a32, mut a33) = (zi, zi, zi, zi);
                for gi in 0..ng {
                    for qq in 0..gquads {
                        let q = gi * gquads + qq;
                        let c0 = vld1q_u8(wp[0].add(q * 16));
                        let c1 = vld1q_u8(wp[1].add(q * 16));
                        let c2 = vld1q_u8(wp[2].add(q * 16));
                        let c3 = vld1q_u8(wp[3].add(q * 16));
                        let x_off = q * 4 * 16;
                        mm16!(
                            a00,
                            a01,
                            a02,
                            a03,
                            a10,
                            a11,
                            a12,
                            a13,
                            a20,
                            a21,
                            a22,
                            a23,
                            a30,
                            a31,
                            a32,
                            a33,
                            unpack!(c0, 0),
                            unpack!(c1, 0),
                            unpack!(c2, 0),
                            unpack!(c3, 0),
                            vld1q_s8(xp[0].add(x_off)),
                            vld1q_s8(xp[1].add(x_off)),
                            vld1q_s8(xp[2].add(x_off)),
                            vld1q_s8(xp[3].add(x_off))
                        );
                        mm16!(
                            a00,
                            a01,
                            a02,
                            a03,
                            a10,
                            a11,
                            a12,
                            a13,
                            a20,
                            a21,
                            a22,
                            a23,
                            a30,
                            a31,
                            a32,
                            a33,
                            unpack!(c0, 1),
                            unpack!(c1, 1),
                            unpack!(c2, 1),
                            unpack!(c3, 1),
                            vld1q_s8(xp[0].add(x_off + 16)),
                            vld1q_s8(xp[1].add(x_off + 16)),
                            vld1q_s8(xp[2].add(x_off + 16)),
                            vld1q_s8(xp[3].add(x_off + 16))
                        );
                        mm16!(
                            a00,
                            a01,
                            a02,
                            a03,
                            a10,
                            a11,
                            a12,
                            a13,
                            a20,
                            a21,
                            a22,
                            a23,
                            a30,
                            a31,
                            a32,
                            a33,
                            unpack!(c0, 2),
                            unpack!(c1, 2),
                            unpack!(c2, 2),
                            unpack!(c3, 2),
                            vld1q_s8(xp[0].add(x_off + 32)),
                            vld1q_s8(xp[1].add(x_off + 32)),
                            vld1q_s8(xp[2].add(x_off + 32)),
                            vld1q_s8(xp[3].add(x_off + 32))
                        );
                        mm16!(
                            a00,
                            a01,
                            a02,
                            a03,
                            a10,
                            a11,
                            a12,
                            a13,
                            a20,
                            a21,
                            a22,
                            a23,
                            a30,
                            a31,
                            a32,
                            a33,
                            unpack!(c0, 3),
                            unpack!(c1, 3),
                            unpack!(c2, 3),
                            unpack!(c3, 3),
                            vld1q_s8(xp[0].add(x_off + 48)),
                            vld1q_s8(xp[1].add(x_off + 48)),
                            vld1q_s8(xp[2].add(x_off + 48)),
                            vld1q_s8(xp[3].add(x_off + 48))
                        );
                    }
                    let (sx, sw) = group_scales(x, w, gi, t, o, ng);
                    epilogue16!(
                        f, a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31,
                        a32, a33, sx, sw
                    );
                }
                store_tile(&f, y, t, o, out, n, ep);
                t += 8;
            }
            o += 8;
        }
        tc = tc_end;
    }
}
