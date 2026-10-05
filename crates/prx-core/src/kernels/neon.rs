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
    let (inp, g, out, n) = (w.inp, w.group, w.out, x.n);
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
                let sx = *x.scales_t.get_unchecked(gi * n + t + c);
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
                tile_4x4(wp, xp, sw, x.scales_t.as_ptr().add(t), n, ng, g, yp, out);
            } else {
                tile_edge(w, x, y, o, rows, t, toks);
            }
            t += toks;
        }
        o += rows;
    }
}
