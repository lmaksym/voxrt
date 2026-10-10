//! Load-time repack of ternary weights and per-group int8 activation quantisation.
use half::f16;

use crate::trits::unpack_row;

/// Memory layout of the int8 weight matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Row-major `[out, inp]` (scalar reference, SDOT tile).
    Rows,
    /// Row pairs interleaved per 8-column slab for `smmla`: for pair `rp`, slab `s`: 16 bytes
    /// `[w[2rp][8s..8s+8], w[2rp+1][8s..8s+8]]`.
    Pairs,
    /// `Pairs` packed 4 trits per byte (ternary only): for pair `rp`, quad `q` (4 slabs = 32 columns), 16 bytes where
    /// byte `b`'s 2-bit field `j` is `code = trit + 1` of operand slab `4q + j`, position `b`. One 16-byte load unpacks
    /// (shift, and, sub) into the four `smmla` operands of the quad; 4× fewer weight bytes than `Pairs`.
    Packed2,
    /// Blocks of 8 rows interleaved per 16-column chunk: for row block `rb`, chunk `c`: 128 bytes
    /// `[w[8rb][16c..16c+16], …, w[8rb+7][16c..16c+16]]`. One sequential stream for the single-token GEMV
    /// (decoder weights, 3c.17); rows padded to a multiple of 8.
    Rows8,
    /// Blocks of 16 rows × 4 columns (x86 AVX2 / AVX-512 VNNI): for row block `rb`, column quad `q`: 64 bytes
    /// `[w[16rb][4q..4q+4], …, w[16rb+15][4q..4q+4]]`, so one load feeds a 4-byte dot product per row lane and a
    /// lane accumulates its row's group sum without horizontal adds. Rows padded to a multiple of 16 (zeros);
    /// activations in this layout are plain rows.
    Quad16,
}

/// Byte offset of `w[r][k]` in the `Quad16` plane of a matrix with `inp` columns.
#[inline]
pub fn quad16_offset(r: usize, k: usize, inp: usize) -> usize {
    ((r / 16) * (inp / 4) + k / 4) * 64 + (r % 16) * 4 + k % 4
}

impl Layout {
    /// Bytes of the weight plane for an `[out, inp]` matrix in this layout.
    pub fn plane_bytes(self, out: usize, inp: usize) -> usize {
        match self {
            Layout::Packed2 => out * inp / 4,
            Layout::Rows8 => out.div_ceil(8) * 8 * inp,
            Layout::Quad16 => out.div_ceil(16) * 16 * inp,
            _ => out * inp,
        }
    }
}

/// Ternary matrix `[out, in]` as int8 {-1,0,+1} plus f32 scales `[out, in/group]`.
pub struct TernaryMatrix {
    pub out: usize,
    pub inp: usize,
    pub group: usize,
    pub layout: Layout,
    pub w: Vec<i8>,       // `layout`-ordered, out*inp bytes
    pub scales: Vec<f32>, // row-major [out, inp/group]
    /// `Quad16` only: scales per row block and group, `[out/16][inp/group][16]` (padded rows 0), so a 16-row lane
    /// vector of scales is one load.
    pub scales_q: Vec<f32>,
    /// Weights are in {-1, 0, +1} (from the packed ternary format); arbitrary int8 otherwise.
    pub ternary: bool,
}

impl TernaryMatrix {
    pub fn from_packed(
        qweight: &[u8],
        scales_f16: &[u8],
        out: usize,
        inp: usize,
        group: usize,
    ) -> Self {
        Self::from_packed_layout(qweight, scales_f16, out, inp, group, Layout::Rows)
    }

    pub fn from_packed_layout(
        qweight: &[u8],
        scales_f16: &[u8],
        out: usize,
        inp: usize,
        group: usize,
        layout: Layout,
    ) -> Self {
        assert_eq!(inp % group, 0);
        assert!(
            layout != Layout::Rows8,
            "Rows8 is for int8 row matrices (from_i8_rows)"
        );
        assert!(
            matches!(layout, Layout::Rows | Layout::Quad16)
                || (out.is_multiple_of(2) && inp.is_multiple_of(8))
        );
        assert!(layout != Layout::Packed2 || inp.is_multiple_of(32));
        assert!(layout != Layout::Quad16 || inp.is_multiple_of(4));
        let rb = inp.div_ceil(5);
        let ng = inp / group;
        let mut w = vec![0i8; layout.plane_bytes(out, inp)];
        let mut row = vec![0i8; inp];
        for r in 0..out {
            unpack_row(&qweight[r * rb..(r + 1) * rb], inp, &mut row);
            match layout {
                Layout::Rows8 => unreachable!("rejected above"),
                Layout::Rows => w[r * inp..(r + 1) * inp].copy_from_slice(&row),
                Layout::Quad16 => put_quad16(&mut w, r, &row),
                Layout::Pairs => {
                    let (rp, half) = (r / 2, r % 2);
                    for (sl, chunk) in row.as_chunks::<8>().0.iter().enumerate() {
                        let o = (rp * (inp / 8) + sl) * 16 + half * 8;
                        w[o..o + 8].copy_from_slice(chunk);
                    }
                }
                Layout::Packed2 => {
                    let (rp, half) = (r / 2, r % 2);
                    let nq = inp / 32;
                    for (k, &t) in row.iter().enumerate() {
                        let (slab, pos) = (k / 8, k % 8);
                        let (q, j) = (slab / 4, slab % 4);
                        let b = half * 8 + pos;
                        let o = (rp * nq + q) * 16 + b;
                        w[o] |= (((t + 1) as u8) << (2 * j)) as i8;
                    }
                }
            }
        }
        let scales = scales_f16
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(scales.len(), out * ng);
        Self::finish(out, inp, group, layout, w, scales, true)
    }

    fn finish(
        out: usize,
        inp: usize,
        group: usize,
        layout: Layout,
        w: Vec<i8>,
        scales: Vec<f32>,
        ternary: bool,
    ) -> Self {
        let ng = inp / group;
        let mut scales_q = Vec::new();
        if layout == Layout::Quad16 {
            scales_q = vec![0f32; out.div_ceil(16) * ng * 16];
            for r in 0..out {
                for g in 0..ng {
                    scales_q[((r / 16) * ng + g) * 16 + r % 16] = scales[r * ng + g];
                }
            }
        }
        Self {
            out,
            inp,
            group,
            layout,
            w,
            scales,
            scales_q,
            ternary,
        }
    }

    /// Build from arbitrary int8 rows `[out, inp]` (not necessarily ternary) with scales `[out, inp/group]`.
    pub fn from_i8_rows(
        rows: &[i8],
        scales: &[f32],
        out: usize,
        inp: usize,
        group: usize,
        layout: Layout,
    ) -> Self {
        assert_eq!(rows.len(), out * inp);
        assert_eq!(scales.len(), out * (inp / group));
        assert!(layout != Layout::Packed2, "Packed2 is ternary-only");
        assert!(layout != Layout::Rows8 || inp.is_multiple_of(16));
        assert!(layout != Layout::Quad16 || inp.is_multiple_of(4));
        let mut w = vec![0i8; layout.plane_bytes(out, inp)];
        for r in 0..out {
            let row = &rows[r * inp..(r + 1) * inp];
            match layout {
                Layout::Rows => w[r * inp..(r + 1) * inp].copy_from_slice(row),
                Layout::Quad16 => put_quad16(&mut w, r, row),
                Layout::Packed2 => unreachable!("Packed2 is ternary-only"),
                Layout::Rows8 => {
                    let (rb, sub) = (r / 8, r % 8);
                    for (c, chunk) in row.as_chunks::<16>().0.iter().enumerate() {
                        let o = (rb * (inp / 16) + c) * 128 + sub * 16;
                        w[o..o + 16].copy_from_slice(chunk);
                    }
                }
                Layout::Pairs => {
                    let (rp, half) = (r / 2, r % 2);
                    for (sl, chunk) in row.as_chunks::<8>().0.iter().enumerate() {
                        let o = (rp * (inp / 8) + sl) * 16 + half * 8;
                        w[o..o + 8].copy_from_slice(chunk);
                    }
                }
            }
        }
        Self::finish(out, inp, group, layout, w, scales.to_vec(), false)
    }

    /// Quantise f32 rows produced by `fill(r, row)` into an int8 matrix (absmax/127 per (row, group), like
    /// `QuantAct`): activations used as the *weight* operand of a GEMM, e.g. keys and relative positions in the
    /// attention core. Single-threaded; callers parallelise across heads.
    pub fn from_f32_rows<F: FnMut(usize, &mut [f32])>(
        out: usize,
        inp: usize,
        group: usize,
        layout: Layout,
        mut fill: F,
    ) -> Self {
        assert!(layout != Layout::Packed2, "Packed2 is ternary-only");
        assert!(layout != Layout::Rows8, "Rows8: use from_i8_rows");
        assert!(
            matches!(layout, Layout::Rows | Layout::Quad16)
                || (out.is_multiple_of(2) && inp.is_multiple_of(8))
        );
        assert!(layout != Layout::Quad16 || inp.is_multiple_of(4));
        assert_eq!(inp % group, 0);
        let ng = inp / group;
        let slabs = inp / 8;
        let mut w = vec![0i8; layout.plane_bytes(out, inp)];
        let mut scales = vec![0f32; out * ng];
        let mut tmp = vec![0f32; inp];
        let mut qrow = vec![0i8; inp];
        for r in 0..out {
            fill(r, &mut tmp);
            for g in 0..ng {
                scales[r * ng + g] = quantize_group(
                    &tmp[g * group..(g + 1) * group],
                    &mut qrow[g * group..(g + 1) * group],
                );
            }
            match layout {
                Layout::Rows => w[r * inp..(r + 1) * inp].copy_from_slice(&qrow),
                Layout::Quad16 => put_quad16(&mut w, r, &qrow),
                Layout::Pairs => {
                    let (rp, half) = (r / 2, r % 2);
                    for sl in 0..slabs {
                        let o = (rp * slabs + sl) * 16 + half * 8;
                        w[o..o + 8].copy_from_slice(&qrow[sl * 8..sl * 8 + 8]);
                    }
                }
                Layout::Packed2 | Layout::Rows8 => unreachable!(),
            }
        }
        Self::finish(out, inp, group, layout, w, scales, false)
    }

    /// Element access independent of layout (reference paths only).
    #[inline]
    pub fn get(&self, r: usize, k: usize) -> i8 {
        match self.layout {
            Layout::Rows => self.w[r * self.inp + k],
            Layout::Rows8 => {
                self.w[((r / 8) * (self.inp / 16) + k / 16) * 128 + (r % 8) * 16 + k % 16]
            }
            Layout::Pairs => self.w[((r / 2) * (self.inp / 8) + k / 8) * 16 + (r % 2) * 8 + k % 8],
            Layout::Packed2 => {
                let (slab, pos) = (k / 8, k % 8);
                let (q, j) = (slab / 4, slab % 4);
                let byte = self.w[((r / 2) * (self.inp / 32) + q) * 16 + (r % 2) * 8 + pos] as u8;
                ((byte >> (2 * j)) & 3) as i8 - 1
            }
            Layout::Quad16 => self.w[quad16_offset(r, k, self.inp)],
        }
    }

    pub fn groups(&self) -> usize {
        self.inp / self.group
    }
}

/// Write row `r` (`inp` columns) into a `Quad16` plane.
fn put_quad16(w: &mut [i8], r: usize, row: &[i8]) {
    let inp = row.len();
    for (q, quad) in row.as_chunks::<4>().0.iter().enumerate() {
        let o = ((r / 16) * (inp / 4) + q) * 64 + (r % 16) * 4;
        w[o..o + 4].copy_from_slice(quad);
    }
}

/// Quantised activations int8 with scales `[n, inp/group]`, stored in the kernel's `Layout`
/// (`Rows`: `[n, inp]`; `Pairs`: token pairs interleaved per 8-column slab, `n_pad` tokens).
pub struct QuantAct {
    pub n: usize,
    pub inp: usize,
    pub group: usize,
    pub layout: Layout,
    pub n_pad: usize,       // n rounded up to a multiple of 8 (zero tokens appended)
    pub x: Vec<i8>,         // `layout`-ordered, n_pad*inp bytes
    pub scales: Vec<f32>,   // [n, inp/group]
    pub scales_t: Vec<f32>, // transposed [inp/group, n_pad] — contiguous per group for the SIMD tiles
    /// `Quad16` (x86 kernels) only: Σ of the int8 codes per (token, group), `[n, inp/group]`, for the unsigned-operand
    /// offset correction (`Σ (w + c)·x = Σ w·x + c·Σ x`).
    pub gsum: Vec<i32>,
}

impl QuantAct {
    /// Row layout (reference paths, SDOT tile).
    pub fn quantize(x: &[f32], n: usize, inp: usize, group: usize) -> Self {
        Self::quantize_layout(x, n, inp, group, Layout::Rows)
    }

    /// Symmetric int8, absmax/127 per (token, group); round half to even like `torch.round`.
    pub fn quantize_layout(x: &[f32], n: usize, inp: usize, group: usize, layout: Layout) -> Self {
        debug_assert_eq!(x.len(), n * inp);
        Self::from_rows(n, inp, group, layout, |t, row| {
            row.copy_from_slice(&x[t * inp..(t + 1) * inp])
        })
    }

    /// Build from a row producer `fill(t, row)` (e.g. a fused LayerNorm): each token's f32 row is produced into a
    /// thread-local scratch buffer, quantised with NEON into a scratch int8 row, then copied into `layout` by
    /// 8-byte slabs. Parallel over chunks of token pairs when the work is large.
    pub fn from_rows<F: Fn(usize, &mut [f32]) + Sync>(
        n: usize,
        inp: usize,
        group: usize,
        layout: Layout,
        fill: F,
    ) -> Self {
        use rayon::prelude::*;
        let ng = inp / group;
        let n_pad = n.div_ceil(8) * 8;
        let mut q = vec![0i8; n_pad * inp];
        let mut scales = vec![0f32; n_pad * ng];
        let slabs = inp / 8;
        // Quad16 (x86): Σ of the codes per (token, group), filled per token inside the parallel pass
        let mut gsum = vec![0i32; if layout == Layout::Quad16 { n * ng } else { 0 }];
        let (qp, sp) = (q.as_mut_ptr() as usize, scales.as_mut_ptr() as usize);
        let gp = gsum.as_mut_ptr() as usize;
        let do_token = |t: usize, tmp: &mut [f32], qrow: &mut [i8]| {
            fill(t, tmp);
            // SAFETY: token t owns its scale slots and its bytes in `q` (row t, or its half of pair t/2); tasks are disjoint.
            let ds = unsafe { std::slice::from_raw_parts_mut((sp as *mut f32).add(t * ng), ng) };
            match layout {
                Layout::Rows | Layout::Rows8 | Layout::Quad16 => {
                    let dq = unsafe {
                        std::slice::from_raw_parts_mut((qp as *mut i8).add(t * inp), inp)
                    };
                    for g in 0..ng {
                        ds[g] = quantize_group(
                            &tmp[g * group..(g + 1) * group],
                            &mut dq[g * group..(g + 1) * group],
                        );
                    }
                    if layout == Layout::Quad16 {
                        // SAFETY: token t owns gsum[t·ng .. (t+1)·ng].
                        let gs = unsafe {
                            std::slice::from_raw_parts_mut((gp as *mut i32).add(t * ng), ng)
                        };
                        for (g, slot) in gs.iter_mut().enumerate() {
                            *slot = dq[g * group..(g + 1) * group]
                                .iter()
                                .map(|&v| v as i32)
                                .sum();
                        }
                    }
                }
                Layout::Pairs | Layout::Packed2 => {
                    // activations are never 2-bit packed: Packed2 weights consume `Pairs` activations
                    for g in 0..ng {
                        ds[g] = quantize_group(
                            &tmp[g * group..(g + 1) * group],
                            &mut qrow[g * group..(g + 1) * group],
                        );
                    }
                    let (tp, half) = (t / 2, t % 2);
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            (qp as *mut i8).add(tp * slabs * 16),
                            slabs * 16,
                        )
                    };
                    for sl in 0..slabs {
                        dst[sl * 16 + half * 8..sl * 16 + half * 8 + 8]
                            .copy_from_slice(&qrow[sl * 8..sl * 8 + 8]);
                    }
                }
            }
        };
        thread_local! {
            static SCRATCH: std::cell::RefCell<(Vec<f32>, Vec<i8>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
        }
        let with_scratch = |f: &dyn Fn(&mut [f32], &mut [i8])| {
            SCRATCH.with(|c| {
                let mut b = c.borrow_mut();
                if b.0.len() < inp {
                    b.0.resize(inp, 0.0);
                    b.1.resize(inp, 0);
                }
                let (tmp, qrow) = &mut *b;
                f(&mut tmp[..inp], &mut qrow[..inp]);
            })
        };
        if n * inp >= crate::tensor::par_min_elems() {
            // chunks of whole token pairs so two tasks never share a pair block
            let pairs = n.div_ceil(2);
            let per_pairs = pairs.div_ceil(crate::tensor::threads() * 4).max(1);
            (0..pairs.div_ceil(per_pairs))
                .into_par_iter()
                .for_each(|c| {
                    with_scratch(&|tmp, qrow| {
                        for t in c * per_pairs * 2..((c + 1) * per_pairs * 2).min(n) {
                            do_token(t, tmp, qrow);
                        }
                    })
                });
        } else {
            with_scratch(&|tmp, qrow| {
                for t in 0..n {
                    do_token(t, tmp, qrow);
                }
            });
        }
        let mut scales_t = vec![0f32; ng * n_pad];
        for t in 0..n {
            for g in 0..ng {
                scales_t[g * n_pad + t] = scales[t * ng + g];
            }
        }
        scales.truncate(n * ng);
        Self {
            n,
            inp,
            group,
            layout,
            n_pad,
            x: q,
            scales,
            scales_t,
            gsum,
        }
    }

    /// Element access independent of layout (reference paths only).
    #[inline]
    pub fn get(&self, t: usize, k: usize) -> i8 {
        match self.layout {
            Layout::Rows | Layout::Rows8 | Layout::Quad16 => self.x[t * self.inp + k],
            Layout::Pairs | Layout::Packed2 => {
                self.x[((t / 2) * (self.inp / 8) + k / 8) * 16 + (t % 2) * 8 + k % 8]
            }
        }
    }
}

/// Quantise one group: returns the scale (absmax/127, 1.0 for an all-zero group) and writes int8 codes.
/// Round half to even (`frintn`), identical to the scalar `round_ties_even`; results are bit-identical across paths.
#[inline]
pub fn quantize_group(src: &[f32], dst: &mut [i8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; tails are handled by the scalar loops inside.
        let amax = unsafe { absmax_neon(src) };
        let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        // SAFETY: as above.
        unsafe { quantize_neon(src, 1.0 / s, dst) };
        s
    }
    #[cfg(target_arch = "x86_64")]
    if crate::tensor::x86_fma() {
        // SAFETY: AVX2 detected.
        return unsafe { quantize_group_avx2(src, dst) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    quantize_group_portable(src, dst)
}

/// AVX2 `quantize_group`: the absmax is order-independent and `vroundps` rounds half to even, so the scale and codes
/// equal the portable path's (|v·inv| ≤ 127 after rounding; the saturating packs never clamp).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_group_avx2(src: &[f32], dst: &mut [i8]) -> f32 {
    use std::arch::x86_64::*;
    let n = src.len();
    let p = src.as_ptr();
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut m = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        m = _mm256_max_ps(m, _mm256_and_ps(_mm256_loadu_ps(p.add(i)), abs_mask));
        i += 8;
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), m);
    let mut amax = lanes.iter().fold(0f32, |a, &v| a.max(v));
    while i < n {
        amax = amax.max(src[i].abs());
        i += 1;
    }
    let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
    let inv = 1.0 / s;
    let vinv = _mm256_set1_ps(inv);
    let q = dst.as_mut_ptr();
    i = 0;
    while i + 16 <= n {
        let rnd = |v: __m256| {
            _mm256_cvtps_epi32(_mm256_round_ps::<
                { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
            >(_mm256_mul_ps(v, vinv)))
        };
        let a = rnd(_mm256_loadu_ps(p.add(i)));
        let b = rnd(_mm256_loadu_ps(p.add(i + 8)));
        // packs work per 128-bit lane: [a0..3 b0..3 | a4..7 b4..7] -> fix the order with a 64-bit permute
        let w16 = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(a, b));
        let b8 = _mm_packs_epi16(
            _mm256_castsi256_si128(w16),
            _mm256_extracti128_si256::<1>(w16),
        );
        _mm_storeu_si128(q.add(i) as *mut __m128i, b8);
        i += 16;
    }
    while i < n {
        dst[i] = (src[i] * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
        i += 1;
    }
    s
}

crate::tensor::fma_dispatch! {
/// Fallback of `quantize_group` (same scale and rounding as the NEON path; `roundps` instead of a libm `rintf` call
/// on AVX2 machines).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn quantize_group_portable(src: &[f32], dst: &mut [i8]) -> f32 {
    let amax = src.iter().fold(0f32, |m, v| m.max(v.abs()));
    let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
    let inv = 1.0 / s;
    for (d, v) in dst.iter_mut().zip(src) {
        *d = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
    }
    s
}
}

#[cfg(target_arch = "aarch64")]
unsafe fn absmax_neon(src: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let mut m = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 4 <= src.len() {
        m = vmaxq_f32(m, vabsq_f32(vld1q_f32(src.as_ptr().add(i))));
        i += 4;
    }
    let mut amax = vmaxvq_f32(m);
    while i < src.len() {
        amax = amax.max(src[i].abs());
        i += 1;
    }
    amax
}

#[cfg(target_arch = "aarch64")]
unsafe fn quantize_neon(src: &[f32], inv: f32, dst: &mut [i8]) {
    use std::arch::aarch64::*;
    let vinv = vdupq_n_f32(inv);
    let mut i = 0;
    while i + 8 <= src.len() {
        // frintn = round to nearest, ties to even; |v·inv| ≤ 127 so the saturating narrows never clamp
        let a = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vld1q_f32(src.as_ptr().add(i)), vinv)));
        let b = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(
            vld1q_f32(src.as_ptr().add(i + 4)),
            vinv,
        )));
        let s8 = vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)));
        vst1_s8(dst.as_mut_ptr().add(i), s8);
        i += 8;
    }
    while i < src.len() {
        dst[i] = (src[i] * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
        i += 1;
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod portable_tests {
    #[test]
    fn neon_quantize_equals_portable() {
        let mut seed = 3u64;
        for n in [1usize, 7, 8, 128, 133, 384] {
            let src: Vec<f32> = (0..n)
                .map(|i| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    // include exact .5 ties after scaling
                    if i % 5 == 0 {
                        (i as f32 + 0.5) * 0.1
                    } else {
                        ((seed >> 40) as f32 / 1e6) - 8.0
                    }
                })
                .collect();
            let (mut a, mut b) = (vec![0i8; n], vec![0i8; n]);
            let sa = super::quantize_group(&src, &mut a);
            let sb = super::quantize_group_portable(&src, &mut b);
            assert_eq!((sa.to_bits(), &a), (sb.to_bits(), &b), "n={n}");
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod x86_tests {
    #[test]
    fn avx2_quantize_equals_portable() {
        if !crate::tensor::x86_fma() {
            eprintln!("no AVX2/FMA; skipped");
            return;
        }
        let mut seed = 3u64;
        for n in [1usize, 7, 8, 15, 16, 17, 128, 133, 384, 4096] {
            let src: Vec<f32> = (0..n)
                .map(|i| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    if i % 5 == 0 {
                        (i as f32 + 0.5) * 0.1
                    } else {
                        ((seed >> 40) as f32 / 1e6) - 8.0
                    }
                })
                .collect();
            let (mut a, mut b) = (vec![0i8; n], vec![0i8; n]);
            let sa = unsafe { super::quantize_group_avx2(&src, &mut a) };
            let sb = super::quantize_group_portable(&src, &mut b);
            assert_eq!((sa.to_bits(), &a), (sb.to_bits(), &b), "n={n}");
        }
    }
}
