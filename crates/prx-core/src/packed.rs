//! Load-time repack of ternary weights (PLAN 3.3) and per-group int8 activation quantisation (D-3.2).
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
}

/// Ternary matrix `[out, in]` as int8 {-1,0,+1} plus f32 scales `[out, in/group]`.
pub struct TernaryMatrix {
    pub out: usize,
    pub inp: usize,
    pub group: usize,
    pub layout: Layout,
    pub w: Vec<i8>,       // `layout`-ordered, out*inp bytes
    pub scales: Vec<f32>, // row-major [out, inp/group]
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
        assert!(layout == Layout::Rows || (out % 2 == 0 && inp % 8 == 0));
        let rb = inp.div_ceil(5);
        let ng = inp / group;
        let mut w = vec![0i8; out * inp];
        let mut row = vec![0i8; inp];
        for r in 0..out {
            unpack_row(&qweight[r * rb..(r + 1) * rb], inp, &mut row);
            match layout {
                Layout::Rows => w[r * inp..(r + 1) * inp].copy_from_slice(&row),
                Layout::Pairs => {
                    let (rp, half) = (r / 2, r % 2);
                    for (sl, chunk) in row.chunks_exact(8).enumerate() {
                        let o = (rp * (inp / 8) + sl) * 16 + half * 8;
                        w[o..o + 8].copy_from_slice(chunk);
                    }
                }
            }
        }
        let scales = scales_f16
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(scales.len(), out * ng);
        Self {
            out,
            inp,
            group,
            layout,
            w,
            scales,
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
        let mut w = vec![0i8; out * inp];
        for r in 0..out {
            let row = &rows[r * inp..(r + 1) * inp];
            match layout {
                Layout::Rows => w[r * inp..(r + 1) * inp].copy_from_slice(row),
                Layout::Pairs => {
                    let (rp, half) = (r / 2, r % 2);
                    for (sl, chunk) in row.chunks_exact(8).enumerate() {
                        let o = (rp * (inp / 8) + sl) * 16 + half * 8;
                        w[o..o + 8].copy_from_slice(chunk);
                    }
                }
            }
        }
        Self {
            out,
            inp,
            group,
            layout,
            w,
            scales: scales.to_vec(),
        }
    }

    /// Element access independent of layout (reference paths only).
    #[inline]
    pub fn get(&self, r: usize, k: usize) -> i8 {
        match self.layout {
            Layout::Rows => self.w[r * self.inp + k],
            Layout::Pairs => self.w[((r / 2) * (self.inp / 8) + k / 8) * 16 + (r % 2) * 8 + k % 8],
        }
    }

    pub fn groups(&self) -> usize {
        self.inp / self.group
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
        let (qp, sp) = (q.as_mut_ptr() as usize, scales.as_mut_ptr() as usize);
        let do_token = |t: usize, tmp: &mut [f32], qrow: &mut [i8]| {
            fill(t, tmp);
            // SAFETY: token t owns its scale slots and its bytes in `q` (row t, or its half of pair t/2); tasks are disjoint.
            let ds = unsafe { std::slice::from_raw_parts_mut((sp as *mut f32).add(t * ng), ng) };
            match layout {
                Layout::Rows => {
                    let dq = unsafe {
                        std::slice::from_raw_parts_mut((qp as *mut i8).add(t * inp), inp)
                    };
                    for g in 0..ng {
                        ds[g] = quantize_group(
                            &tmp[g * group..(g + 1) * group],
                            &mut dq[g * group..(g + 1) * group],
                        );
                    }
                }
                Layout::Pairs => {
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
        }
    }

    /// Element access independent of layout (reference paths only).
    #[inline]
    pub fn get(&self, t: usize, k: usize) -> i8 {
        match self.layout {
            Layout::Rows => self.x[t * self.inp + k],
            Layout::Pairs => self.x[((t / 2) * (self.inp / 8) + k / 8) * 16 + (t % 2) * 8 + k % 8],
        }
    }
}

/// Quantise one group: returns the scale (absmax/127, 1.0 for an all-zero group) and writes int8 codes.
/// Round half to even (`frintn`), identical to the scalar `round_ties_even`; results are bit-identical across paths.
#[inline]
pub fn quantize_group(src: &[f32], dst: &mut [i8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64; tails are handled by the scalar loops inside.
    let amax = unsafe { absmax_neon(src) };
    #[cfg(not(target_arch = "aarch64"))]
    let amax = src.iter().fold(0f32, |m, v| m.max(v.abs()));
    let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
    let inv = 1.0 / s;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: as above.
    unsafe {
        quantize_neon(src, inv, dst)
    };
    #[cfg(not(target_arch = "aarch64"))]
    for (d, v) in dst.iter_mut().zip(src) {
        *d = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
    }
    s
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
