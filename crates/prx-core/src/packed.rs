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
    /// scratch buffer and quantised straight into `layout`, parallel over token pairs when the work is large.
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
        // quantise one f32 row into `dst` (a token's row for Rows, or the pair block for Pairs with `half`)
        let quant_into = |row: &[f32], sc: &mut [f32], write: &mut dyn FnMut(usize, i8)| {
            for g in 0..ng {
                let sg = &row[g * group..(g + 1) * group];
                let amax = sg.iter().fold(0f32, |m, v| m.max(v.abs()));
                let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
                sc[g] = s;
                let inv = 1.0 / s;
                for (c, v) in sg.iter().enumerate() {
                    write(
                        g * group + c,
                        (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8,
                    );
                }
            }
        };
        let parallel = n * inp >= crate::tensor::par_min_elems();
        match layout {
            Layout::Rows => {
                let work = |t: usize, dq: &mut [i8], ds: &mut [f32], tmp: &mut [f32]| {
                    fill(t, tmp);
                    quant_into(tmp, ds, &mut |k, v| dq[k] = v);
                };
                if parallel {
                    q.par_chunks_mut(inp)
                        .zip(scales.par_chunks_mut(ng))
                        .take(n)
                        .enumerate()
                        .for_each(|(t, (dq, ds))| {
                            let mut tmp = vec![0f32; inp];
                            work(t, dq, ds, &mut tmp);
                        });
                } else {
                    let mut tmp = vec![0f32; inp];
                    for t in 0..n {
                        work(
                            t,
                            &mut q[t * inp..(t + 1) * inp],
                            &mut scales[t * ng..(t + 1) * ng],
                            &mut tmp,
                        );
                    }
                }
            }
            Layout::Pairs => {
                let pair_len = slabs * 16;
                let work = |tp: usize, dst: &mut [i8], sc2: &mut [f32], tmp: &mut [f32]| {
                    for half in 0..2 {
                        let t = 2 * tp + half;
                        if t >= n {
                            break;
                        }
                        fill(t, tmp);
                        quant_into(tmp, &mut sc2[half * ng..(half + 1) * ng], &mut |k, v| {
                            dst[(k / 8) * 16 + half * 8 + k % 8] = v
                        });
                    }
                };
                if parallel {
                    q.par_chunks_mut(pair_len)
                        .zip(scales.par_chunks_mut(2 * ng))
                        .enumerate()
                        .for_each(|(tp, (dst, sc2))| {
                            let mut tmp = vec![0f32; inp];
                            work(tp, dst, sc2, &mut tmp);
                        });
                } else {
                    let mut tmp = vec![0f32; inp];
                    for tp in 0..n_pad / 2 {
                        work(
                            tp,
                            &mut q[tp * pair_len..(tp + 1) * pair_len],
                            &mut scales[tp * 2 * ng..(tp + 1) * 2 * ng],
                            &mut tmp,
                        );
                    }
                }
            }
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
