//! Load-time repack of ternary weights (PLAN 3.3) and per-group int8 activation quantisation (D-3.2).
use half::f16;

use crate::trits::unpack_row;

/// Ternary matrix `[out, in]` as int8 {-1,0,+1} rows plus f32 scales `[out, in/group]`.
pub struct TernaryMatrix {
    pub out: usize,
    pub inp: usize,
    pub group: usize,
    pub w: Vec<i8>,       // row-major [out, inp]
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
        assert_eq!(inp % group, 0);
        let rb = inp.div_ceil(5);
        let ng = inp / group;
        let mut w = vec![0i8; out * inp];
        for r in 0..out {
            unpack_row(
                &qweight[r * rb..(r + 1) * rb],
                inp,
                &mut w[r * inp..(r + 1) * inp],
            );
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
            w,
            scales,
        }
    }

    pub fn groups(&self) -> usize {
        self.inp / self.group
    }
}

/// Quantised activations `[n, inp]` int8 with scales `[n, inp/group]`.
pub struct QuantAct {
    pub n: usize,
    pub inp: usize,
    pub group: usize,
    pub x: Vec<i8>,
    pub scales: Vec<f32>,   // [n, inp/group]
    pub scales_t: Vec<f32>, // transposed [inp/group, n] — contiguous per group for the SIMD tile
}

impl QuantAct {
    /// Symmetric int8, absmax/127 per (token, group); round half to even like `torch.round`.
    pub fn quantize(x: &[f32], n: usize, inp: usize, group: usize) -> Self {
        debug_assert_eq!(x.len(), n * inp);
        let ng = inp / group;
        let mut q = vec![0i8; n * inp];
        let mut scales = vec![0f32; n * ng];
        for t in 0..n {
            for g in 0..ng {
                let src = &x[t * inp + g * group..t * inp + (g + 1) * group];
                let amax = src.iter().fold(0f32, |m, v| m.max(v.abs()));
                let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
                scales[t * ng + g] = s;
                let inv = 1.0 / s;
                let dst = &mut q[t * inp + g * group..t * inp + (g + 1) * group];
                for (d, v) in dst.iter_mut().zip(src) {
                    *d = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
                }
            }
        }
        let mut scales_t = vec![0f32; ng * n];
        for t in 0..n {
            for g in 0..ng {
                scales_t[g * n + t] = scales[t * ng + g];
            }
        }
        Self {
            n,
            inp,
            group,
            x: q,
            scales,
            scales_t,
        }
    }
}
