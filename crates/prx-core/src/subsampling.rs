//! dw-striding Conv2d subsampling, 8× in time (PLAN 2.6, 3c.6). HF `ParakeetEncoderSubsamplingConv2D`.
//!
//! Channels-last layout `[T, F, C]` throughout: the 3×3 stride-2 convs vectorise over the channel axis and the
//! 1×1 pointwise convs become plain GEMMs on the f32 NEON tile. Input `[T, F]` (time × mel); output `[T/8, d_model]`.
use crate::format::ModelFile;
use crate::tensor::{linear, load_f32_shaped, par_rows};
use crate::Result;

pub struct Subsampling {
    pub channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub n_mels: usize,
    pub d_model: usize,
    conv0_w: Vec<f32>, // [k*k, C]  (tap-major, channel-minor)
    conv0_b: Vec<f32>,
    dw_w: [Vec<f32>; 2], // [k*k, C]
    dw_b: [Vec<f32>; 2],
    pw_w: [Vec<f32>; 2], // [C_out, C_in] for `linear`
    pw_b: [Vec<f32>; 2],
    lin_w: Vec<f32>, // [d_model, F3*C] with columns permuted to channels-last order (f*C + c)
    lin_b: Vec<f32>,
}

#[inline]
fn out_len(n: usize, k: usize, s: usize, p: usize) -> usize {
    (n + 2 * p - k) / s + 1
}

impl Subsampling {
    pub fn load(m: &ModelFile) -> Result<Self> {
        let e = &m.config.encoder_config;
        let (c, k, s) = (
            e.subsampling_conv_channels,
            e.subsampling_conv_kernel_size,
            e.subsampling_conv_stride,
        );
        let f_out = e.num_mel_bins / e.subsampling_factor;
        let p = "encoder.subsampling.layers";
        // conv0 [C,1,k,k] → [k*k, C]
        let w0 = load_f32_shaped(m, &format!("{p}.0.weight"), &[c, 1, k, k])?;
        let mut conv0_w = vec![0f32; k * k * c];
        for ch in 0..c {
            for tap in 0..k * k {
                conv0_w[tap * c + ch] = w0[ch * k * k + tap];
            }
        }
        let conv0_b = load_f32_shaped(m, &format!("{p}.0.bias"), &[c])?;
        let mut dw_w: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        let mut dw_b: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        let mut pw_w: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        let mut pw_b: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        for (si, (dwi, pwi)) in [(2usize, 3usize), (5, 6)].into_iter().enumerate() {
            let w = load_f32_shaped(m, &format!("{p}.{dwi}.weight"), &[c, 1, k, k])?;
            let mut t = vec![0f32; k * k * c];
            for ch in 0..c {
                for tap in 0..k * k {
                    t[tap * c + ch] = w[ch * k * k + tap];
                }
            }
            dw_w[si] = t;
            dw_b[si] = load_f32_shaped(m, &format!("{p}.{dwi}.bias"), &[c])?;
            pw_w[si] = load_f32_shaped(m, &format!("{p}.{pwi}.weight"), &[c, c, 1, 1])?; // [out, in]
            pw_b[si] = load_f32_shaped(m, &format!("{p}.{pwi}.bias"), &[c])?;
        }
        // final linear expects HF flatten order c*F3 + f; permute its columns to channels-last f*C + c
        let lw = load_f32_shaped(
            m,
            "encoder.subsampling.linear.weight",
            &[e.hidden_size, c * f_out],
        )?;
        let mut lin_w = vec![0f32; e.hidden_size * c * f_out];
        for o in 0..e.hidden_size {
            for ch in 0..c {
                for f in 0..f_out {
                    lin_w[o * c * f_out + f * c + ch] = lw[o * c * f_out + ch * f_out + f];
                }
            }
        }
        let lin_b = load_f32_shaped(m, "encoder.subsampling.linear.bias", &[e.hidden_size])?;
        Ok(Self {
            channels: c,
            kernel: k,
            stride: s,
            n_mels: e.num_mel_bins,
            d_model: e.hidden_size,
            conv0_w,
            conv0_b,
            dw_w,
            dw_b,
            pw_w,
            pw_b,
            lin_w,
            lin_b,
        })
    }

    /// Output time length for `t` input frames.
    pub fn out_frames(&self, t: usize) -> usize {
        let p = (self.kernel - 1) / 2;
        let mut n = t;
        for _ in 0..3 {
            n = out_len(n, self.kernel, self.stride, p);
        }
        n
    }

    /// `feats`: `[t, n_mels]` → `[out_frames(t), d_model]`.
    pub fn forward(&self, feats: &[f32], t: usize) -> Vec<f32> {
        let (c, k, s) = (self.channels, self.kernel, self.stride);
        let p = (k - 1) / 2;
        // stage 0: 1 → C channels over (T, F), ReLU; output [t1, f1, C]
        let (t1, f1) = (out_len(t, k, s, p), out_len(self.n_mels, k, s, p));
        let mut x = vec![0f32; t1 * f1 * c];
        {
            let (w, b, nm) = (&self.conv0_w, &self.conv0_b, self.n_mels);
            par_rows(&mut x, f1 * c, |ot, row| {
                for of in 0..f1 {
                    let o = &mut row[of * c..(of + 1) * c];
                    o.copy_from_slice(b);
                    for kt in 0..k {
                        let it = (ot * s + kt) as isize - p as isize;
                        if it < 0 || it >= t as isize {
                            continue;
                        }
                        for kf in 0..k {
                            let i_f = (of * s + kf) as isize - p as isize;
                            if i_f < 0 || i_f >= nm as isize {
                                continue;
                            }
                            let v = feats[it as usize * nm + i_f as usize];
                            let wt = &w[(kt * k + kf) * c..(kt * k + kf + 1) * c];
                            for ch in 0..c {
                                o[ch] += wt[ch] * v;
                            }
                        }
                    }
                    o.iter_mut().for_each(|v| *v = v.max(0.0));
                }
            });
        }
        let (mut tc, mut fc) = (t1, f1);
        for stage in 0..2 {
            // depthwise 3×3 stride 2 (no activation): [t2, f2, C], vectorised over channels
            let (t2, f2) = (out_len(tc, k, s, p), out_len(fc, k, s, p));
            let mut y = vec![0f32; t2 * f2 * c];
            {
                let (w, b, xin) = (&self.dw_w[stage], &self.dw_b[stage], &x);
                par_rows(&mut y, f2 * c, |ot, row| {
                    for of in 0..f2 {
                        let o = &mut row[of * c..(of + 1) * c];
                        o.copy_from_slice(b);
                        for kt in 0..k {
                            let it = (ot * s + kt) as isize - p as isize;
                            if it < 0 || it >= tc as isize {
                                continue;
                            }
                            for kf in 0..k {
                                let i_f = (of * s + kf) as isize - p as isize;
                                if i_f < 0 || i_f >= fc as isize {
                                    continue;
                                }
                                let xi = &xin[(it as usize * fc + i_f as usize) * c
                                    ..(it as usize * fc + i_f as usize + 1) * c];
                                let wt = &w[(kt * k + kf) * c..(kt * k + kf + 1) * c];
                                for ch in 0..c {
                                    o[ch] += wt[ch] * xi[ch];
                                }
                            }
                        }
                    }
                });
            }
            // pointwise 1×1 as a GEMM over positions, then ReLU
            let n_pos = t2 * f2;
            let mut z = vec![0f32; n_pos * c];
            linear(
                &y,
                n_pos,
                c,
                &self.pw_w[stage],
                Some(&self.pw_b[stage]),
                c,
                &mut z,
            );
            z.iter_mut().for_each(|v| *v = v.max(0.0));
            x = z;
            tc = t2;
            fc = f2;
        }
        // [t3, f3*C] is already the channels-last flatten; the linear's columns were permuted to match
        let flat_in = fc * c;
        let mut out = vec![0f32; tc * self.d_model];
        linear(
            &x,
            tc,
            flat_in,
            &self.lin_w,
            Some(&self.lin_b),
            self.d_model,
            &mut out,
        );
        out
    }
}
