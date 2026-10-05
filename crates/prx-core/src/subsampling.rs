//! dw-striding Conv2d subsampling, 8× in time (PLAN 2.6). HF `ParakeetEncoderSubsamplingConv2D`.
//!
//! Input `[T, F]` (time × mel) as one channel; output `[T/8, d_model]`.
use crate::format::ModelFile;
use crate::tensor::{linear, load_f32_shaped};
use crate::Result;

pub struct Conv2dDense {
    pub w: Vec<f32>, // [out, in, k, k]
    pub b: Vec<f32>,
    pub out_c: usize,
    pub in_c: usize,
}

pub struct Subsampling {
    pub channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub n_mels: usize,
    pub d_model: usize,
    conv0: Conv2dDense, // 1 → C, k3 s2 p1
    dw: [Vec<f32>; 2],  // depthwise [C,1,3,3]
    dw_b: [Vec<f32>; 2],
    pw: [Vec<f32>; 2], // pointwise [C,C,1,1]
    pw_b: [Vec<f32>; 2],
    lin_w: Vec<f32>, // [d_model, C * F_out]
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
        let conv0 = Conv2dDense {
            w: load_f32_shaped(m, &format!("{p}.0.weight"), &[c, 1, k, k])?,
            b: load_f32_shaped(m, &format!("{p}.0.bias"), &[c])?,
            out_c: c,
            in_c: 1,
        };
        let dw = [
            load_f32_shaped(m, &format!("{p}.2.weight"), &[c, 1, k, k])?,
            load_f32_shaped(m, &format!("{p}.5.weight"), &[c, 1, k, k])?,
        ];
        let dw_b = [
            load_f32_shaped(m, &format!("{p}.2.bias"), &[c])?,
            load_f32_shaped(m, &format!("{p}.5.bias"), &[c])?,
        ];
        let pw = [
            load_f32_shaped(m, &format!("{p}.3.weight"), &[c, c, 1, 1])?,
            load_f32_shaped(m, &format!("{p}.6.weight"), &[c, c, 1, 1])?,
        ];
        let pw_b = [
            load_f32_shaped(m, &format!("{p}.3.bias"), &[c])?,
            load_f32_shaped(m, &format!("{p}.6.bias"), &[c])?,
        ];
        let lin_w = load_f32_shaped(
            m,
            "encoder.subsampling.linear.weight",
            &[e.hidden_size, c * f_out],
        )?;
        let lin_b = load_f32_shaped(m, "encoder.subsampling.linear.bias", &[e.hidden_size])?;
        Ok(Self {
            channels: c,
            kernel: k,
            stride: s,
            n_mels: e.num_mel_bins,
            d_model: e.hidden_size,
            conv0,
            dw,
            dw_b,
            pw,
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
        // stage 0: full conv 1 → C over (T, F), ReLU. Layout [C, T, F].
        let (t1, f1) = (out_len(t, k, s, p), out_len(self.n_mels, k, s, p));
        let mut x = vec![0f32; c * t1 * f1];
        for oc in 0..c {
            let w = &self.conv0.w[oc * k * k..(oc + 1) * k * k];
            let b = self.conv0.b[oc];
            for ot in 0..t1 {
                for of in 0..f1 {
                    let mut acc = b;
                    for kt in 0..k {
                        let it = (ot * s + kt) as isize - p as isize;
                        if it < 0 || it >= t as isize {
                            continue;
                        }
                        for kf in 0..k {
                            let i_f = (of * s + kf) as isize - p as isize;
                            if i_f < 0 || i_f >= self.n_mels as isize {
                                continue;
                            }
                            acc += w[kt * k + kf] * feats[it as usize * self.n_mels + i_f as usize];
                        }
                    }
                    x[(oc * t1 + ot) * f1 + of] = acc.max(0.0);
                }
            }
        }
        let (mut tc, mut fc) = (t1, f1);
        for stage in 0..2 {
            // depthwise k3 s2 p1 per channel (no activation), then pointwise 1×1 across channels, ReLU
            let (t2, f2) = (out_len(tc, k, s, p), out_len(fc, k, s, p));
            let mut y = vec![0f32; c * t2 * f2];
            for ch in 0..c {
                let w = &self.dw[stage][ch * k * k..(ch + 1) * k * k];
                let b = self.dw_b[stage][ch];
                let xin = &x[ch * tc * fc..(ch + 1) * tc * fc];
                for ot in 0..t2 {
                    for of in 0..f2 {
                        let mut acc = b;
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
                                acc += w[kt * k + kf] * xin[it as usize * fc + i_f as usize];
                            }
                        }
                        y[(ch * t2 + ot) * f2 + of] = acc;
                    }
                }
            }
            // pointwise: z[oc, pos] = b[oc] + Σ_ic w[oc, ic] y[ic, pos]; ReLU
            let n_pos = t2 * f2;
            let mut z = vec![0f32; c * n_pos];
            let w = &self.pw[stage];
            for oc in 0..c {
                let zo = &mut z[oc * n_pos..(oc + 1) * n_pos];
                zo.iter_mut().for_each(|v| *v = self.pw_b[stage][oc]);
                for ic in 0..c {
                    let wv = w[oc * c + ic];
                    let yi = &y[ic * n_pos..(ic + 1) * n_pos];
                    for (zv, yv) in zo.iter_mut().zip(yi) {
                        *zv += wv * yv;
                    }
                }
                zo.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            x = z;
            tc = t2;
            fc = f2;
        }
        // flatten per time step: index = ch * fc + f  (HF: transpose(1,2).reshape(B, T, C*F))
        let flat_in = c * fc;
        let mut flat = vec![0f32; tc * flat_in];
        for ch in 0..c {
            for ot in 0..tc {
                for of in 0..fc {
                    flat[ot * flat_in + ch * fc + of] = x[(ch * tc + ot) * fc + of];
                }
            }
        }
        let mut out = vec![0f32; tc * self.d_model];
        linear(
            &flat,
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
