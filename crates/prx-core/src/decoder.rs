//! TDT prediction network, joint and greedy loop (PLAN 2.10). Semantics: `docs/tdt.md` (NeMo rule).
use crate::format::ModelFile;
use crate::packed::{quantize_group, QuantAct, TernaryMatrix};
use crate::tensor::{gemv_f16, linear, load_f16_bits, load_f32_shaped, sigmoid};
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Emission {
    pub token: usize,
    pub frame: usize,
    pub duration: usize,
}

/// One greedy step (for golden comparison): frame, argmax token, chosen duration, joint logits, candidate LSTM state.
pub struct Step {
    pub t: usize,
    pub token: usize,
    pub duration: usize,
    pub logits: Vec<f32>,
    pub h: Vec<f32>, // [layers * hidden]
    pub c: Vec<f32>,
}

struct LstmLayer {
    w_ih: Vec<u16>,                 // [4h, in], f16 bits (memory-bound per-step GEMV)
    w_hh: Vec<u16>,                 // [4h, h]
    b: Vec<f32>,                    // b_ih + b_hh, [4h]
    w_ih_i8: Option<TernaryMatrix>, // per-row int8 (PRX_DECODER=int8)
    w_hh_i8: Option<TernaryMatrix>,
}

#[derive(Clone)]
pub struct LstmState {
    pub h: Vec<Vec<f32>>,
    pub c: Vec<Vec<f32>>,
}

pub struct Decoder {
    pub blank: usize,
    pub vocab: usize,
    pub hidden: usize,
    pub durations: Vec<usize>,
    pub max_symbols: usize,
    embedding: Vec<f32>, // [vocab, hidden]
    lstm: Vec<LstmLayer>,
    proj_w: Vec<u16>, // decoder_projector [h, h], f16 bits
    proj_b: Vec<f32>,
    enc_w: Vec<f32>, // encoder_projector [h, d]
    enc_b: Vec<f32>,
    head_w: Vec<u16>, // [vocab + n_dur, h], f16 bits
    head_b: Vec<f32>,
    head_i8: Option<TernaryMatrix>, // per-row int8 head (PRX_HEAD=int8): halves the per-step weight stream
    d_model: usize,
}

impl Decoder {
    pub fn load(m: &ModelFile) -> Result<Self> {
        let c = &m.config;
        let (h, v, d) = (
            c.decoder_hidden_size,
            c.vocab_size,
            c.encoder_config.hidden_size,
        );
        let embedding = load_f32_shaped(m, "decoder.embedding.weight", &[v, h])?;
        let mut lstm = Vec::new();
        for l in 0..c.num_decoder_layers {
            let w_ih = load_f16_bits(m, &format!("decoder.lstm.weight_ih_l{l}"), &[4 * h, h])?;
            let w_hh = load_f16_bits(m, &format!("decoder.lstm.weight_hh_l{l}"), &[4 * h, h])?;
            let b_ih = load_f32_shaped(m, &format!("decoder.lstm.bias_ih_l{l}"), &[4 * h])?;
            let b_hh = load_f32_shaped(m, &format!("decoder.lstm.bias_hh_l{l}"), &[4 * h])?;
            let b = b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect();
            lstm.push(LstmLayer {
                w_ih,
                w_hh,
                b,
                w_ih_i8: None,
                w_hh_i8: None,
            });
        }
        let n_out = v + c.durations.len();
        let mut dec = Self {
            blank: c.blank_token_id,
            vocab: v,
            hidden: h,
            durations: c.durations.clone(),
            max_symbols: c.max_symbols_per_step,
            embedding,
            lstm,
            proj_w: load_f16_bits(m, "decoder.decoder_projector.weight", &[h, h])?,
            proj_b: load_f32_shaped(m, "decoder.decoder_projector.bias", &[h])?,
            enc_w: load_f32_shaped(m, "encoder_projector.weight", &[h, d])?,
            enc_b: load_f32_shaped(m, "encoder_projector.bias", &[h])?,
            head_w: load_f16_bits(m, "joint.head.weight", &[n_out, h])?,
            head_i8: None,
            head_b: load_f32_shaped(m, "joint.head.bias", &[n_out])?,
            d_model: d,
        };
        // PRX_DECODER=int8: per-row int8 joint head and LSTM weights (halves the per-step weight stream; argmax
        // decisions can move on near-ties, validated on the token oracle). PRX_HEAD=int8 quantises the head only.
        // default int8 (test-clean WER 1.95 vs 1.96 f32, D-3c.7); `PRX_DECODER=f16` restores the exact f16 path
        let mode = std::env::var("PRX_DECODER").unwrap_or_else(|_| "int8".to_string());
        let int8 = mode == "int8";
        if int8 || std::env::var("PRX_HEAD").as_deref() == Ok("int8") {
            dec.head_i8 = Some(int8_rows(&dec.head_w, n_out, h));
        }
        if int8 {
            for l in dec.lstm.iter_mut() {
                l.w_ih_i8 = Some(int8_rows(&l.w_ih, 4 * h, h));
                l.w_hh_i8 = Some(int8_rows(&l.w_hh, 4 * h, h));
            }
        }
        Ok(dec)
    }

    pub fn zero_state(&self) -> LstmState {
        LstmState {
            h: vec![vec![0f32; self.hidden]; self.lstm.len()],
            c: vec![vec![0f32; self.hidden]; self.lstm.len()],
        }
    }

    /// `encoder_projector(enc)`: `[t, d]` → `[t, h]`.
    pub fn project_encoder(&self, enc: &[f32], t: usize) -> Vec<f32> {
        let mut out = vec![0f32; t * self.hidden];
        linear(
            enc,
            t,
            self.d_model,
            &self.enc_w,
            Some(&self.enc_b),
            self.hidden,
            &mut out,
        );
        out
    }

    /// Prediction network step: embed `token`, run the LSTM stack from `state`, project. Returns `(g [h], new_state)`.
    pub fn pred_step(&self, token: usize, state: &LstmState) -> (Vec<f32>, LstmState) {
        let h = self.hidden;
        let mut x = self.embedding[token * h..(token + 1) * h].to_vec();
        let mut new = state.clone();
        for (l, layer) in self.lstm.iter().enumerate() {
            let (hp, cp) = (&state.h[l], &state.c[l]);
            let mut gates = vec![0f32; 4 * h];
            let mut gh = vec![0f32; 4 * h];
            match (&layer.w_ih_i8, &layer.w_hh_i8) {
                (Some(wi), Some(wh)) => {
                    let qx = QuantAct::quantize_layout(&x, 1, h, wi.group, wi.layout);
                    crate::kernels::gemm_single(wi, &qx, &mut gates, Some(&layer.b));
                    let qh = QuantAct::quantize_layout(hp, 1, h, wh.group, wh.layout);
                    crate::kernels::gemm_single(wh, &qh, &mut gh, None);
                }
                _ => {
                    gemv_f16(&x, h, &layer.w_ih, Some(&layer.b), 4 * h, &mut gates);
                    gemv_f16(hp, h, &layer.w_hh, None, 4 * h, &mut gh);
                }
            }
            for (g, v) in gates.iter_mut().zip(&gh) {
                *g += v;
            }
            let (hn, cn) = (&mut new.h[l], &mut new.c[l]);
            for k in 0..h {
                let i = sigmoid(gates[k]);
                let f = sigmoid(gates[h + k]);
                let g = gates[2 * h + k].tanh();
                let o = sigmoid(gates[3 * h + k]);
                cn[k] = f * cp[k] + i * g;
                hn[k] = o * cn[k].tanh();
            }
            x = hn.clone();
        }
        let mut g = vec![0f32; h];
        gemv_f16(&x, h, &self.proj_w, Some(&self.proj_b), h, &mut g);
        (g, new)
    }

    /// Joint: `head(relu(enc_t + g))` → logits `[vocab + n_dur]`.
    pub fn joint(&self, enc_t: &[f32], g: &[f32]) -> Vec<f32> {
        let h = self.hidden;
        let z: Vec<f32> = enc_t.iter().zip(g).map(|(a, b)| (a + b).max(0.0)).collect();
        let n = self.head_b.len();
        let mut out = vec![0f32; n];
        if let Some(wq) = &self.head_i8 {
            // z quantised per 128-group; single-threaded kernel (one dispatch per step would cost more than it saves)
            let q = QuantAct::quantize_layout(&z, 1, h, wq.group, wq.layout);
            crate::kernels::gemm_single(wq, &q, &mut out, Some(&self.head_b));
        } else {
            gemv_f16(&z, h, &self.head_w, Some(&self.head_b), n, &mut out);
        }
        out
    }

    fn argmax(v: &[f32]) -> usize {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    }

    /// Greedy TDT over projected encoder frames `proj [t, h]` (NeMo semantics, `docs/tdt.md`).
    pub fn greedy(
        &self,
        proj: &[f32],
        t: usize,
        mut trace: Option<&mut Vec<Step>>,
    ) -> Vec<Emission> {
        let h = self.hidden;
        let mut out = Vec::new();
        let mut state = self.zero_state();
        let mut last = self.blank;
        let mut tf = 0usize;
        // the prediction network depends only on (last, state): after a blank both are unchanged, so its output
        // is reused instead of recomputed (identical arithmetic; ~1/4 of the greedy steps follow a blank)
        let mut pred: Option<(Vec<f32>, LstmState)> = None;
        while tf < t {
            let mut symbols = 0;
            let mut dur;
            loop {
                if pred.is_none() {
                    pred = Some(self.pred_step(last, &state));
                }
                let (g, new_state) = pred.as_ref().unwrap();
                let logits = self.joint(&proj[tf * h..(tf + 1) * h], g);
                let k = Self::argmax(&logits[..self.vocab]);
                let di = Self::argmax(&logits[self.vocab..]);
                dur = self.durations[di];
                if let Some(tr) = trace.as_deref_mut() {
                    tr.push(Step {
                        t: tf,
                        token: k,
                        duration: dur,
                        logits: logits.clone(),
                        h: new_state.h.concat(),
                        c: new_state.c.concat(),
                    });
                }
                if k != self.blank {
                    out.push(Emission {
                        token: k,
                        frame: tf,
                        duration: dur,
                    });
                    let (_, new_state) = pred.take().unwrap();
                    state = new_state;
                    last = k;
                }
                symbols += 1;
                tf += dur;
                if dur != 0 || symbols >= self.max_symbols {
                    break;
                }
                if k == self.blank {
                    // blank with duration 0: state and logits would not change → NeMo spins to the cap, then +1
                    break;
                }
            }
            if dur == 0 {
                tf += 1;
            }
        }
        out
    }
}

/// Per-row symmetric int8 quantisation of an f16 `[out, inp]` matrix into the kernel's layout. The row scale is
/// replicated per 128-group so the ternary GEMM kernels (which expect `[out, inp/128]` scales) can run it unchanged.
fn int8_rows(w_f16: &[u16], out: usize, inp: usize) -> TernaryMatrix {
    use half::f16;
    let group = 128;
    let ng = inp / group;
    // row layout: single-token GEMVs run on the `sdot` GEMV kernel (3c.16), which wants contiguous rows
    let layout = crate::packed::Layout::Rows;
    let mut rows = vec![0i8; out * inp];
    let mut scales = vec![0f32; out * ng];
    let mut row = vec![0f32; inp];
    let mut qrow = vec![0i8; inp];
    for r in 0..out {
        for k in 0..inp {
            row[k] = f16::from_bits(w_f16[r * inp + k]).to_f32();
        }
        let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        // one scale per row: quantise the whole row with it (quantize_group would pick per-group scales)
        let inv = 1.0 / s;
        for k in 0..inp {
            qrow[k] = (row[k] * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
        let _ = quantize_group;
        for g in 0..ng {
            scales[r * ng + g] = s;
        }
        rows[r * inp..(r + 1) * inp].copy_from_slice(&qrow);
    }
    TernaryMatrix::from_i8_rows(&rows, &scales, out, inp, group, layout)
}
