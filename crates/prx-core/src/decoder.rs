//! TDT prediction network, joint and greedy loop (PLAN 2.10). Semantics: `docs/tdt.md` (NeMo rule).
use crate::format::ModelFile;
use crate::tensor::{dot, linear, load_f32_shaped, sigmoid};
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
    w_ih: Vec<f32>, // [4h, in]
    w_hh: Vec<f32>, // [4h, h]
    b: Vec<f32>,    // b_ih + b_hh, [4h]
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
    proj_w: Vec<f32>, // decoder_projector [h, h]
    proj_b: Vec<f32>,
    enc_w: Vec<f32>, // encoder_projector [h, d]
    enc_b: Vec<f32>,
    head_w: Vec<f32>, // [vocab + n_dur, h]
    head_b: Vec<f32>,
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
            let w_ih = load_f32_shaped(m, &format!("decoder.lstm.weight_ih_l{l}"), &[4 * h, h])?;
            let w_hh = load_f32_shaped(m, &format!("decoder.lstm.weight_hh_l{l}"), &[4 * h, h])?;
            let b_ih = load_f32_shaped(m, &format!("decoder.lstm.bias_ih_l{l}"), &[4 * h])?;
            let b_hh = load_f32_shaped(m, &format!("decoder.lstm.bias_hh_l{l}"), &[4 * h])?;
            let b = b_ih.iter().zip(&b_hh).map(|(a, b)| a + b).collect();
            lstm.push(LstmLayer { w_ih, w_hh, b });
        }
        let n_out = v + c.durations.len();
        Ok(Self {
            blank: c.blank_token_id,
            vocab: v,
            hidden: h,
            durations: c.durations.clone(),
            max_symbols: c.max_symbols_per_step,
            embedding,
            lstm,
            proj_w: load_f32_shaped(m, "decoder.decoder_projector.weight", &[h, h])?,
            proj_b: load_f32_shaped(m, "decoder.decoder_projector.bias", &[h])?,
            enc_w: load_f32_shaped(m, "encoder_projector.weight", &[h, d])?,
            enc_b: load_f32_shaped(m, "encoder_projector.bias", &[h])?,
            head_w: load_f32_shaped(m, "joint.head.weight", &[n_out, h])?,
            head_b: load_f32_shaped(m, "joint.head.bias", &[n_out])?,
            d_model: d,
        })
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
            let gates: Vec<f32> = (0..4 * h)
                .map(|g| {
                    layer.b[g]
                        + dot(&layer.w_ih[g * h..(g + 1) * h], &x)
                        + dot(&layer.w_hh[g * h..(g + 1) * h], hp)
                })
                .collect();
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
        linear(&x, 1, h, &self.proj_w, Some(&self.proj_b), h, &mut g);
        (g, new)
    }

    /// Joint: `head(relu(enc_t + g))` → logits `[vocab + n_dur]`.
    pub fn joint(&self, enc_t: &[f32], g: &[f32]) -> Vec<f32> {
        let h = self.hidden;
        let z: Vec<f32> = enc_t.iter().zip(g).map(|(a, b)| (a + b).max(0.0)).collect();
        let n = self.head_b.len();
        let mut out = vec![0f32; n];
        linear(&z, 1, h, &self.head_w, Some(&self.head_b), n, &mut out);
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
        while tf < t {
            let mut symbols = 0;
            let mut dur;
            loop {
                let (g, new_state) = self.pred_step(last, &state);
                let logits = self.joint(&proj[tf * h..(tf + 1) * h], &g);
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
