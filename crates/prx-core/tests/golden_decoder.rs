mod common;
use common::*;
use prx_core::decoder::Decoder;
use prx_core::format::{default_model_dir, ModelFile};

#[test]
fn decoder_matches_golden_from_golden_encoder_output() {
    if !have_golden() || !default_model_dir().join("model.safetensors").exists() {
        return;
    }
    std::env::set_var("PRX_DECODER", "f16"); // exactness test: the int8 decoder is validated on the token oracle instead
    let m = ModelFile::open(default_model_dir()).unwrap();
    let dec = Decoder::load(&m).unwrap();
    for id in golden_ids() {
        let d = golden_dir().join(&id);
        let enc = read_npy(d.join("encoder_out.npy"));
        let te = enc.shape[0];
        let proj = dec.project_encoder(&enc.data, te);
        let gp = read_npy(d.join("encoder_proj.npy"));
        let dp = max_abs_diff(&proj, &gp.data);
        assert!(dp < 1e-3, "{id}: encoder_proj diff {dp}");
        let mut steps = Vec::new();
        let em = dec.greedy(&proj, te, Some(&mut steps));
        let g_t = read_npy(d.join("step_t.npy")).data;
        let g_tok = read_npy(d.join("step_token.npy")).data;
        let g_dur = read_npy(d.join("step_dur.npy")).data;
        let g_logits = read_npy(d.join("step_logits.npy"));
        let g_h = read_npy(d.join("step_h.npy"));
        let g_em = read_npy(d.join("emissions.npy"));
        assert_eq!(steps.len(), g_t.len(), "{id}: step count");
        let mut worst_logit = 0f32;
        let mut worst_h = 0f32;
        for (i, s) in steps.iter().enumerate() {
            assert_eq!(
                (s.t, s.token, s.duration),
                (g_t[i] as usize, g_tok[i] as usize, g_dur[i] as usize),
                "{id}: step {i}"
            );
            worst_logit = worst_logit.max(max_abs_diff(
                &s.logits,
                &g_logits.data[i * 8198..(i + 1) * 8198],
            ));
            worst_h = worst_h.max(max_abs_diff(&s.h, &g_h.data[i * 1280..(i + 1) * 1280]));
        }
        let ours: Vec<f32> = em
            .iter()
            .flat_map(|e| [e.token as f32, e.frame as f32, e.duration as f32])
            .collect();
        assert_eq!(ours, g_em.data, "{id}: emissions");
        eprintln!("{id}: steps={} emissions={} exact; logits max|Δ|={worst_logit:.2e} lstm-h max|Δ|={worst_h:.2e}", steps.len(), em.len());
        assert!(worst_h < 1e-4);
    }
}
