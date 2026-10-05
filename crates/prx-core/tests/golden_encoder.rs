mod common;
use common::*;
use prx_core::encoder::{rel_pos_table, Encoder, LayerTrace};
use prx_core::format::{default_model_dir, ModelFile};
use prx_core::frontend::{Frontend, MelConfig};

fn rel(a: &[f32], b: &[f32]) -> (f32, f32) {
    let d = max_abs_diff(a, b);
    let s = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    (d, d / s.max(1e-12))
}

#[test]
fn pos_table_matches_golden() {
    if !have_golden() {
        return;
    }
    for id in golden_ids().into_iter().take(2) {
        let g = read_npy(golden_dir().join(&id).join("pos_emb.npy"));
        let te = g.shape[0].div_ceil(2);
        let ours = rel_pos_table(te, 1024);
        let (d, _) = rel(&ours, &g.data);
        assert!(d < 1e-4, "{id}: pos table diff {d}");
    }
}

#[test]
fn encoder_matches_golden() {
    if !have_golden() || !default_model_dir().join("model.safetensors").exists() {
        return;
    }
    let m = ModelFile::open(default_model_dir()).unwrap();
    let t0 = std::time::Instant::now();
    let enc = Encoder::load(&m).unwrap();
    eprintln!(
        "encoder loaded (dequantised) in {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    let fe = Frontend::new(MelConfig::default());
    let ids = golden_ids();
    // sublayer + per-layer check on the shortest and the medium utterance, final-output check on all
    for (n, id) in ids.iter().enumerate() {
        let d = golden_dir().join(id);
        let wav = read_npy(d.join("waveform.npy")).data;
        let (feats, t) = fe.compute(&wav);
        let mut outs = Vec::new();
        let mut tr = LayerTrace::default();
        let t1 = std::time::Instant::now();
        let (out, te) = enc.forward(&feats, t, Some(&mut outs), Some(&mut tr));
        let secs = t1.elapsed().as_secs_f32();
        if n < 2 {
            for (name, ours) in [
                ("layer00_norm_ff1_out", &tr.norm_ff1),
                ("layer00_ff1_out", &tr.ff1),
                ("layer00_norm_attn_out", &tr.norm_attn),
                ("layer00_attn_out", &tr.attn),
                ("layer00_norm_conv_out", &tr.norm_conv),
                ("layer00_conv_out", &tr.conv),
                ("layer00_ff2_out", &tr.ff2),
            ] {
                let g = read_npy(d.join(format!("{name}.npy")));
                let (da, dr) = rel(ours, &g.data[..te * 1024]);
                eprintln!("{id}: {name:24} max|Δ|={da:.2e} rel={dr:.2e}");
                assert!(dr < 1e-3, "{id}: {name} relative diff {dr}");
            }
            for (i, o) in outs.iter().enumerate() {
                let g = read_npy(d.join(format!("layer{i:02}_out.npy")));
                let (da, dr) = rel(o, &g.data[..te * 1024]);
                eprintln!("{id}: layer{i:02} max|Δ|={da:.2e} rel={dr:.2e}");
            }
        }
        let g = read_npy(d.join("encoder_out.npy")); // sliced to the valid length by dump_golden
        assert_eq!(te, g.shape[0]);
        let (da, dr) = rel(&out, &g.data);
        eprintln!("{id}: T_e={te} encoder_out max|Δ|={da:.2e} rel={dr:.2e}  ({secs:.2}s, {:.1}× real time)", wav.len() as f32 / 16000.0 / secs);
        assert!(da < 1e-2, "{id}: encoder_out abs diff {da}");
    }
}
