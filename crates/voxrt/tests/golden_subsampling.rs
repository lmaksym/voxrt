mod common;
use common::*;
use voxrt::format::{default_model_dir, ModelFile};
use voxrt::frontend::{Frontend, MelConfig};
use voxrt::subsampling::Subsampling;

#[test]
fn subsampling_matches_golden() {
    if !have_golden() || !default_model_dir().join("model.safetensors").exists() {
        return;
    }
    let m = ModelFile::open(default_model_dir()).unwrap();
    let sub = Subsampling::load(&m).unwrap();
    let fe = Frontend::new(MelConfig::default());
    for id in golden_ids() {
        let d = golden_dir().join(&id);
        let wav = read_npy(d.join("waveform.npy")).data;
        let (feats, t) = fe.compute(&wav);
        let out = sub.forward(&feats, t);
        let g = read_npy(d.join("subsampling_out.npy"));
        let te = sub.out_frames(t);
        // HF runs the convs on floor(L/160)+1 frames (last one masked) and may emit one extra row: compare valid rows
        assert!(
            g.shape[0] == te || g.shape[0] == te + 1,
            "{id}: encoder frames {} vs ours {te}",
            g.shape[0]
        );
        let gd = &g.data[..te * 1024];
        let diff = max_abs_diff(&out, gd);
        let scale = gd.iter().fold(0f32, |a, &v| a.max(v.abs()));
        eprintln!("{id}: T_e={te} subsampling max|Δ|={diff:.2e} (golden absmax {scale:.2})");
        assert!(
            diff / scale < 1e-5,
            "{id}: subsampling relative diff {}",
            diff / scale
        );
    }
}
