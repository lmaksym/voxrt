mod common;
use common::*;
use prx_core::format::{default_model_dir, ModelFile};
use prx_core::frontend::{Frontend, MelConfig};
use prx_core::subsampling::Subsampling;
use prx_core::vad::{segment, SegmentConfig, VadHead};

#[test]
fn vad_probs_match_python_probe() {
    if !have_golden() || !default_model_dir().join("model.safetensors").exists() {
        return;
    }
    let m = ModelFile::open(default_model_dir()).unwrap();
    let sub = Subsampling::load(&m).unwrap();
    let vad = VadHead::load(&m).unwrap();
    let fe = Frontend::new(MelConfig::default());
    for id in golden_ids() {
        let gp = golden_dir().join("vad").join(format!("{id}.npy"));
        if !gp.exists() {
            continue;
        }
        let g = read_npy(gp);
        let wav = read_npy(golden_dir().join(&id).join("waveform.npy")).data;
        let (feats, t) = fe.compute(&wav);
        let so = sub.forward(&feats, t);
        let te = sub.out_frames(t);
        let p = vad.probs(&so, te);
        // HF's extra masked frame leaks into the Python probe's last ctx-conv window (k=5): compare frames unaffected by it
        let n = if g.data.len() > te {
            te.saturating_sub(2)
        } else {
            te
        };
        let d = max_abs_diff(&p[..n], &g.data[..n]);
        let segs = segment(&p, &SegmentConfig::default());
        eprintln!(
            "{id}: T_e={te} golden frames={} vad max|Δ|={d:.2e} segments={:?}",
            g.data.len(),
            segs
        );
        assert!(d < 1e-4, "{id}: vad diff {d}");
    }
}

#[test]
fn segmentation_rules() {
    // rules under an explicit config (min 5 s, cap 30 s): 10 s speech, 1 s pause, 5 s speech → two segments;
    // then a 60 s speech run is split under the cap
    let cfg = SegmentConfig {
        max_frames: 375,
        min_frames: 63,
        ..SegmentConfig::default()
    };
    let mut p = vec![0.9f32; 125];
    p.extend(vec![0.1f32; 13]);
    p.extend(vec![0.9f32; 63]);
    let s = segment(&p, &cfg);
    assert_eq!(s.len(), 2, "{s:?}");
    assert!(
        s[0].start == 0 && s[0].end == 127 && s[1].start == 136 && s[1].end == 201,
        "{s:?}"
    );
    let long = vec![0.95f32; 750];
    let s = segment(&long, &cfg);
    assert!(s.iter().all(|x| x.end - x.start <= 375 + 2), "{s:?}");
    assert_eq!(s.first().unwrap().start, 0);
    assert_eq!(s.last().unwrap().end, 750);
    // defaults (min 10 s, cap 60 s, 2026-10-06): the 5 s run merges into the first segment; 90 s of speech is
    // split under the 60 s cap
    let d = SegmentConfig::default();
    assert!((d.max_frames, d.min_frames) == (750, 125));
    let s = segment(&p, &d);
    assert_eq!(s.len(), 1, "{s:?}");
    let long = vec![0.95f32; 1125];
    let s = segment(&long, &d);
    assert!(s.len() >= 2 && s.iter().all(|x| x.end - x.start <= 750 + 2), "{s:?}");
}
