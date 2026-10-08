//! Integration tests that need the real weights (skipped when the model dir is absent).
use voxrt::format::{default_model_dir, ModelFile};

fn model() -> Option<ModelFile> {
    let dir = default_model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: no model at {}", dir.display());
        return None;
    }
    Some(ModelFile::open(dir).expect("open"))
}

#[test]
fn validates_cleanly_and_matches_inventory() {
    let Some(m) = model() else { return };
    let problems = m.validate();
    assert!(problems.is_empty(), "{problems:#?}");
    let infos = m.tensor_infos();
    assert_eq!(infos.len(), 993);
    assert_eq!(m.ternary.quantized_modules.len(), 264);
    assert_eq!(m.config.vocab_size, 8193);
    assert_eq!(m.config.blank_token_id, 8192);
}

#[test]
fn zero_fractions_match_ternary_json() {
    let Some(m) = model() else { return };
    let mut worst = 0f64;
    for q in &m.ternary.quantized_modules {
        let (info, bytes) = m.tensor(&format!("{}.qweight", q.name)).unwrap();
        let rb = q.in_features.div_ceil(5);
        assert_eq!(info.shape, vec![q.out_features, rb]);
        let mut row = vec![0i8; q.in_features];
        let mut zeros = 0usize;
        for r in 0..q.out_features {
            voxrt::trits::unpack_row(&bytes[r * rb..(r + 1) * rb], q.in_features, &mut row);
            zeros += row.iter().filter(|&&t| t == 0).count();
        }
        let zf = zeros as f64 / (q.out_features * q.in_features) as f64;
        worst = worst.max((zf - q.zero_fraction).abs());
    }
    assert!(worst < 1e-6, "worst zero-fraction mismatch {worst}");
}
