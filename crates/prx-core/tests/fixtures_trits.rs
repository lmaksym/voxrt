//! Unit tests against fixtures produced by `reference-py/make_fixtures.py` (PLAN 2.3).
use std::path::PathBuf;

fn fx(name: &str) -> Vec<u8> {
    std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn trits_and_dequant_match_python_reference() {
    let meta: serde_json::Value = serde_json::from_slice(&fx("ternary_fixtures.json")).unwrap();
    for (tag, m) in meta.as_object().unwrap() {
        let rows = m["rows"].as_u64().unwrap() as usize;
        let inf = m["in_features"].as_u64().unwrap() as usize;
        let g = m["group_size"].as_u64().unwrap() as usize;
        let q = fx(&format!("{tag}.qweight.bin"));
        let s = fx(&format!("{tag}.scales.f16.bin"));
        let want_t: Vec<i8> = fx(&format!("{tag}.trits.i8.bin"))
            .into_iter()
            .map(|b| b as i8)
            .collect();
        let want_w: Vec<f32> = fx(&format!("{tag}.weight.f32.bin"))
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let rb = inf.div_ceil(5);
        let mut got_t = vec![0i8; rows * inf];
        for r in 0..rows {
            prx_core::trits::unpack_row(
                &q[r * rb..(r + 1) * rb],
                inf,
                &mut got_t[r * inf..(r + 1) * inf],
            );
        }
        assert_eq!(got_t, want_t, "{tag} trits");
        let got_w = prx_core::trits::dequant_matrix(&q, &s, rows, inf, g);
        assert_eq!(got_w, want_w, "{tag} dequant (exact: f16 scale × trit)");
        let zf = got_t.iter().filter(|&&t| t == 0).count() as f64 / got_t.len() as f64;
        assert!(
            (zf - m["zero_fraction_rows"].as_f64().unwrap()).abs() < 1e-9,
            "{tag} zero fraction"
        );
    }
}
