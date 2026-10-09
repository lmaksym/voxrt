//! Ternary-GEMM kernel variants on the GPU (development): `cargo test --release -p voxrt --features metal --test
//! metal_bench -- --ignored --nocapture`.
#![cfg(all(feature = "metal", target_vendor = "apple"))]

#[test]
#[ignore]
fn tgemm_variants() {
    // (kernel, tile m, tile n, K splits)
    // (kernel, tile m, tile n, threads, K splits)
    let variants: &[(&str, usize, usize, usize, usize)] = &[
        ("tgemm_16", 16, 64, 128, 1),
        ("tgemm_16", 16, 64, 128, 4),
        ("tgemm_32", 32, 64, 128, 1),
        ("tgemm_32", 32, 64, 128, 4),
    ];
    for &(m, n, k) in &[
        (34, 4096, 1024),
        (34, 1024, 4096),
        (72, 4096, 1024),
        (72, 1024, 4096),
        (200, 4096, 1024),
        (200, 1024, 4096),
        (333, 4096, 1024),
        (333, 1024, 4096),
        (746, 4096, 1024),
        (746, 1024, 4096),
    ] {
        println!("M={m:4} N={n:4} K={k:4}");
        for &(name, bm, bn, threads, splits) in variants {
            if k / splits < 128 {
                continue;
            }
            let s = voxrt::metal::bench_tgemm(name, bm, bn, threads, m, n, k, splits, 40).unwrap();
            println!(
                "   {name:20} split {splits:2}: {:7.1} µs {:5.2} TFLOP/s",
                s * 1e6,
                2.0 * (m * n * k) as f64 / s / 1e12
            );
        }
    }
}
