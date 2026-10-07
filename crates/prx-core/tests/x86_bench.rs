//! Single-core throughput of the x86 kernels (run on the target box:
//! `cargo test --release -p prx-core --test x86_bench -- --ignored --nocapture`).
#![cfg(target_arch = "x86_64")]

use std::time::Instant;

use prx_core::kernels::{x86, Epilogue};
use prx_core::packed::{Layout, QuantAct, TernaryMatrix};
use prx_core::trits::pack_row;

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}

type RangeFn =
    unsafe fn(&TernaryMatrix, &QuantAct, &mut [f32], usize, usize, usize, usize, Epilogue);

fn best_of<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore]
fn x86_gemm_throughput() {
    let mut seed = 1u64;
    // encoder shapes: d = 1024 -> 4096 (FF up), 4096 -> 1024 (FF down); T = 376 frames
    for &(out, inp) in &[(4096usize, 1024usize), (1024, 4096)] {
        let n = 376;
        let rb = inp.div_ceil(5);
        let mut q = vec![0u8; out * rb];
        for r in 0..out {
            let trits: Vec<i8> = (0..inp).map(|_| (lcg(&mut seed) % 3) as i8 - 1).collect();
            pack_row(&trits, &mut q[r * rb..(r + 1) * rb]);
        }
        let s16: Vec<u8> = (0..out * inp / 128)
            .flat_map(|_| half::f16::from_f32(0.02).to_le_bytes())
            .collect();
        let w = TernaryMatrix::from_packed_layout(&q, &s16, out, inp, 128, Layout::Quad16);
        let xf: Vec<f32> = (0..n * inp)
            .map(|_| (lcg(&mut seed) % 2001) as f32 / 100.0 - 10.0)
            .collect();
        let x = QuantAct::quantize_layout(&xf, n, inp, 128, Layout::Quad16);
        let mut y = vec![0f32; n * out];
        let mut kernels: Vec<(&str, RangeFn)> = Vec::new();
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            kernels.push(("avx2", x86::gemm_avx2_range));
        }
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
            kernels.push(("vnni", x86::gemm_vnni_range));
        }
        for (name, f) in kernels {
            // a 256-row slice keeps the run short; the per-core rate is what matters
            let rows = 256;
            let s = best_of(5, || unsafe {
                f(&w, &x, &mut y, 0, rows, 0, n, Epilogue::Store)
            });
            let macs = (rows * inp * n) as f64;
            println!(
                "{name:5} {out}x{inp} n={n}: {:.1} GMAC/s per core",
                macs / s / 1e9
            );
        }
    }
    // decoder joint head GEMV: [8198, 640] int8 × one token
    {
        let (out, inp) = (8198usize, 640usize);
        let rows: Vec<i8> = (0..out * inp)
            .map(|_| ((lcg(&mut seed) % 255) as i32 - 127) as i8)
            .collect();
        let sc: Vec<f32> = (0..out * inp / 128).map(|_| 0.01).collect();
        let w = TernaryMatrix::from_i8_rows(&rows, &sc, out, inp, 128, Layout::Quad16);
        let xf: Vec<f32> = (0..inp)
            .map(|_| (lcg(&mut seed) % 2001) as f32 / 100.0 - 10.0)
            .collect();
        let x = QuantAct::quantize_layout(&xf, 1, inp, 128, Layout::Quad16);
        let mut y = vec![0f32; out];
        let mut kernels: Vec<(&str, RangeFn)> = Vec::new();
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            kernels.push(("avx2", x86::gemm_avx2_range));
        }
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
            kernels.push(("vnni", x86::gemm_vnni_range));
        }
        for (name, f) in kernels {
            let s = best_of(20, || unsafe {
                f(&w, &x, &mut y, 0, out, 0, 1, Epilogue::Store)
            });
            println!("{name:5} GEMV {out}x{inp}: {:.1} µs per call", s * 1e6);
        }
    }
    // f32 tile (subsampling / decoder projection): 4 tokens × 1024 outputs × 4096 inputs
    let (in_f, out_f) = (4096usize, 1024usize);
    let xf: Vec<f32> = (0..4 * in_f)
        .map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0)
        .collect();
    let wf: Vec<f32> = (0..out_f * in_f)
        .map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0)
        .collect();
    let mut y = vec![0f32; 4 * out_f];
    let s = best_of(5, || {
        prx_core::tensor::tile_f32(&xf, 4, in_f, &wf, None, out_f, &mut y)
    });
    println!(
        "f32 tile 4x{out_f}x{in_f}: {:.1} GFLOP/s per core",
        2.0 * (4 * out_f * in_f) as f64 / s / 1e9
    );
}
