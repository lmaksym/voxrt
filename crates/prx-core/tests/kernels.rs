//! Kernel exactness: SIMD == scalar on random tiles; quantised path ≈ f32 path (PLAN 3.4 / 3.7).
use prx_core::kernels::{detect, gemm, gemm_scalar, Kernel};
use prx_core::packed::{QuantAct, TernaryMatrix};
use prx_core::tensor::linear;
use prx_core::trits::pack_row;

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}

fn random_matrix(out: usize, inp: usize, seed: &mut u64) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    // returns packed qweight bytes, f16 scale bytes, and the dequantised f32 matrix
    let rb = inp.div_ceil(5);
    let ng = inp / 128;
    let mut q = vec![0u8; out * rb];
    let mut s16 = vec![0u8; out * ng * 2];
    let mut wf = vec![0f32; out * inp];
    for r in 0..out {
        let trits: Vec<i8> = (0..inp).map(|_| (lcg(seed) % 3) as i8 - 1).collect();
        pack_row(&trits, &mut q[r * rb..(r + 1) * rb]);
        for g in 0..ng {
            let s = half::f16::from_f32(0.01 + (lcg(seed) % 100) as f32 * 0.001);
            s16[(r * ng + g) * 2..(r * ng + g) * 2 + 2].copy_from_slice(&s.to_le_bytes());
            for c in g * 128..(g + 1) * 128 {
                wf[r * inp + c] = s.to_f32() * trits[c] as f32;
            }
        }
    }
    (q, s16, wf)
}

#[test]
fn simd_equals_scalar_exactly() {
    let mut seed = 42u64;
    for &(out, inp) in &[(7usize, 128usize), (9, 256), (64, 1024), (130, 4096)] {
        let (q, s16, _) = random_matrix(out, inp, &mut seed);
        let w = TernaryMatrix::from_packed(&q, &s16, out, inp, 128);
        for &n in &[1usize, 3, 4, 5, 17] {
            let x: Vec<f32> = (0..n * inp)
                .map(|_| (lcg(&mut seed) % 2001) as f32 / 100.0 - 10.0)
                .collect();
            let qa = QuantAct::quantize(&x, n, inp, 128);
            let mut y_ref = vec![0f32; n * out];
            gemm_scalar(&w, &qa, &mut y_ref, 0, out);
            let mut y = vec![0f32; n * out];
            gemm(&w, &qa, &mut y, None);
            assert_eq!(y, y_ref, "out={out} inp={inp} n={n} kernel={:?}", detect());
        }
    }
}

#[test]
fn quantised_path_tracks_f32_path() {
    let mut seed = 7u64;
    let (out, inp, n) = (256usize, 1024usize, 16usize);
    let (q, s16, wf) = random_matrix(out, inp, &mut seed);
    let w = TernaryMatrix::from_packed(&q, &s16, out, inp, 128);
    let x: Vec<f32> = (0..n * inp)
        .map(|_| (lcg(&mut seed) % 2001) as f32 / 1000.0 - 1.0)
        .collect();
    let mut yf = vec![0f32; n * out];
    linear(&x, n, inp, &wf, None, out, &mut yf);
    let qa = QuantAct::quantize(&x, n, inp, 128);
    let mut yq = vec![0f32; n * out];
    gemm(&w, &qa, &mut yq, None);
    let scale = yf.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = yf
        .iter()
        .zip(&yq)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "f32 vs int8-per-group-128: max abs err {err:.3e} on values up to {scale:.2} (rel {:.2e})",
        err / scale
    );
    assert!(err / scale < 1e-2);
}

#[test]
fn accumulation_cannot_overflow() {
    // worst case: every trit ±1 aligned with x = ±127 → |group sum| = 128 · 127 = 16256 (fits i16, trivially i32)
    let inp = 4096;
    let trits: Vec<i8> = vec![1; inp];
    let mut q = vec![0u8; inp.div_ceil(5)];
    pack_row(&trits, &mut q);
    let s16: Vec<u8> = (0..inp / 128)
        .flat_map(|_| half::f16::from_f32(1.0).to_le_bytes())
        .collect();
    let w = TernaryMatrix::from_packed(&q, &s16, 1, inp, 128);
    let x = vec![127.0f32; inp];
    let qa = QuantAct::quantize(&x, 1, inp, 128);
    let mut y = vec![0f32; 1];
    gemm(&w, &qa, &mut y, None);
    assert_eq!(y[0], (inp as f32) * 127.0);
    assert_ne!(
        detect(),
        Kernel::Scalar,
        "expected a SIMD kernel on this machine"
    );
}
