//! Kernel exactness: SIMD == scalar on random tiles; quantised path ≈ f32 path (PLAN 3.4 / 3.7).
use prx_core::kernels::{detect, gemm, gemm_ep, gemm_rows_range, gemm_scalar, Epilogue, Kernel};
use prx_core::packed::{Layout, QuantAct, TernaryMatrix};
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
            gemm_scalar(&w, &qa, &mut y_ref, 0, out, Epilogue::Store);
            let mut y = vec![0f32; n * out];
            gemm(&w, &qa, &mut y, None);
            assert_eq!(y, y_ref, "out={out} inp={inp} n={n} kernel={:?}", detect());
        }
    }
}

#[test]
fn i8mm_equals_scalar_exactly() {
    if detect() != Kernel::NeonI8mm {
        eprintln!("no i8mm on this machine; skipped");
        return;
    }
    let mut seed = 99u64;
    for &(out, inp) in &[(8usize, 128usize), (12, 256), (64, 1024), (132, 4096)] {
        let (q, s16, _) = random_matrix(out, inp, &mut seed);
        let w_rows = TernaryMatrix::from_packed_layout(&q, &s16, out, inp, 128, Layout::Rows);
        let w_pairs = TernaryMatrix::from_packed_layout(&q, &s16, out, inp, 128, Layout::Pairs);
        for r in 0..out {
            for k in 0..inp {
                assert_eq!(w_rows.get(r, k), w_pairs.get(r, k));
            }
        }
        for &n in &[1usize, 2, 3, 4, 5, 6, 7, 17] {
            let x: Vec<f32> = (0..n * inp)
                .map(|_| (lcg(&mut seed) % 2001) as f32 / 100.0 - 10.0)
                .collect();
            let qa = QuantAct::quantize(&x, n, inp, 128);
            let qp = QuantAct::quantize_layout(&x, n, inp, 128, Layout::Pairs);
            for tk in 0..n {
                for k in 0..inp {
                    assert_eq!(qa.get(tk, k), qp.get(tk, k));
                }
            }
            let mut y_ref = vec![0f32; n * out];
            gemm_scalar(&w_rows, &qa, &mut y_ref, 0, out, Epilogue::Store);
            let mut y = vec![0f32; n * out];
            gemm(&w_pairs, &qp, &mut y, None);
            assert_eq!(y, y_ref, "i8mm out={out} inp={inp} n={n}");
        }
    }
}

#[test]
fn epilogues_are_identical_across_kernels() {
    let mut seed = 5u64;
    let (out, inp, n) = (64usize, 1024usize, 13usize);
    let (q, s16, _) = random_matrix(out, inp, &mut seed);
    let layout = detect().layout();
    let w_rows = TernaryMatrix::from_packed_layout(&q, &s16, out, inp, 128, Layout::Rows);
    let w = TernaryMatrix::from_packed_layout(&q, &s16, out, inp, 128, layout);
    let x: Vec<f32> = (0..n * inp)
        .map(|_| (lcg(&mut seed) % 2001) as f32 / 100.0 - 10.0)
        .collect();
    let qa = QuantAct::quantize(&x, n, inp, 128);
    let qk = QuantAct::quantize_layout(&x, n, inp, 128, layout);
    let base: Vec<f32> = (0..n * out).map(|i| (i % 7) as f32 - 3.0).collect();
    for ep in [Epilogue::Store, Epilogue::Silu, Epilogue::AddScaled(0.5)] {
        let mut y_ref = base.clone();
        gemm_scalar(&w_rows, &qa, &mut y_ref, 0, out, ep);
        let mut y = base.clone();
        gemm_ep(&w, &qk, &mut y, None, ep);
        assert_eq!(y, y_ref, "{ep:?} on {:?}", detect());
    }
}

#[test]
fn packed2_equals_scalar_exactly() {
    if detect() != Kernel::NeonI8mm {
        eprintln!("no i8mm on this machine; skipped");
        return;
    }
    let mut seed = 1234u64;
    for &(out, inp) in &[(8usize, 128usize), (16, 256), (64, 1024), (136, 4096)] {
        let (q, s16, _) = random_matrix(out, inp, &mut seed);
        let w_rows = TernaryMatrix::from_packed_layout(&q, &s16, out, inp, 128, Layout::Rows);
        let w2 = TernaryMatrix::from_packed_layout(&q, &s16, out, inp, 128, Layout::Packed2);
        assert_eq!(w2.w.len(), out * inp / 4);
        for r in 0..out {
            for k in 0..inp {
                assert_eq!(w_rows.get(r, k), w2.get(r, k), "r={r} k={k}");
            }
        }
        for &n in &[1usize, 3, 8, 13, 40] {
            let x: Vec<f32> = (0..n * inp)
                .map(|_| (lcg(&mut seed) % 2001) as f32 / 100.0 - 10.0)
                .collect();
            let qa = QuantAct::quantize(&x, n, inp, 128);
            let qp = QuantAct::quantize_layout(&x, n, inp, 128, Layout::Pairs);
            for ep in [Epilogue::Store, Epilogue::AddScaled(0.5)] {
                let base: Vec<f32> = (0..n * out).map(|i| (i % 5) as f32).collect();
                let mut y_ref = base.clone();
                gemm_scalar(&w_rows, &qa, &mut y_ref, 0, out, ep);
                let mut y = base.clone();
                gemm_ep(&w2, &qp, &mut y, None, ep);
                assert_eq!(y, y_ref, "packed2 out={out} inp={inp} n={n} {ep:?}");
            }
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

/// The token-range entry used by the attention core (3c.8): any (row range, token range) of an int8 (non-ternary)
/// matrix × int8 activations equals the scalar reference bit for bit, on every kernel.
#[test]
fn row_token_ranges_equal_scalar_exactly() {
    use prx_core::kernels::kernel;
    use prx_core::packed::Layout;
    let mut seed = 7u64;
    let layout = kernel().int8_layout();
    for &(out, inp, n) in &[
        (14usize, 128usize, 11usize),
        (34, 128, 19),
        (8, 256, 8),
        (6, 128, 3),
    ] {
        let wf: Vec<f32> = (0..out * inp)
            .map(|_| (lcg(&mut seed) % 2001) as f32 / 100.0 - 10.0)
            .collect();
        let w = TernaryMatrix::from_f32_rows(out, inp, 128, layout, |r, row| {
            row.copy_from_slice(&wf[r * inp..(r + 1) * inp])
        });
        let w_rows = TernaryMatrix::from_f32_rows(out, inp, 128, Layout::Rows, |r, row| {
            row.copy_from_slice(&wf[r * inp..(r + 1) * inp])
        });
        let x: Vec<f32> = (0..n * inp)
            .map(|_| (lcg(&mut seed) % 2001) as f32 / 100.0 - 10.0)
            .collect();
        let qa = QuantAct::quantize_layout(&x, n, inp, 128, layout);
        let qa_rows = QuantAct::quantize(&x, n, inp, 128);
        let mut y_ref = vec![0f32; n * out];
        gemm_scalar(&w_rows, &qa_rows, &mut y_ref, 0, out, Epilogue::Store);
        for &(r0, r1) in &[
            (0usize, out),
            (2, out),
            (0, (out / 2) & !1),
            (4, 6.min(out)),
        ] {
            for &(t0, t1) in &[
                (0usize, n),
                (0, n.min(8)),
                (8.min(n) & !1, n),
                (2.min(n), n.min(5)),
            ] {
                if r0 >= r1 || t0 >= t1 {
                    continue;
                }
                let mut y = vec![f32::NAN; n * out];
                gemm_rows_range(&w, &qa, &mut y, r0, r1, t0, t1, Epilogue::Store);
                for t in t0..t1 {
                    for o in r0..r1 {
                        assert_eq!(
                            y[t * out + o],
                            y_ref[t * out + o],
                            "out={out} inp={inp} n={n} rows {r0}..{r1} toks {t0}..{t1} at ({t},{o}) kernel={:?}",
                            kernel()
                        );
                    }
                }
                // nothing outside the requested block was touched
                for t in 0..n {
                    for o in 0..out {
                        let inside = (t0..t1).contains(&t) && (r0..r1).contains(&o);
                        assert_eq!(y[t * out + o].is_nan(), !inside, "stray write at ({t},{o})");
                    }
                }
            }
        }
    }
}

/// Single-token GEMV (decoder shapes, row layout) is bit-identical to the scalar reference.
#[test]
fn gemv_equals_scalar_exactly() {
    use prx_core::kernels::gemm_single;
    use prx_core::packed::Layout;
    let mut seed = 99u64;
    for &(out, inp) in &[
        (8194usize, 640usize),
        (2560, 640),
        (13, 128),
        (8, 256),
        (3, 128),
    ] {
        let rows: Vec<i8> = (0..out * inp)
            .map(|_| (lcg(&mut seed) % 255) as i8)
            .collect();
        let scales: Vec<f32> = (0..out * inp / 128)
            .map(|_| (lcg(&mut seed) % 100) as f32 / 1000.0 + 0.001)
            .collect();
        let w = TernaryMatrix::from_i8_rows(&rows, &scales, out, inp, 128, Layout::Rows);
        let x: Vec<f32> = (0..inp)
            .map(|_| (lcg(&mut seed) % 2001) as f32 / 100.0 - 10.0)
            .collect();
        let bias: Vec<f32> = (0..out)
            .map(|_| (lcg(&mut seed) % 100) as f32 / 10.0)
            .collect();
        let qa = QuantAct::quantize(&x, 1, inp, 128);
        let mut y_ref = vec![0f32; out];
        gemm_scalar(&w, &qa, &mut y_ref, 0, out, Epilogue::Store);
        for (v, b) in y_ref.iter_mut().zip(&bias) {
            *v += b;
        }
        let mut y = vec![0f32; out];
        gemm_single(&w, &qa, &mut y, Some(&bias));
        assert_eq!(y, y_ref, "out={out} inp={inp} kernel={:?}", detect());
    }
}
