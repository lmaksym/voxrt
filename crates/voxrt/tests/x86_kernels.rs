//! x86 kernels: AVX2 and AVX-512 VNNI on `Quad16` equal the scalar reference bit for bit, for ternary
//! and arbitrary int8 weights, partial row/token ranges, every epilogue, and saturation-prone extremes.
#![cfg(target_arch = "x86_64")]

use voxrt::kernels::{gemm_scalar_range, x86, Epilogue};
use voxrt::packed::{Layout, QuantAct, TernaryMatrix};
use voxrt::trits::pack_row;

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}

fn have_avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

fn have_vnni() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("fma")
}

fn ternary(out: usize, inp: usize, seed: &mut u64, extreme: bool) -> TernaryMatrix {
    let rb = inp.div_ceil(5);
    let ng = inp / 128;
    let mut q = vec![0u8; out * rb];
    let mut s16 = Vec::new();
    for r in 0..out {
        let trits: Vec<i8> = (0..inp)
            .map(|_| {
                if extreme {
                    1
                } else {
                    (lcg(seed) % 3) as i8 - 1
                }
            })
            .collect();
        pack_row(&trits, &mut q[r * rb..(r + 1) * rb]);
        for _ in 0..ng {
            s16.extend(half::f16::from_f32(0.01 + (lcg(seed) % 100) as f32 * 0.001).to_le_bytes());
        }
    }
    TernaryMatrix::from_packed_layout(&q, &s16, out, inp, 128, Layout::Quad16)
}

fn int8(out: usize, inp: usize, group: usize, seed: &mut u64, extreme: bool) -> TernaryMatrix {
    let rows: Vec<i8> = (0..out * inp)
        .map(|i| {
            if extreme {
                if i % 2 == 0 {
                    127
                } else {
                    -127
                }
            } else {
                ((lcg(seed) % 255) as i32 - 127) as i8
            }
        })
        .collect();
    let scales: Vec<f32> = (0..out * (inp / group))
        .map(|_| 0.001 + (lcg(seed) % 1000) as f32 * 1e-5)
        .collect();
    TernaryMatrix::from_i8_rows(&rows, &scales, out, inp, group, Layout::Quad16)
}

fn acts(n: usize, inp: usize, group: usize, seed: &mut u64, extreme: bool) -> QuantAct {
    let x: Vec<f32> = (0..n * inp)
        .map(|i| {
            if extreme {
                if i % 2 == 0 {
                    5.0
                } else {
                    -5.0
                }
            } else {
                (lcg(seed) % 2001) as f32 / 100.0 - 10.0
            }
        })
        .collect();
    QuantAct::quantize_layout(&x, n, inp, group, Layout::Quad16)
}

type RangeFn =
    unsafe fn(&TernaryMatrix, &QuantAct, &mut [f32], usize, usize, usize, usize, Epilogue);

fn check(name: &str, f: RangeFn, w: &TernaryMatrix, x: &QuantAct, seed: &mut u64) {
    let (out, n) = (w.out, x.n);
    let ranges = [
        (0, out, 0, n),
        (1.min(out), out, 0, n),
        (0, out.min(17), n / 2, n),
        (out / 3, out - out / 4, 0, n.div_ceil(2)),
    ];
    for ep in [Epilogue::Store, Epilogue::Silu, Epilogue::AddScaled(0.37)] {
        for &(r0, r1, t0, t1) in &ranges {
            if r0 >= r1 || t0 >= t1 {
                continue;
            }
            let init: Vec<f32> = (0..n * out)
                .map(|_| (lcg(seed) % 100) as f32 * 0.01)
                .collect();
            let mut want = init.clone();
            gemm_scalar_range(w, x, &mut want, r0, r1, t0, t1, ep);
            let mut got = init.clone();
            unsafe { f(w, x, &mut got, r0, r1, t0, t1, ep) };
            let bad = want
                .iter()
                .zip(&got)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert!(
                bad.is_none(),
                "{name}: out={out} inp={} group={} n={n} ternary={} rows {r0}..{r1} toks {t0}..{t1} {ep:?}: first diff at {:?}",
                w.inp,
                w.group,
                w.ternary,
                bad.map(|i| (i, want[i], got[i]))
            );
        }
    }
}

fn kernels() -> Vec<(&'static str, RangeFn)> {
    let mut k: Vec<(&'static str, RangeFn)> = Vec::new();
    if have_avx2() {
        k.push(("avx2", x86::gemm_avx2_range));
    }
    if have_vnni() {
        k.push(("vnni", x86::gemm_vnni_range));
    }
    if k.is_empty() {
        eprintln!("no AVX2/VNNI on this machine; skipped");
    }
    k
}

#[test]
fn ternary_equals_scalar_exactly() {
    let mut seed = 3u64;
    for (name, f) in kernels() {
        for &(out, inp) in &[(7usize, 128usize), (16, 256), (33, 1024), (130, 4096)] {
            let w = ternary(out, inp, &mut seed, false);
            for n in [1usize, 2, 3, 4, 5, 8, 9, 17] {
                let x = acts(n, inp, 128, &mut seed, false);
                check(name, f, &w, &x, &mut seed);
            }
        }
    }
}

#[test]
fn int8_equals_scalar_exactly() {
    let mut seed = 5u64;
    for (name, f) in kernels() {
        // decoder-like (group 128) and attention-like shapes (dh = 128 groups; probabilities × vᵀ with one long group)
        for &(out, inp, group) in &[
            (7usize, 128usize, 128usize),
            (40, 640, 128),
            (128, 384, 384),
            (19, 1024, 1024),
        ] {
            let w = int8(out, inp, group, &mut seed, false);
            for n in [1usize, 2, 5, 8, 11] {
                let x = acts(n, inp, group, &mut seed, false);
                check(name, f, &w, &x, &mut seed);
            }
        }
    }
}

#[test]
fn extremes_do_not_saturate() {
    let mut seed = 9u64;
    for (name, f) in kernels() {
        let w = ternary(32, 1024, &mut seed, true);
        let x = acts(5, 1024, 128, &mut seed, true);
        check(name, f, &w, &x, &mut seed);
        let w = int8(32, 1024, 1024, &mut seed, true);
        let x = acts(5, 1024, 1024, &mut seed, true);
        check(name, f, &w, &x, &mut seed);
    }
}

#[test]
fn f32_rows_operand_equals_scalar_exactly() {
    // the attention core quantises keys / relative positions into the weight operand (from_f32_rows)
    let mut seed = 13u64;
    for (name, f) in kernels() {
        let (out, inp) = (45usize, 128usize);
        let src: Vec<f32> = (0..out * inp)
            .map(|_| (lcg(&mut seed) % 2001) as f32 / 300.0 - 3.0)
            .collect();
        let w = TernaryMatrix::from_f32_rows(out, inp, inp, Layout::Quad16, |r, row| {
            row.copy_from_slice(&src[r * inp..(r + 1) * inp])
        });
        let x = acts(8, inp, inp, &mut seed, false);
        check(name, f, &w, &x, &mut seed);
    }
}
