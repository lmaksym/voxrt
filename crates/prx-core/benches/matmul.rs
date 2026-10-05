//! PLAN 3.1: the four ternary shapes × token counts. Reports GMAC/s so kernels are comparable.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use prx_core::tensor::linear;

const SHAPES: [(usize, usize); 4] = [(1024, 1024), (4096, 1024), (1024, 4096), (2048, 1024)];
const TOKENS: [usize; 3] = [1, 64, 512];

fn bench_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("linear_f32_scalar");
    g.sample_size(10);
    for &(out, inp) in &SHAPES {
        let w: Vec<f32> = (0..out * inp)
            .map(|i| ((i * 7919) % 3) as f32 - 1.0)
            .collect();
        for &n in &TOKENS {
            let x: Vec<f32> = (0..n * inp).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
            let mut y = vec![0f32; n * out];
            g.throughput(Throughput::Elements((n * out * inp) as u64));
            g.bench_with_input(BenchmarkId::new(format!("{out}x{inp}"), n), &n, |b, _| {
                b.iter(|| linear(black_box(&x), n, inp, black_box(&w), None, out, &mut y))
            });
        }
    }
    g.finish();
}

criterion_group!(benches, bench_f32);
criterion_main!(benches, benches_i8);

fn bench_i8(c: &mut Criterion) {
    use prx_core::kernels::gemm;
    use prx_core::packed::{QuantAct, TernaryMatrix};
    use prx_core::trits::pack_row;
    let mut g = c.benchmark_group("gemm_ternary_int8");
    g.sample_size(10);
    for &(out, inp) in &SHAPES {
        let rb = inp.div_ceil(5);
        let mut q = vec![0u8; out * rb];
        for r in 0..out {
            let trits: Vec<i8> = (0..inp).map(|i| ((i * 7919 + r) % 3) as i8 - 1).collect();
            pack_row(&trits, &mut q[r * rb..(r + 1) * rb]);
        }
        let s16: Vec<u8> = (0..out * inp / 128)
            .flat_map(|_| half::f16::from_f32(0.02).to_le_bytes())
            .collect();
        let w = TernaryMatrix::from_packed(&q, &s16, out, inp, 128);
        for &n in &TOKENS {
            let x: Vec<f32> = (0..n * inp).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
            let qa = QuantAct::quantize(&x, n, inp, 128);
            let mut y = vec![0f32; n * out];
            g.throughput(Throughput::Elements((n * out * inp) as u64));
            g.bench_with_input(BenchmarkId::new(format!("{out}x{inp}"), n), &n, |b, _| {
                b.iter(|| gemm(black_box(&w), black_box(&qa), &mut y, None))
            });
        }
    }
    g.finish();
}

criterion_group!(benches_i8, bench_i8);
