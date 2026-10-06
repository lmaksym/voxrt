//! Single-token GEMV cost on the decoder shapes (ignored; `cargo test --release --test gemv_bench -- --ignored --nocapture`).
use prx_core::kernels::{gemm_single, kernel};
use prx_core::packed::{QuantAct, TernaryMatrix};
use std::time::Instant;

#[test]
#[ignore]
fn single_token_gemv() {
    let _ = kernel();
    use prx_core::packed::Layout;
    for &(out, inp, name, layout) in &[
        (8194usize, 640usize, "joint head rows", Layout::Rows),
        (8194, 640, "joint head rows8", Layout::Rows8),
        (2560, 640, "lstm gate rows", Layout::Rows),
        (2560, 640, "lstm gate rows8", Layout::Rows8),
    ] {
        let rows: Vec<i8> = (0..out * inp).map(|i| ((i * 7919) % 255) as i8).collect();
        let scales = vec![0.01f32; out * inp / 128];
        let w = TernaryMatrix::from_i8_rows(&rows, &scales, out, inp, 128, layout);
        let x: Vec<f32> = (0..inp).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
        let mut y = vec![0f32; out];
        let reps = 200;
        let t0 = Instant::now();
        for _ in 0..reps {
            let q = QuantAct::quantize_layout(&x, 1, inp, 128, Layout::Rows);
            gemm_single(&w, &q, &mut y, None);
        }
        let per = t0.elapsed().as_secs_f64() / reps as f64;
        let bytes = (out * inp) as f64;
        // plain streaming read of the same bytes, one core
        let t1 = Instant::now();
        let mut acc = 0i64;
        for _ in 0..reps {
            for ch in w.w.chunks_exact(64) {
                acc += ch[0] as i64 + ch[32] as i64;
            }
        }
        let read = t1.elapsed().as_secs_f64() / reps as f64;
        println!("{name:18} {out}x{inp}: gemv {:.0} µs = {:.1} GB/s, {:.0} GMAC/s · plain read {:.0} µs = {:.1} GB/s (acc {acc})",
            per * 1e6, bytes / per / 1e9, bytes / per / 1e9, read * 1e6, bytes / read / 1e9);
    }
}
