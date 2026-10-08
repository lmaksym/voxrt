//! A ternary linear layer in either precision: dequantised f32 (reference path) or packed int8 with
//! per-group activation quantisation (kernel path). `Precision` is chosen at load time.
use crate::format::ModelFile;
use crate::kernels::{gemm_ep, Epilogue};
use crate::packed::{QuantAct, TernaryMatrix};
use crate::tensor::linear;
use crate::trits::dequant_matrix;
use crate::{Error, Result};
use std::sync::atomic::{AtomicU64, Ordering};

/// Nanoseconds spent quantising activations / in GEMMs (diagnostics, `VOXRT_PROFILE`).
pub static QUANT_NS: AtomicU64 = AtomicU64::new(0);
pub static GEMM_NS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    Int8,
}

impl Precision {
    pub fn from_env() -> Self {
        match std::env::var("VOXRT_PRECISION").as_deref() {
            Ok("f32") => Precision::F32,
            _ => Precision::Int8,
        }
    }
}

pub enum TernaryLinear {
    F32 { w: Vec<f32>, out: usize, inp: usize },
    Int8(TernaryMatrix),
}

/// Quantised (or raw) input shared across several linears that consume the same activation.
pub enum ActIn<'a> {
    F32(&'a [f32]),
    F32Owned(Vec<f32>),
    Int8(QuantAct),
}

impl ActIn<'_> {
    /// The f32 rows when available (reference path), for tracing.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            ActIn::F32(x) => Some(x),
            ActIn::F32Owned(v) => Some(v),
            ActIn::Int8(_) => None,
        }
    }
}

impl TernaryLinear {
    pub fn load(m: &ModelFile, name: &str, precision: Precision) -> Result<Self> {
        Self::load_stacked(m, &[name], precision)
    }

    /// Several ternary modules with the same input stacked along the output dimension (e.g. q/k/v) into one GEMM.
    pub fn load_stacked(m: &ModelFile, names: &[&str], precision: Precision) -> Result<Self> {
        let mut qw_all: Vec<u8> = Vec::new();
        let mut sc_all: Vec<u8> = Vec::new();
        let (mut out, mut inp, mut group) = (0usize, 0usize, 0usize);
        for name in names {
            let q = m
                .ternary
                .quantized_modules
                .iter()
                .find(|q| q.name == *name)
                .ok_or_else(|| Error::Format(format!("{name} is not a ternary module")))?;
            if inp != 0 && (q.in_features != inp || q.group_size != group) {
                return Err(Error::Format(format!(
                    "{name}: cannot stack, shapes differ"
                )));
            }
            inp = q.in_features;
            group = q.group_size;
            out += q.out_features;
            let (_, qw) = m.tensor(&format!("{name}.qweight"))?;
            let (_, sc) = m.tensor(&format!("{name}.scales"))?;
            qw_all.extend_from_slice(qw);
            sc_all.extend_from_slice(sc);
        }
        Ok(match precision {
            Precision::F32 => TernaryLinear::F32 {
                w: dequant_matrix(&qw_all, &sc_all, out, inp, group),
                out,
                inp,
            },
            Precision::Int8 => TernaryLinear::Int8(TernaryMatrix::from_packed_layout(
                &qw_all,
                &sc_all,
                out,
                inp,
                group,
                crate::kernels::kernel().layout(),
            )),
        })
    }

    pub fn is_int8(&self) -> bool {
        matches!(self, TernaryLinear::Int8(_))
    }

    pub fn out_features(&self) -> usize {
        match self {
            TernaryLinear::F32 { out, .. } => *out,
            TernaryLinear::Int8(m) => m.out,
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            TernaryLinear::F32 { inp, .. } => *inp,
            TernaryLinear::Int8(m) => m.inp,
        }
    }

    /// Prepare an input `[n, in]` for this layer (quantises once; reuse for siblings with the same input).
    pub fn prepare<'a>(&self, x: &'a [f32], n: usize) -> ActIn<'a> {
        match self {
            TernaryLinear::F32 { .. } => ActIn::F32(x),
            TernaryLinear::Int8(m) => {
                let t0 = std::time::Instant::now();
                let q = QuantAct::quantize_layout(x, n, m.inp, m.group, act_layout(m.layout));
                QUANT_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                ActIn::Int8(q)
            }
        }
    }

    /// Fused LayerNorm(x) → input: the normalised row goes straight into the int8 layout (no f32 intermediate).
    pub fn prepare_normed<'a>(
        &self,
        x: &[f32],
        n: usize,
        gamma: &[f32],
        beta: &[f32],
    ) -> ActIn<'a> {
        let d = self.in_features();
        let ln_row = |t: usize, row: &mut [f32]| {
            crate::tensor::layer_norm_row(&x[t * d..(t + 1) * d], gamma, beta, row)
        };
        match self {
            TernaryLinear::F32 { .. } => {
                let mut out = vec![0f32; n * d];
                crate::tensor::par_rows(&mut out, d, |t, row| ln_row(t, row));
                ActIn::F32Owned(out)
            }
            TernaryLinear::Int8(m) => {
                let t0 = std::time::Instant::now();
                let q = QuantAct::from_rows(n, m.inp, m.group, act_layout(m.layout), ln_row);
                QUANT_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                ActIn::Int8(q)
            }
        }
    }

    pub fn forward_prepared(&self, a: &ActIn, n: usize, y: &mut [f32]) {
        self.forward_prepared_ep(a, n, y, Epilogue::Store)
    }

    /// GEMM with an epilogue applied at the store (`Silu`, or `AddScaled` into a pre-filled residual `y`).
    pub fn forward_prepared_ep(&self, a: &ActIn, n: usize, y: &mut [f32], ep: Epilogue) {
        let t0 = std::time::Instant::now();
        match (self, a.as_f32()) {
            (TernaryLinear::F32 { w, out, inp }, Some(x)) => {
                if ep == Epilogue::Store {
                    linear(x, n, *inp, w, None, *out, y);
                } else {
                    let mut tmp = vec![0f32; n * out];
                    linear(x, n, *inp, w, None, *out, &mut tmp);
                    for (yv, v) in y.iter_mut().zip(&tmp) {
                        *yv = ep.apply(*yv, *v);
                    }
                }
            }
            (TernaryLinear::Int8(m), None) => match a {
                ActIn::Int8(q) => gemm_ep(m, q, y, None, ep),
                _ => unreachable!(),
            },
            _ => panic!("precision mismatch between layer and prepared input"),
        }
        GEMM_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    pub fn forward(&self, x: &[f32], n: usize, y: &mut [f32]) {
        let a = self.prepare(x, n);
        self.forward_prepared(&a, n, y)
    }
}

/// Activation layout matching a weight layout (2-bit packed weights consume pair-interleaved int8 activations).
fn act_layout(w: crate::packed::Layout) -> crate::packed::Layout {
    match w {
        crate::packed::Layout::Packed2 => crate::packed::Layout::Pairs,
        l => l,
    }
}
