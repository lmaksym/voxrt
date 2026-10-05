//! A ternary linear layer in either precision: dequantised f32 (reference path) or packed int8 with
//! per-group activation quantisation (kernel path). `Precision` is chosen at load time.
use crate::format::ModelFile;
use crate::kernels::gemm;
use crate::packed::{QuantAct, TernaryMatrix};
use crate::tensor::linear;
use crate::trits::dequant_matrix;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    Int8,
}

impl Precision {
    pub fn from_env() -> Self {
        match std::env::var("PRX_PRECISION").as_deref() {
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
    Int8(QuantAct),
}

impl TernaryLinear {
    pub fn load(m: &ModelFile, name: &str, precision: Precision) -> Result<Self> {
        let q = m
            .ternary
            .quantized_modules
            .iter()
            .find(|q| q.name == name)
            .ok_or_else(|| Error::Format(format!("{name} is not a ternary module")))?;
        let (_, qw) = m.tensor(&format!("{name}.qweight"))?;
        let (_, sc) = m.tensor(&format!("{name}.scales"))?;
        Ok(match precision {
            Precision::F32 => TernaryLinear::F32 {
                w: dequant_matrix(qw, sc, q.out_features, q.in_features, q.group_size),
                out: q.out_features,
                inp: q.in_features,
            },
            Precision::Int8 => TernaryLinear::Int8(TernaryMatrix::from_packed(
                qw,
                sc,
                q.out_features,
                q.in_features,
                q.group_size,
            )),
        })
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
            TernaryLinear::Int8(m) => ActIn::Int8(QuantAct::quantize(x, n, m.inp, m.group)),
        }
    }

    pub fn forward_prepared(&self, a: &ActIn, n: usize, y: &mut [f32]) {
        match (self, a) {
            (TernaryLinear::F32 { w, out, inp }, ActIn::F32(x)) => {
                linear(x, n, *inp, w, None, *out, y)
            }
            (TernaryLinear::Int8(m), ActIn::Int8(q)) => gemm(m, q, y, None),
            _ => panic!("precision mismatch between layer and prepared input"),
        }
    }

    pub fn forward(&self, x: &[f32], n: usize, y: &mut [f32]) {
        let a = self.prepare(x, n);
        self.forward_prepared(&a, n, y)
    }
}
