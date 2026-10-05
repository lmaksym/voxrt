//! Model format: `config.json`, `ternary.json` and the safetensors file (PLAN 2.2).
//! Spec: `docs/model-format.md`, inventory: `docs/tensors.md`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_mel_bins: usize,
    pub conv_kernel_size: usize,
    pub subsampling_conv_channels: usize,
    pub subsampling_conv_kernel_size: usize,
    pub subsampling_conv_stride: usize,
    pub subsampling_factor: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub scale_input: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub encoder_config: EncoderConfig,
    pub blank_token_id: usize,
    pub decoder_hidden_size: usize,
    pub num_decoder_layers: usize,
    pub durations: Vec<usize>,
    pub max_symbols_per_step: usize,
    pub vocab_size: usize,
    pub ternary_group_size: usize,
    pub ternary_modules: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizedModule {
    pub name: String,
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
    pub as_conv1d: bool,
    pub has_bias: bool,
    pub zero_fraction: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VadHeadSpec {
    pub d_model: usize,
    pub hidden: usize,
    pub kernel: usize,
    pub frame_seconds: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TernarySpec {
    pub format: String,
    pub names: String,
    pub vad_head: VadHeadSpec,
    pub quantized_modules: Vec<QuantizedModule>,
    pub n_quantized_params: u64,
    pub n_dense_params: u64,
}

pub const FORMAT: &str = "thrush-ternary-v2";

/// A loaded model directory: parsed configs plus the memory-mapped safetensors file.
pub struct ModelFile {
    pub dir: PathBuf,
    pub config: Config,
    pub ternary: TernarySpec,
    mmap: Mmap,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl ModelFile {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let config: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let ternary: TernarySpec =
            serde_json::from_slice(&std::fs::read(dir.join("ternary.json"))?)?;
        if ternary.format != FORMAT {
            return Err(Error::Format(format!(
                "unsupported format {:?}, expected {FORMAT}",
                ternary.format
            )));
        }
        if ternary.names != "hf" {
            return Err(Error::Format(format!(
                "unsupported tensor naming {:?}",
                ternary.names
            )));
        }
        let file = std::fs::File::open(dir.join("model.safetensors"))?;
        // SAFETY: the file is opened read-only and never modified while mapped.
        let mmap = unsafe { Mmap::map(&file)? };
        SafeTensors::deserialize(&mmap)?; // header must parse
        Ok(Self {
            dir,
            config,
            ternary,
            mmap,
        })
    }

    pub fn tensors(&self) -> SafeTensors<'_> {
        SafeTensors::deserialize(&self.mmap).expect("validated at open")
    }

    pub fn tensor_infos(&self) -> BTreeMap<String, TensorInfo> {
        self.tensors()
            .tensors()
            .into_iter()
            .map(|(n, v)| {
                (
                    n,
                    TensorInfo {
                        dtype: v.dtype(),
                        shape: v.shape().to_vec(),
                    },
                )
            })
            .collect()
    }

    /// Raw bytes and dtype/shape of one tensor.
    pub fn tensor(&self, name: &str) -> Result<(TensorInfo, &[u8])> {
        let st = self.tensors();
        let v = st
            .tensor(name)
            .map_err(|_| Error::Format(format!("missing tensor {name}")))?;
        Ok((
            TensorInfo {
                dtype: v.dtype(),
                shape: v.shape().to_vec(),
            },
            v.data(),
        ))
    }

    /// Every expected tensor exists with the right dtype and shape; returns the list of mismatches.
    pub fn validate(&self) -> Vec<String> {
        let infos = self.tensor_infos();
        let mut problems = Vec::new();
        let mut extra = Vec::new();
        let mut expect = |name: &str, dtype: Dtype, shape: Vec<usize>| match infos.get(name) {
            None => problems.push(format!("missing {name}")),
            Some(i) if i.dtype != dtype || i.shape != shape => problems.push(format!(
                "{name}: got {:?} {:?}, expected {dtype:?} {shape:?}",
                i.dtype, i.shape
            )),
            _ => {}
        };
        let c = &self.config;
        let e = &c.encoder_config;
        let d = e.hidden_size;
        let h = c.decoder_hidden_size;
        for m in &self.ternary.quantized_modules {
            expect(
                &format!("{}.qweight", m.name),
                Dtype::U8,
                vec![m.out_features, m.in_features.div_ceil(5)],
            );
            expect(
                &format!("{}.scales", m.name),
                Dtype::F16,
                vec![m.out_features, m.in_features / m.group_size],
            );
            if m.has_bias {
                expect(
                    &format!("{}.bias", m.name),
                    Dtype::F32,
                    vec![m.out_features],
                );
            }
            if infos.contains_key(&format!("{}.weight", m.name)) {
                extra.push(format!("{}.weight present alongside qweight", m.name));
            }
        }
        let ch = e.subsampling_conv_channels;
        let k = e.subsampling_conv_kernel_size;
        for (idx, shape) in [
            (0, vec![ch, 1, k, k]),
            (2, vec![ch, 1, k, k]),
            (3, vec![ch, ch, 1, 1]),
            (5, vec![ch, 1, k, k]),
            (6, vec![ch, ch, 1, 1]),
        ] {
            expect(
                &format!("encoder.subsampling.layers.{idx}.weight"),
                Dtype::F16,
                shape,
            );
            expect(
                &format!("encoder.subsampling.layers.{idx}.bias"),
                Dtype::F32,
                vec![ch],
            );
        }
        let sub_out = e.num_mel_bins / e.subsampling_factor;
        expect(
            "encoder.subsampling.linear.weight",
            Dtype::F16,
            vec![d, ch * sub_out],
        );
        expect("encoder.subsampling.linear.bias", Dtype::F32, vec![d]);
        let heads = e.num_attention_heads;
        for i in 0..e.num_hidden_layers {
            let p = format!("encoder.layers.{i}");
            for n in [
                "norm_feed_forward1",
                "norm_self_att",
                "norm_conv",
                "norm_feed_forward2",
                "norm_out",
                "conv.norm",
            ] {
                expect(&format!("{p}.{n}.weight"), Dtype::F32, vec![d]);
                expect(&format!("{p}.{n}.bias"), Dtype::F32, vec![d]);
            }
            expect(&format!("{p}.conv.norm.running_mean"), Dtype::F32, vec![d]);
            expect(&format!("{p}.conv.norm.running_var"), Dtype::F32, vec![d]);
            expect(
                &format!("{p}.conv.depthwise_conv.weight"),
                Dtype::F16,
                vec![d, 1, e.conv_kernel_size],
            );
            expect(
                &format!("{p}.self_attn.bias_u"),
                Dtype::F16,
                vec![heads, d / heads],
            );
            expect(
                &format!("{p}.self_attn.bias_v"),
                Dtype::F16,
                vec![heads, d / heads],
            );
        }
        expect("encoder_projector.weight", Dtype::F16, vec![h, d]);
        expect("encoder_projector.bias", Dtype::F32, vec![h]);
        expect(
            "decoder.embedding.weight",
            Dtype::F16,
            vec![c.vocab_size, h],
        );
        for l in 0..c.num_decoder_layers {
            expect(
                &format!("decoder.lstm.weight_ih_l{l}"),
                Dtype::F16,
                vec![4 * h, h],
            );
            expect(
                &format!("decoder.lstm.weight_hh_l{l}"),
                Dtype::F16,
                vec![4 * h, h],
            );
            expect(
                &format!("decoder.lstm.bias_ih_l{l}"),
                Dtype::F32,
                vec![4 * h],
            );
            expect(
                &format!("decoder.lstm.bias_hh_l{l}"),
                Dtype::F32,
                vec![4 * h],
            );
        }
        expect("decoder.decoder_projector.weight", Dtype::F16, vec![h, h]);
        expect("decoder.decoder_projector.bias", Dtype::F32, vec![h]);
        expect(
            "joint.head.weight",
            Dtype::F16,
            vec![c.vocab_size + c.durations.len(), h],
        );
        expect(
            "joint.head.bias",
            Dtype::F32,
            vec![c.vocab_size + c.durations.len()],
        );
        let v = &self.ternary.vad_head;
        expect(
            "vad_head.proj.weight",
            Dtype::F16,
            vec![v.hidden, v.d_model, 1],
        );
        expect("vad_head.proj.bias", Dtype::F16, vec![v.hidden]);
        expect(
            "vad_head.ctx.weight",
            Dtype::F16,
            vec![v.hidden, v.hidden, v.kernel],
        );
        expect("vad_head.ctx.bias", Dtype::F16, vec![v.hidden]);
        expect("vad_head.out.weight", Dtype::F16, vec![1, v.hidden, 1]);
        expect("vad_head.out.bias", Dtype::F16, vec![1]);
        problems.extend(extra);
        problems
    }
}

/// Default model directory used by tests and the CLI when none is given.
pub fn default_model_dir() -> PathBuf {
    if let Ok(p) = std::env::var("PRX_MODEL_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/parakeet-redux")
}
