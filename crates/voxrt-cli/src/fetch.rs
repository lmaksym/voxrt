//! Model resolution with a first-run download into the Hugging Face cache (`~/.cache/huggingface/hub`, or `HF_HOME`),
//! the same cache and files the Python package uses. Later runs read the cache and need no network.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::Cache;
use voxrt::format::default_model_dir;
use voxrt::transcriber::Device;

const MODEL_REPO: &str = "moondream/parakeet-redux";
/// The files the runtime reads; the rest of the repo (README, eval results) is not downloaded.
const MODEL_FILES: [&str; 4] = [
    "config.json",
    "model.safetensors",
    "ternary.json",
    "tokenizer.json",
];
/// The Neural Engine encoder for the default model (Core ML, built by tools/coreml).
const COREML_REPO: &str = "MaksL/voxrt-parakeet-redux-coreml";

fn api(quiet: bool) -> Result<Api> {
    Ok(ApiBuilder::from_env().with_progress(!quiet).build()?)
}

/// The cached snapshot directory of `repo` when it already holds `file`.
fn cached(repo: &str, file: &str) -> Option<PathBuf> {
    Cache::from_env()
        .model(repo.to_string())
        .get(file)?
        .parent()
        .map(Path::to_path_buf)
}

/// `--model DIR`, else `VOXRT_MODEL_DIR`, else the source tree's `models/parakeet-redux` when it holds a model,
/// else `moondream/parakeet-redux` from the Hugging Face cache (downloaded on first use, ≈ 180 MB). The flag says
/// whether the default model was picked, which is the only one the Core ML bundle is built for.
pub fn model_dir(explicit: Option<PathBuf>, quiet: bool) -> Result<(PathBuf, bool)> {
    if let Some(dir) = explicit {
        return Ok((dir, false));
    }
    let local = default_model_dir();
    if std::env::var_os("VOXRT_MODEL_DIR").is_some() || local.join("model.safetensors").exists() {
        return Ok((local, false));
    }
    if let Some(dir) = MODEL_FILES
        .iter()
        .map(|f| cached(MODEL_REPO, f))
        .collect::<Option<Vec<_>>>()
    {
        return Ok((dir[0].clone(), true));
    }
    if !quiet {
        eprintln!("voxrt: fetching {MODEL_REPO} (≈ 180 MB, first run only)");
    }
    let repo = api(quiet)?.model(MODEL_REPO.to_string());
    let mut dir = None;
    for f in MODEL_FILES {
        let path = repo.get(f).with_context(|| {
            format!("downloading {MODEL_REPO}/{f} (pass --model DIR to use a local copy)")
        })?;
        dir = path.parent().map(Path::to_path_buf);
    }
    Ok((dir.expect("MODEL_FILES is not empty"), true))
}

/// The Core ML bundle for `VOXRT_DEVICE=ane`: `VOXRT_COREML_DIR` or `<model>/coreml` when present, else, for the
/// default model on an Apple-silicon Mac, `MaksL/voxrt-parakeet-redux-coreml` from the Hugging Face cache
/// (≈ 300 MB, once). None otherwise, or when the download fails: the GPU path needs no bundle.
pub fn coreml_dir(model_dir: &Path, default_model: bool, quiet: bool) -> Option<PathBuf> {
    if Device::from_env() != Device::Ane
        || !default_model
        || !cfg!(all(target_os = "macos", target_arch = "aarch64"))
        || std::env::var_os("VOXRT_COREML_DIR").is_some()
        || model_dir.join("coreml/encoder.json").exists()
    {
        return None;
    }
    // encoder.json is fetched last, so its presence in the cache means the whole bundle is there
    if let Some(dir) = cached(COREML_REPO, "encoder.json") {
        return Some(dir);
    }
    let fetch = || -> Result<PathBuf> {
        let repo = api(quiet)?.model(COREML_REPO.to_string());
        let files: Vec<String> = repo
            .info()?
            .siblings
            .into_iter()
            .map(|s| s.rfilename)
            .filter(|f| f == "encoder.json" || f.starts_with("encoder.mlmodelc/"))
            .collect();
        for f in files.iter().filter(|f| f.as_str() != "encoder.json") {
            repo.get(f)?;
        }
        let json = repo.get("encoder.json")?;
        Ok(json.parent().expect("a file in the snapshot").to_path_buf())
    };
    if !quiet {
        eprintln!("voxrt: fetching {COREML_REPO} (≈ 300 MB, first run only)");
    }
    match fetch() {
        Ok(dir) => Some(dir),
        Err(e) => {
            eprintln!("voxrt: Neural Engine bundle not downloaded ({e}); using the GPU");
            None
        }
    }
}
