"""Model resolution: local directory, `VOXRT_MODEL_DIR`, or a Hugging Face repo downloaded into the HF cache."""
from __future__ import annotations

import os
import platform
import sys
import warnings
from typing import Optional, Union

DEFAULT_REPO = "moondream/parakeet-redux"
# the files the runtime reads; the rest of the repo (README, eval results) is not downloaded
MODEL_FILES = ["config.json", "model.safetensors", "ternary.json", "tokenizer.json"]
# the Neural Engine encoder for the default model (Core ML, Apple silicon Macs; built by tools/coreml)
COREML_REPO = "MaksL/voxrt-parakeet-redux-coreml"
COREML_FILES = ["encoder.json", "encoder.mlmodelc/*"]


def resolve_model(model: Optional[Union[str, "os.PathLike[str]"]], revision: Optional[str] = None) -> str:
    if model is None:
        env = os.environ.get("VOXRT_MODEL_DIR")
        if env:
            return env
        model = DEFAULT_REPO
    path = os.fspath(model)
    if os.path.isdir(path):
        return path
    if path.count("/") != 1 or path.startswith((".", "/", "~")):
        raise FileNotFoundError(f"model directory not found: {path!r} (pass a directory or a Hugging Face repo id)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - declared dependency
        raise ImportError("downloading a model needs huggingface_hub: pip install huggingface_hub") from e
    return snapshot_download(repo_id=path, revision=revision, allow_patterns=MODEL_FILES)


def resolve_coreml(model_dir: str, device: Optional[str], download: bool) -> Optional[str]:
    """The Core ML bundle for the Neural Engine: `VOXRT_COREML_DIR`, `<model>/coreml`, or, for the default model on
    Apple silicon Macs with device "auto"/"ane", `COREML_REPO` downloaded into the Hugging Face cache (≈ 700 MB,
    once). None otherwise."""
    if os.environ.get("VOXRT_COREML_DIR"):
        return os.environ["VOXRT_COREML_DIR"]
    local = os.path.join(model_dir, "coreml")
    if os.path.exists(os.path.join(local, "encoder.json")):
        return local
    device = device or os.environ.get("VOXRT_DEVICE", "auto")
    if not download or device not in ("auto", "ane") or sys.platform != "darwin" or platform.machine() != "arm64":
        return None
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=COREML_REPO, allow_patterns=COREML_FILES)
    except Exception as e:  # offline, or the Hub unreachable: the GPU path needs no bundle
        warnings.warn(f"voxrt: Neural Engine bundle not downloaded ({e}); using the GPU", stacklevel=3)
        return None
