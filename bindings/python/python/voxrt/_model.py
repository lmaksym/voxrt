"""Model resolution: local directory, `VOXRT_MODEL_DIR`, or a Hugging Face repo downloaded into the HF cache."""
from __future__ import annotations

import os
from typing import Optional, Union

DEFAULT_REPO = "moondream/parakeet-redux"
# the files the runtime reads; the rest of the repo (README, eval results) is not downloaded
MODEL_FILES = ["config.json", "model.safetensors", "ternary.json", "tokenizer.json"]


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
