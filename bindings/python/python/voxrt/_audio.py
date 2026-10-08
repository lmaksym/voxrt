"""Audio input helpers: non-WAV decoding (soundfile, then ffmpeg) and conversion to float32 bytes for the extension."""
from __future__ import annotations

import shutil
import subprocess
from typing import Any, Tuple


def to_f32_bytes(audio: Any) -> bytes:
    """Array-like (1-D mono, or [n, channels] averaged to mono) → little-endian float32 bytes."""
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)
    try:
        import numpy as np
    except ImportError:
        import array

        return array.array("f", audio).tobytes()
    a = np.asarray(audio, dtype="<f4")
    if a.ndim == 2:
        a = a.mean(axis=1, dtype="<f4")
    elif a.ndim != 1:
        raise ValueError(f"audio must be 1-D mono or [n, channels], got shape {a.shape}")
    return np.ascontiguousarray(a).tobytes()


def load_audio(path: str) -> Tuple[Any, int]:
    """Decode a non-WAV file to (float32 samples, sample rate): soundfile if installed, else ffmpeg (16 kHz mono)."""
    err = None
    try:
        import soundfile as sf

        try:
            x, sr = sf.read(path, dtype="float32", always_2d=False)
            return x, int(sr)
        except Exception as e:  # format not supported by libsndfile (e.g. M4A): try ffmpeg
            err = e
    except ImportError:
        pass
    if shutil.which("ffmpeg"):
        p = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
            capture_output=True,
        )
        if p.returncode == 0:
            return p.stdout, 16000
        err = RuntimeError(p.stderr.decode(errors="replace").strip() or "ffmpeg failed")
    hint = "install soundfile (pip install soundfile) for FLAC/OGG/MP3, or ffmpeg for other formats"
    raise RuntimeError(f"cannot decode {path!r}: {err or 'no decoder available'}; {hint}")
