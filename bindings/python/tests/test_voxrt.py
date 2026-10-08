"""voxrt Python package tests. Model-dependent tests run when a model is available: `VOXRT_MODEL_DIR`, or
`VOXRT_TEST_DOWNLOAD=1` to fetch moondream/parakeet-redux into the Hugging Face cache (CI does this, cached)."""
import os
import re
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import voxrt
from voxrt._audio import to_f32_bytes
from voxrt._model import resolve_model

CLIP = Path(__file__).parent / "audio" / "1089-134686-0001.flac"
REF = "stuff it into you his belly counselled him"


def norm(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


@pytest.fixture(scope="module")
def t():
    if not (os.environ.get("VOXRT_MODEL_DIR") or os.environ.get("VOXRT_TEST_DOWNLOAD")):
        pytest.skip("no model: set VOXRT_MODEL_DIR or VOXRT_TEST_DOWNLOAD=1")
    return voxrt.Transcriber()


# ---- no model needed ----


def test_version():
    assert re.match(r"\d+\.\d+\.\d+", voxrt.__version__)


def test_to_f32_bytes_mono_and_downmix():
    x = np.array([0.5, -0.25], dtype=np.float64)
    assert to_f32_bytes(x) == np.array([0.5, -0.25], dtype="<f4").tobytes()
    st = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)  # [n, channels]
    assert to_f32_bytes(st) == np.array([0.5, -0.5], dtype="<f4").tobytes()
    assert to_f32_bytes([0.0, 1.0]) == np.array([0.0, 1.0], dtype="<f4").tobytes()
    with pytest.raises(ValueError):
        to_f32_bytes(np.zeros((2, 2, 2)))


def test_resolve_model_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXRT_MODEL_DIR", str(tmp_path))
    assert resolve_model(None) == str(tmp_path)
    assert resolve_model(tmp_path) == str(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_model("./definitely/not/here")


# ---- with the model ----


def test_flac_file(t):
    r = t.transcribe(CLIP)
    assert norm(r.text) == REF
    assert isinstance(r, voxrt.Transcript) and isinstance(r.segments[0], voxrt.Segment)
    w = r.segments[0].words
    assert [x.word for x in w][0].lower().startswith("stuff")
    assert all(a.start <= a.end <= b.start + 1e-6 for a, b in zip(w, w[1:]))
    assert 3.0 < r.duration < 3.6 and str(r) == r.text


def test_wav_file_and_array_inputs_agree(t, tmp_path):
    x, sr = sf.read(CLIP, dtype="float32")
    wav = tmp_path / "clip.wav"
    sf.write(wav, x, sr, subtype="FLOAT")
    from_wav = t.transcribe(wav)
    from_array = t.transcribe(x, sample_rate=sr)
    stereo = t.transcribe(np.stack([x, x], axis=1), sample_rate=sr)
    assert from_wav.text == from_array.text == stereo.text
    assert norm(from_wav.text) == REF


def test_resampled_input(t):
    x, sr = sf.read(CLIP, dtype="float32")
    x48 = np.interp(np.arange(0, len(x), sr / 48000), np.arange(len(x)), x).astype(np.float32)
    assert norm(t.transcribe(x48, sample_rate=48000).text) == REF


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_ffmpeg_fallback_m4a(t, tmp_path):
    m4a = tmp_path / "clip.m4a"
    os.system(f"ffmpeg -nostdin -loglevel error -i {CLIP} {m4a}")
    assert norm(t.transcribe(m4a).text) == REF


def test_stream_equals_offline(t):
    x, sr = sf.read(CLIP, dtype="float32")
    assert sr == 16000
    offline = t.transcribe(x)
    st = t.stream()
    segs = []
    for i in range(0, len(x), 4000):
        segs += st.push(x[i : i + 4000])
    segs += st.flush()
    assert " ".join(s.text for s in segs).strip() == offline.text.strip()


def test_threads_share_one_transcriber(t):
    x, _ = sf.read(CLIP, dtype="float32")
    out = []
    th = [threading.Thread(target=lambda: out.append(t.transcribe(x).text)) for _ in range(4)]
    [h.start() for h in th]
    [h.join() for h in th]
    assert len(set(out)) == 1 and norm(out[0]) == REF


def test_unknown_format_error(t, tmp_path):
    bad = tmp_path / "x.xyz"
    bad.write_bytes(b"not audio")
    with pytest.raises(RuntimeError):
        t.transcribe(bad)
