"""Small fixtures for the Rust unit tests (crates/prx-core/tests/fixtures): first rows of three ternary modules,
raw packed bytes + f16 scales + expected unpacked trits + expected dequantised f32 (PLAN 2.3)."""
import json, struct
from pathlib import Path
import numpy as np
from trits import unpack, dequant
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/parakeet-redux"
OUT = ROOT / "crates/prx-core/tests/fixtures"; OUT.mkdir(parents=True, exist_ok=True)
with open(MODEL / "model.safetensors", "rb") as f:
    hn = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(hn)); base = 8 + hn
    def load(name):
        v = hdr[name]; a, b = v["data_offsets"]; f.seek(base + a)
        dt = {"U8": np.uint8, "F16": np.float16, "F32": np.float32}[v["dtype"]]
        return np.frombuffer(f.read(b - a), dtype=dt).reshape(v["shape"])
    tern = {m["name"]: m for m in json.load(open(MODEL / "ternary.json"))["quantized_modules"]}
    meta = {}
    for name, rows in (("encoder.layers.0.self_attn.q_proj", 4), ("encoder.layers.0.feed_forward1.linear2", 2), ("encoder.layers.7.conv.pointwise_conv1", 3)):
        m = tern[name]; i = m["in_features"]
        q = load(name + ".qweight")[:rows]; s = load(name + ".scales")[:rows]
        t = unpack(q, i); w = dequant(q, s, i)
        tag = name.replace(".", "_")
        (OUT / f"{tag}.qweight.bin").write_bytes(q.tobytes())
        (OUT / f"{tag}.scales.f16.bin").write_bytes(s.tobytes())
        (OUT / f"{tag}.trits.i8.bin").write_bytes(t.astype(np.int8).tobytes())
        (OUT / f"{tag}.weight.f32.bin").write_bytes(w.astype("<f4").tobytes())
        meta[tag] = {"rows": rows, "in_features": i, "group_size": m["group_size"], "zero_fraction_rows": float(np.mean(t == 0))}
    json.dump(meta, open(OUT / "ternary_fixtures.json", "w"), indent=1)
print({k: v["rows"] for k, v in meta.items()}, "total bytes:", sum(p.stat().st_size for p in OUT.iterdir()))

# --- frontend fixtures: librosa slaney mel filterbank (what HF uses) + PCM16 WAVs of the golden utterances ---
import librosa, soundfile as sf
fb = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000.0, norm="slaney").astype("<f4")
(OUT / "mel_fb_128x257.f32.bin").write_bytes(fb.tobytes())
for uid in (ROOT / "reference-py/golden_ids.txt").read_text().split():
    d = ROOT / "golden" / uid
    wav = np.load(d / "waveform.npy")
    sf.write(d / "audio.wav", wav, 16000, subtype="PCM_16")
    back, _ = sf.read(d / "audio.wav", dtype="float32")
    assert np.array_equal(back, wav), uid   # FLAC int16 → PCM16 WAV → f32 is exact
print("mel fb + golden wavs written")
