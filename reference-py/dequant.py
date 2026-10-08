"""Dequantize parakeet-redux into a standard HF checkpoint dir: models/parakeet-redux-fp/ .
Ternary weights become F16 (exact), dense tensors keep their dtype, vad_head.* goes to a separate file."""
import json, shutil, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file, save_file
from trits import unpack

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models/parakeet-redux"
DST = ROOT / "models/parakeet-redux-fp"

def main():
    tern = json.load(open(SRC / "ternary.json"))
    cfg = json.load(open(SRC / "config.json"))
    sd = load_file(str(SRC / "model.safetensors"))
    out, vad = {}, {}
    for name, t in sd.items():
        if name.startswith("vad_head."):
            vad[name] = t
        elif name.endswith(".qweight") or name.endswith(".scales"):
            continue
        else:
            out[name] = t
    for m in tern["quantized_modules"]:
        n, i, o, g = m["name"], m["in_features"], m["out_features"], m["group_size"]
        trits = unpack(sd[n + ".qweight"], i).astype(np.float16)           # exact in f16
        scales = sd[n + ".scales"]                                           # f16
        w = trits * np.repeat(scales, g, axis=1)                             # f16 × {−1,0,1}: exact
        assert w.shape == (o, i)
        if m["as_conv1d"]:
            w = w[:, :, None]
        out[n + ".weight"] = np.ascontiguousarray(w)
    DST.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST / "model.safetensors"), metadata={"format": "pt"})
    save_file(vad, str(DST / "vad_head.safetensors"), metadata={"format": "pt"})
    hf_cfg = {k: v for k, v in cfg.items() if k not in ("ternary_modules", "ternary_group_size")}
    hf_cfg["architectures"] = ["ParakeetForTDT"]
    json.dump(hf_cfg, open(DST / "config.json", "w"), indent=2)
    shutil.copy(SRC / "tokenizer.json", DST / "tokenizer.json")
    fe = {"feature_extractor_type": "ParakeetFeatureExtractor", "feature_size": 128, "hop_length": 160, "n_fft": 512,
          "padding_side": "right", "padding_value": 0.0, "preemphasis": 0.97, "return_attention_mask": True,
          "sampling_rate": 16000, "win_length": 400}
    json.dump(fe, open(DST / "preprocessor_config.json", "w"), indent=2)                      # flat: ParakeetFeatureExtractor
    json.dump({"feature_extractor": fe, "processor_class": "ParakeetProcessor"},
              open(DST / "processor_config.json", "w"), indent=2)                             # nested: ParakeetProcessor
    print(f"wrote {DST}: {len(out)} model tensors, {len(vad)} vad tensors, {(DST/'model.safetensors').stat().st_size/1e6:.0f} MB")

if __name__ == "__main__":
    main()
