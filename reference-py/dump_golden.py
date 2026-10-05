"""Dump golden tensors for 5 fixed dev-clean utterances (PLAN 1.6) → golden/<id>/*.npy + golden/manifest.json.
CPU, fp32, deterministic. Hook points are documented in docs/golden.md."""
import hashlib, json, re, sys, platform
from pathlib import Path
import numpy as np, torch, transformers
from prx_ref import Reference, ROOT

GOLDEN = ROOT / "golden"
IDS_FILE = ROOT / "reference-py/golden_ids.txt"

def pick_ids():
    rows = [json.loads(l) for l in (ROOT / "data/manifests/dev-clean.jsonl").read_text().splitlines()]
    rows.sort(key=lambda r: r["seconds"])
    short, median, long = rows[5], rows[len(rows)//2], rows[-20]          # avoid the extreme tails
    numbers = next(r for r in rows if re.search(r"\b(HUNDRED|THOUSAND|TWENTY|FIFTEEN)\b", r["text"]) and 4 < r["seconds"] < 12)
    longer = next(r for r in rows if 15 < r["seconds"] < 22 and r["id"] not in {long["id"]})
    return [short["id"], median["id"], numbers["id"], longer["id"], long["id"]]

def main():
    torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    if not IDS_FILE.exists():
        IDS_FILE.write_text("\n".join(pick_ids()) + "\n")
    ids = IDS_FILE.read_text().split()
    rows = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/manifests/dev-clean.jsonl").read_text().splitlines()}
    ref = Reference(device="cpu")
    m = ref.model
    manifest = {"transformers": transformers.__version__, "torch": torch.__version__, "platform": platform.platform(),
                "model_sha256": hashlib.sha256((ROOT / "models/parakeet-redux/model.safetensors").read_bytes()).hexdigest(),
                "dtype": "float32", "device": "cpu", "utterances": {}}
    for uid in ids:
        r = rows[uid]; d = GOLDEN / uid; d.mkdir(parents=True, exist_ok=True)
        wav = ref.load_audio(ROOT / r["audio"])
        caps = {}
        def hook(name):
            def f(mod, inp, out):
                caps[name] = (out[0] if isinstance(out, tuple) else out).detach().cpu().numpy()
            return f
        hs = [m.encoder.subsampling.register_forward_hook(hook("subsampling_out"))]
        hs.append(m.encoder.encode_positions.register_forward_hook(hook("pos_emb")))
        for i, layer in enumerate(m.encoder.layers):
            hs.append(layer.register_forward_hook(hook(f"layer{i:02d}_out")))
        l0 = m.encoder.layers[0]
        hs += [l0.feed_forward1.register_forward_hook(hook("layer00_ff1_out")),
               l0.norm_feed_forward1.register_forward_hook(hook("layer00_norm_ff1_out")),
               l0.self_attn.register_forward_hook(hook("layer00_attn_out")),
               l0.norm_self_att.register_forward_hook(hook("layer00_norm_attn_out")),
               l0.conv.register_forward_hook(hook("layer00_conv_out")),
               l0.norm_conv.register_forward_hook(hook("layer00_norm_conv_out")),
               l0.feed_forward2.register_forward_hook(hook("layer00_ff2_out")),
               l0.norm_out.register_forward_hook(hook("layer00_norm_out_out"))]
        # features: pre-normalization mel via the extractor's internal fn, normalized via the public call
        fe = ref.fe
        x = torch.tensor(wav)[None]
        x_pre = torch.cat([x[:, :1], x[:, 1:] - fe.preemphasis * x[:, :-1]], dim=1)
        logmel = fe._torch_extract_fbank_features(x_pre)            # [1, T, 128], before normalization
        feats, mask = ref.features(wav)
        with torch.no_grad():
            h, proj = ref.encode(feats, mask)
        for hh in hs: hh.remove()
        em, steps = ref.greedy_tdt(proj, trace=True)
        text = ref.decode_text(em)
        arrays = {"waveform": wav.astype(np.float32), "preemph": x_pre[0].numpy(), "logmel": logmel[0].numpy(),
                  "features": feats[0].cpu().numpy(), "encoder_out": h.cpu().numpy(), "encoder_proj": proj.cpu().numpy(),
                  "step_logits": np.stack([s["logits"] for s in steps]),
                  "step_h": np.stack([s["h"][:, 0] for s in steps]), "step_c": np.stack([s["c"][:, 0] for s in steps]),
                  "step_t": np.array([s["t"] for s in steps], dtype=np.int32),
                  "step_token": np.array([s["token"] for s in steps], dtype=np.int32),
                  "step_dur": np.array([s["dur"] for s in steps], dtype=np.int32),
                  "emissions": np.array([[e.token, e.frame, e.duration] for e in em], dtype=np.int32).reshape(-1, 3)}
        for k, v in caps.items(): arrays[k] = v[0] if v.ndim == 3 else v
        for k, v in arrays.items(): np.save(d / f"{k}.npy", np.ascontiguousarray(v))
        manifest["utterances"][uid] = {"audio": r["audio"], "audio_sha256": hashlib.sha256((ROOT / r["audio"]).read_bytes()).hexdigest(),
                                       "seconds": r["seconds"], "ref_text": r["text"], "hyp_text": text,
                                       "n_samples": int(len(wav)), "n_feature_frames": int(feats.shape[1]), "n_encoder_frames": int(h.shape[0]),
                                       "n_steps": len(steps), "n_emissions": len(em),
                                       "shapes": {k: list(v.shape) for k, v in arrays.items()}}
        print(f"{uid}: {r['seconds']}s feats={feats.shape[1]} enc={h.shape[0]} steps={len(steps)} tokens={len(em)}  | {text}")
    json.dump(manifest, open(GOLDEN / "manifest.json", "w"), indent=1)
    print("manifest written")

if __name__ == "__main__":
    main()
