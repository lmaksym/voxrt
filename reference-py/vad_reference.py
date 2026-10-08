"""VAD head probe (PLAN 1.7): run vad_head.{proj,ctx,out} on the subsampler output of the golden utterances and the
longest dev-clean file; try candidate inter-layer activations and compare against a waveform-energy silence mask."""
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from safetensors.torch import load_file
from voxrt_ref import Reference, ROOT

def energy_mask(wav, n_frames, frame=1280, thresh_db=-40):
    e = np.array([np.mean(wav[i*frame:(i+1)*frame]**2) + 1e-12 for i in range(n_frames)])
    db = 10*np.log10(e / e.max())
    return db > thresh_db

def main():
    ref = Reference(device="cpu")
    vad = load_file(str(ROOT / "models/parakeet-redux-fp/vad_head.safetensors"))
    vad = {k: v.float() for k, v in vad.items()}
    acts = {"relu": F.relu, "silu": F.silu, "gelu": F.gelu, "tanh": torch.tanh, "none": lambda x: x}
    ids = (ROOT / "reference-py/golden_ids.txt").read_text().split()
    rows = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/manifests/dev-clean.jsonl").read_text().splitlines()}
    longest = max(rows.values(), key=lambda r: r["seconds"])
    results = {a: [] for a in acts}
    probs_out = {}
    for uid in ids + [longest["id"]]:
        r = rows[uid]; wav = ref.load_audio(ROOT / r["audio"])
        feats, mask = ref.features(wav)
        with torch.no_grad():
            sub = ref.model.encoder.subsampling(feats, mask)          # [1, T_e, 1024]
            x0 = sub.transpose(1, 2)                                 # [1, 1024, T_e]
            for name, act in acts.items():
                h = F.conv1d(x0, vad["vad_head.proj.weight"], vad["vad_head.proj.bias"])
                h = act(h)
                h = F.conv1d(h, vad["vad_head.ctx.weight"], vad["vad_head.ctx.bias"], padding=2)
                h = act(h)
                logit = F.conv1d(h, vad["vad_head.out.weight"], vad["vad_head.out.bias"])[0, 0]
                p = torch.sigmoid(logit).numpy()
                em = energy_mask(wav, len(p))
                # agreement: fraction of frames where (p>0.5) == energy mask; and mean p on loud vs quiet frames
                agree = float(np.mean((p > 0.5) == em)); loud = float(p[em].mean()) if em.any() else float("nan"); quiet = float(p[~em].mean()) if (~em).any() else float("nan")
                results[name].append((uid, agree, loud, quiet))
                probs_out[(name, uid)] = p
    print("activation | mean agreement with −40 dB energy mask | mean p(loud) | mean p(quiet)")
    for name, rs in results.items():
        ag = np.mean([a for _, a, _, _ in rs]); lo = np.nanmean([l for _, _, l, _ in rs]); qu = np.nanmean([q for _, _, _, q in rs])
        print(f"{name:5s} | {ag:.3f} | {lo:.3f} | {qu:.3f}")
    best = max(results, key=lambda n: np.mean([a for _, a, _, _ in results[n]]))
    print("best:", best)
    out = ROOT / "golden/vad"; out.mkdir(exist_ok=True)
    for (name, uid), p in probs_out.items():
        if name == best: np.save(out / f"{uid}.npy", p.astype(np.float32))
    p = probs_out[(best, longest["id"])]
    print(f"longest file {longest['id']} {longest['seconds']}s: frames={len(p)}  p<0.5 frames={int((p<0.5).sum())}  runs of silence ≥ 3 frames (240 ms):")
    runs = []; i = 0
    while i < len(p):
        if p[i] < 0.5:
            j = i
            while j < len(p) and p[j] < 0.5: j += 1
            if j - i >= 3: runs.append((i*0.08, (j-i)*0.08))
            i = j
        else: i += 1
    print("  ", [(round(s,2), round(d,2)) for s, d in runs[:15]])
    json.dump({"best_activation": best, "longest": longest["id"], "silence_runs_s": runs}, open(out / "summary.json", "w"), indent=1)

if __name__ == "__main__":
    main()
