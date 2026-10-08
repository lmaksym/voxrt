"""Run the reference over a manifest and score it.
usage: eval.py --set dev50|test-clean|dev-clean [--device cpu|mps] [--tag NAME]"""
import argparse, json, sys, time
from pathlib import Path
from voxrt_ref import Reference, ROOT
from score import score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev50"); ap.add_argument("--device", default="cpu"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    man = ROOT / f"data/manifests/{a.set}.jsonl"
    rows = [json.loads(l) for l in man.read_text().splitlines()]
    out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
    tag = a.tag or f"ref_{a.set}_{a.device}_{time.strftime('%Y%m%d')}"
    out = out_dir / f"{tag}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    ref = Reference(device=a.device)
    t0 = time.time(); audio_s = 0.0
    with open(out, "a") as f:
        for i, r in enumerate(rows):
            if r["id"] in done: continue
            res = ref.transcribe(ROOT / r["audio"])
            audio_s += r["seconds"]
            f.write(json.dumps({"id": r["id"], "text": res.text, "n_frames": res.n_frames,
                                "emissions": [[e.token, e.frame, e.duration] for e in res.emissions]}) + "\n"); f.flush()
            if (i + 1) % 50 == 0:
                el = time.time() - t0; print(f"{i+1}/{len(rows)}  {el:.0f}s  RTF×={audio_s/el:.1f}", file=sys.stderr, flush=True)
    el = time.time() - t0
    print(f"{a.set} on {a.device}: {len(rows)} utts, wall {el:.0f}s, {audio_s/max(el,1e-9):.1f}× real time (new utts only)")
    wer = score(out, man)
    (out_dir / f"{tag}.wer.txt").write_text(f"{100*wer:.3f}\n")

if __name__ == "__main__":
    main()
