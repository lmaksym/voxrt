"""LibriSpeech manifests (PLAN 0.5): data/manifests/<set>.jsonl with {id, audio, text, seconds}; fixed dev50 slice."""
import json, sys, random
from pathlib import Path
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
LS = ROOT / "data/librispeech/LibriSpeech"
MAN = ROOT / "data/manifests"
DEV50 = ROOT / "reference-py/dev50.txt"

def build(split: str):
    rows = []
    for trans in sorted((LS / split).rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            audio = trans.parent / f"{uid}.flac"
            info = sf.info(audio)
            rows.append({"id": uid, "audio": str(audio.relative_to(ROOT)), "text": text, "seconds": round(info.frames / info.samplerate, 3)})
    MAN.mkdir(parents=True, exist_ok=True)
    out = MAN / f"{split}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows

def main():
    sets = {}
    for split in ("test-clean", "dev-clean"):
        rows = build(split); sets[split] = rows
        print(f"{split}: {len(rows)} utts, {sum(r['seconds'] for r in rows)/3600:.2f} h")
    # dev50: fixed, reproducible, spread over durations (seed 0), committed in dev50.txt
    if not DEV50.exists():
        rows = sorted(sets["dev-clean"], key=lambda r: r["seconds"])
        rng = random.Random(0)
        idx = sorted(rng.sample(range(len(rows)), 50))
        DEV50.write_text("".join(rows[i]["id"] + "\n" for i in idx))
    ids = set(DEV50.read_text().split())
    rows = [r for r in sets["dev-clean"] if r["id"] in ids]
    (MAN / "dev50.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"dev50: {len(rows)} utts, {sum(r['seconds'] for r in rows)/60:.1f} min")

if __name__ == "__main__":
    main()
