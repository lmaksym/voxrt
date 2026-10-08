"""WER exactly as the Open ASR Leaderboard scores English: its EnglishTextNormalizer on both sides, then
kaldialign.edit_distance(..., merge_compounds=True) (leaderboard `normalizer/eval_utils.py::_align_pair`).
usage: score.py HYP.jsonl [REF.jsonl]   (rows: {id, text}; REF defaults to test-clean manifest)"""
import json, sys
from pathlib import Path
from kaldialign import edit_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))
from open_asr_leaderboard_normalizer.normalizer import EnglishTextNormalizer  # Apache-2.0, vendored

norm = EnglishTextNormalizer()

def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in Path(p).read_text().splitlines() if l.strip())}

def score(hyp_path, ref_path, merge_compounds=True, quiet=False):
    hyp, ref = load(hyp_path), load(ref_path)
    missing = [k for k in ref if k not in hyp]
    if missing:
        print(f"WARNING: {len(missing)} reference utts missing from hyp; scoring only present ones", file=sys.stderr)
    tot = n = S = D = I = 0; utts = 0
    for k in ref:
        if k not in hyp: continue
        r = norm(ref[k]["text"]).split(); h = norm(hyp[k]["text"]).split()
        if not r: continue
        d = edit_distance(tuple(r), tuple(h), merge_compounds=merge_compounds)
        tot += d["total"]; n += len(r); S += d["sub"]; D += d["del"]; I += d["ins"]; utts += 1
    wer = tot / n
    if not quiet:
        print(f"utts={utts} ref_words={n} S={S} D={D} I={I} WER={100*wer:.2f}%  (merge_compounds={merge_compounds})")
    return wer

if __name__ == "__main__":
    hyp = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "data/manifests/test-clean.jsonl")
    score(hyp, ref)
