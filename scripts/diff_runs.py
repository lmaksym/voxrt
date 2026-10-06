#!/usr/bin/env python3
"""Where two runs disagree, and who is right. usage: scripts/diff_runs.py A.jsonl B.jsonl [REF.jsonl] [--show N]
Classifies every utterance: identical raw text / identical after the leaderboard normaliser / different words; for
the last class scores both sides against the reference (same WER method as score.py) and lists the word edits."""
import json, sys
from collections import Counter
from pathlib import Path
from kaldialign import align, edit_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))
from open_asr_leaderboard_normalizer.normalizer import EnglishTextNormalizer
norm = EnglishTextNormalizer()

def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in Path(p).read_text().splitlines() if l.strip())}

show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 0
args = [a for k, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[k - 1] != "--show"]
A, B = load(args[0]), load(args[1])
REF = load(args[2] if len(args) > 2 else str(ROOT / "data/manifests/test-clean.jsonl"))
ids = [i for i in REF if i in A and i in B]
same_raw = same_norm = 0; a_better = b_better = tie = 0; edits_a = Counter(); edits_b = Counter()
err_a = err_b = 0; examples = []
for i in ids:
    ta, tb = A[i]["text"], B[i]["text"]
    if ta == tb:
        same_raw += 1; continue
    na, nb, nr = norm(ta).split(), norm(tb).split(), norm(REF[i]["text"]).split()
    if na == nb:
        same_norm += 1; continue
    da = edit_distance(tuple(nr), tuple(na), merge_compounds=True)["total"]
    db = edit_distance(tuple(nr), tuple(nb), merge_compounds=True)["total"]
    err_a += da; err_b += db
    if da < db: a_better += 1
    elif db < da: b_better += 1
    else: tie += 1
    for x, y in align(tuple(na), tuple(nb), "*"):
        if x != y:
            edits_a[(x, y)] += 1
    if len(examples) < show:
        examples.append((i, da, db, " ".join(na), " ".join(nb), " ".join(nr)))
n_diff = len(ids) - same_raw - same_norm
print(f"{len(ids)} utts: identical raw {same_raw} · identical after normalisation {same_norm} · different words {n_diff}")
print(f"on the {n_diff} differing utts: A better {a_better} · B better {b_better} · same error count {tie} · errors A {err_a} vs B {err_b}")
print("most common word edits A→B:", ", ".join(f"{x}→{y} ×{c}" for (x, y), c in edits_a.most_common(25)))
for i, da, db, a, b, r in examples:
    print(f"\n[{i}] errA={da} errB={db}\n  A: {a}\n  B: {b}\n  R: {r}")
