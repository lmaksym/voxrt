#!/usr/bin/env python3
"""Token oracle: how many utterances of HYP.jsonl have identical emissions / identical text to REF.jsonl.
usage: scripts/oracle.py HYP.jsonl REF.jsonl   (rows: {id, text, emissions})"""
import json, sys

def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(p) if l.strip())}

hyp, ref = load(sys.argv[1]), load(sys.argv[2])
ids = [i for i in ref if i in hyp]
same_em = sum(hyp[i].get("emissions") == ref[i].get("emissions") for i in ids)
same_tx = sum(hyp[i]["text"] == ref[i]["text"] for i in ids)
print(f"{len(ids)} utts: identical emissions {same_em} · identical text {same_tx}")
