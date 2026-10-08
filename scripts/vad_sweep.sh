#!/usr/bin/env bash
# Sweep VAD segmentation parameters on the TED-LIUM long-form set. usage: scripts/vad_sweep.sh "PAUSE PAD MIN" ...
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
for cfg in "$@"; do
  read -r PAUSE PAD MIN <<<"$cfg"
  OUT="$R/results/sweep_vad_p${PAUSE}_pad${PAD}_min${MIN}.jsonl"; rm -f "$OUT"
  VOXRT_VAD_MIN_PAUSE=$PAUSE VOXRT_VAD_PAD=$PAD VOXRT_VAD_MIN=$MIN "$R/target/release/voxrt" batch "$R/data/manifests/tedlium-long.jsonl" "$OUT" 2>/dev/null
  printf "pause=%s pad=%s min=%s  " "$PAUSE" "$PAD" "$MIN"; "$R/reference-py/.venv/bin/python" "$R/reference-py/score.py" "$OUT" "$R/data/manifests/tedlium-long.jsonl"
done
