#!/usr/bin/env bash
# A/B timing: min of N runs per file for the current env. usage: scripts/ab.sh [N]
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"; N="${1:-5}"
for f in 5694-64025-0000 3081-166546-0059 1919-142785-0007; do
  best=999
  for i in $(seq "$N"); do
    s=$("$R/target/release/prx" transcribe "$R/golden/$f/audio.wav" 2>&1 >/dev/null | tail -1 | sed -E 's/.*audio in ([0-9.]+)s.*/\1/')
    best=$(python3 -c "print(min($best,$s))")
  done
  printf "%s %ss  " "${f:0:4}" "$best"
done; echo
