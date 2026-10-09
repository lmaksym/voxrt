#!/usr/bin/env bash
# Run the Rust CLI over a manifest and score it with the leaderboard-exact scorer.
# usage: scripts/wer.sh <dev50|test-clean> [extra voxrt args]
set -euo pipefail
export VOXRT_DEVICE="${VOXRT_DEVICE:-cpu}"   # the CPU path is the accuracy reference; VOXRT_DEVICE=gpu / ane to score the GPU / Neural Engine
R="$(cd "$(dirname "$0")/.." && pwd)"
SET="${1:?set}"; shift || true
cargo build --release -q -p voxrt-cli --manifest-path "$R/Cargo.toml"
OUT="$R/results/rust_${SET}_${VOXRT_DEVICE}_$(date +%Y%m%d).jsonl"
"$R/target/release/voxrt" batch "$R/data/manifests/$SET.jsonl" "$OUT" "$@"
"$R/reference-py/.venv/bin/python" "$R/reference-py/score.py" "$OUT" "$R/data/manifests/$SET.jsonl" | tee "${OUT%.jsonl}.wer.txt"
