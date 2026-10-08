#!/usr/bin/env bash
# PLAN 2.13: run the Rust CLI over a manifest and score it with the leaderboard-exact scorer.
# usage: scripts/wer.sh <dev50|test-clean> [extra voxrt args]
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
SET="${1:?set}"; shift || true
cargo build --release -q -p voxrt-cli --manifest-path "$R/Cargo.toml"
OUT="$R/results/rust_${SET}_$(date +%Y%m%d).jsonl"
"$R/target/release/voxrt" batch "$R/data/manifests/$SET.jsonl" "$OUT" "$@"
"$R/reference-py/.venv/bin/python" "$R/reference-py/score.py" "$OUT" "$R/data/manifests/$SET.jsonl" | tee "${OUT%.jsonl}.wer.txt"
