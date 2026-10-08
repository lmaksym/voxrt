#!/usr/bin/env bash
# Build libvoxrt (static) and the C smoke program, run it on a golden file.
# usage: scripts/c_smoke.sh [MODEL_DIR] [FILE.wav]
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-${VOXRT_MODEL_DIR:-$R/models/parakeet-redux}}"; WAV="${2:-$R/golden/5694-64025-0000/audio.wav}"
cargo build --release -q -p voxrt-ffi --manifest-path "$R/Cargo.toml"
OUT="$R/target/c_smoke"; mkdir -p "$OUT"
cc -std=c99 -O2 -I"$R/include" "$R/examples/c/smoke.c" "$R/target/release/libvoxrt.a" -o "$OUT/smoke" -lpthread -lm $( [ "$(uname)" = Darwin ] && echo "-framework CoreFoundation -liconv" || echo "-ldl" )
"$OUT/smoke" "$MODEL" "$WAV"
