#!/usr/bin/env bash
# Package power per device under a sustained transcription load, from powermetrics (needs root: the CPU, GPU and
# Neural Engine rails are only readable through it). `top`'s energy impact cannot see the Neural Engine, so this is
# the comparison that decides CPU vs GPU vs Neural Engine on energy.
# usage: sudo bench/power.sh [AUDIO.wav] [clip seconds]   (default data/tedlium/ted06.wav, 10 s clips)
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo (powermetrics needs root)"; exit 1; }
R="$(cd "$(dirname "$0")/.." && pwd)"
AUDIO="${1:-$R/data/tedlium/ted06.wav}"
SECS="${2:-10}"
EXE="$R/target/release/voxrt"
USER_NAME="${SUDO_USER:-$USER}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

idle() {
  powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 5 2>/dev/null >"$TMP/idle.txt"
}

measure() { # device → mean mW per rail while the load runs, plus audio processed per second
  local dev="$1"
  # run the load as the invoking user (model cache, Core ML compile cache live in their home)
  sudo -u "$USER_NAME" env VOXRT_DEVICE="$dev" "$EXE" bench "$AUDIO" --seconds "$SECS" --repeat 5000 \
    >/dev/null 2>&1 &
  local pid=$!
  sleep 4 # model load + warm-up
  powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 15 2>/dev/null >"$TMP/$dev.txt"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

mean() { # rail regex, file → mean mW
  grep -E "^$1 Power" "$2" | awk '{s += $(NF-1); n++} END {printf "%.0f", n ? s / n : 0}'
}

idle
for d in cpu gpu ane; do measure "$d"; done
printf "\n%-6s %8s %8s %8s %9s   (mW above idle, %s s clips)\n" device CPU GPU ANE total "$SECS"
ic=$(mean CPU "$TMP/idle.txt"); ig=$(mean GPU "$TMP/idle.txt"); ia=$(mean ANE "$TMP/idle.txt")
for d in cpu gpu ane; do
  c=$(( $(mean CPU "$TMP/$d.txt") - ic )); g=$(( $(mean GPU "$TMP/$d.txt") - ig )); a=$(( $(mean ANE "$TMP/$d.txt") - ia ))
  printf "%-6s %8d %8d %8d %9d\n" "$d" "$c" "$g" "$a" $((c + g + a))
done
echo "idle: CPU ${ic} mW · GPU ${ig} mW · ANE ${ia} mW"
echo "(back-to-back load: energy per audio-minute = total mW × 60 ÷ (× real time), × real time from bench/prod_bench.py)"
