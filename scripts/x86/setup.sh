#!/usr/bin/env bash
# Runs on the x86 box (root, Ubuntu 24.04): toolchain + LibriSpeech test-clean / dev-clean.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq build-essential pkg-config curl ffmpeg parallel >/dev/null
[ -x "$HOME/.cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
mkdir -p /root/voxrt/data/librispeech && cd /root/voxrt/data/librispeech
for set in test-clean dev-clean; do
  [ -d "LibriSpeech/$set" ] || curl -sfL "https://openslr.elda.org/resources/12/$set.tar.gz" | tar xz
done
# `voxrt batch` reads 16 kHz mono PCM WAV next to each .flac (the CLI has no FLAC decoder)
find LibriSpeech -name '*.flac' | parallel --will-cite -j"$(nproc)" \
  '[ -f {.}.wav ] || ffmpeg -loglevel error -i {} -ac 1 -ar 16000 -c:a pcm_s16le {.}.wav'
echo "setup done: $(ls LibriSpeech)"
