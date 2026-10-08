#!/usr/bin/env bash
# PLAN 5.8: build release artefacts into dist/ for this machine (macOS arm64): CLI tarball, C library + header,
# Python wheel, Swift XCFramework zip. Linux x86-64 artefacts come from CI (.github/workflows/release.yml).
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
V="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"(.*)".*/\1/')"; ARCH="$(uname -m)"; OS="$(uname -s | tr A-Z a-z)"
D="$R/dist"; rm -rf "$D"; mkdir -p "$D"
cargo build --release -q -p voxrt-cli -p voxrt-ffi
tar -C target/release -czf "$D/voxrt-$V-$OS-$ARCH.tar.gz" voxrt
tar -czf "$D/libvoxrt-$V-$OS-$ARCH.tar.gz" -C target/release libvoxrt.a libvoxrt.dylib -C "$R" include/voxrt.h
( cd bindings/python && maturin build --release -q -o "$D" )
scripts/build_xcframework.sh >/dev/null && ( cd bindings/swift && zip -qr "$D/CVoxrt-$V.xcframework.zip" CVoxrt.xcframework )
( cd "$D" && shasum -a 256 * > SHA256SUMS )
ls -la "$D"
