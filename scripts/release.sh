#!/usr/bin/env bash
# PLAN 5.8: build the macOS release artefacts into dist/, all universal (arm64 + x86_64): CLI, C library + header,
# Python wheel (universal2), Swift XCFramework zip. Linux x86-64 artefacts come from CI (.github/workflows/release.yml).
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
V="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"(.*)".*/\1/')"; OS="$(uname -s | tr A-Z a-z)"
D="$R/dist"; rm -rf "$D"; mkdir -p "$D"
# macOS artefacts are universal (arm64 + x86_64 for Intel Macs; x86_64 is cross-compiled): CLI, C library, wheel
for T in aarch64-apple-darwin x86_64-apple-darwin; do cargo build --release -q -p voxrt-cli -p voxrt-ffi --target "$T"; done
U="$R/target/universal/macos-cli"; mkdir -p "$U"
for f in voxrt libvoxrt.a libvoxrt.dylib; do
  lipo -create "target/aarch64-apple-darwin/release/$f" "target/x86_64-apple-darwin/release/$f" -output "$U/$f"
done
tar -C "$U" -czf "$D/voxrt-$V-$OS-universal.tar.gz" voxrt
tar -czf "$D/libvoxrt-$V-$OS-universal.tar.gz" -C "$U" libvoxrt.a libvoxrt.dylib -C "$R" include/voxrt.h
( cd bindings/python && maturin build --release -q --target universal2-apple-darwin -o "$D" )
scripts/build_xcframework.sh >/dev/null && ( cd bindings/swift && zip -qr "$D/CVoxrt-$V.xcframework.zip" CVoxrt.xcframework )
( cd "$D" && shasum -a 256 * > SHA256SUMS )
ls -la "$D"
