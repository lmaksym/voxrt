#!/usr/bin/env bash
# PLAN 5.3: static libvoxrt for macOS arm64 (+ iOS device and simulator when the Rust targets are installed) →
# bindings/swift/CVoxrt.xcframework. usage: scripts/build_xcframework.sh
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
H="$R/bindings/swift/Headers"; cp "$R/include/voxrt.h" "$H/voxrt.h"
LIBS=()
build() { # target
  if rustup target list --installed | grep -qx "$1"; then
    cargo build --release -q -p voxrt-ffi --target "$1" --manifest-path "$R/Cargo.toml"
    LIBS+=("$R/target/$1/release/libvoxrt.a")
  else
    echo "skip $1 (rustup target add $1)"
  fi
}
build aarch64-apple-darwin
build aarch64-apple-ios
build aarch64-apple-ios-sim
ARGS=(); for l in "${LIBS[@]}"; do ARGS+=(-library "$l" -headers "$H"); done
rm -rf "$R/bindings/swift/CVoxrt.xcframework"
xcodebuild -create-xcframework "${ARGS[@]}" -output "$R/bindings/swift/CVoxrt.xcframework" | tail -1
ls "$R/bindings/swift/CVoxrt.xcframework"
