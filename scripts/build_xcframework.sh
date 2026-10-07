#!/usr/bin/env bash
# PLAN 5.3: static libprx for macOS arm64 (+ iOS device and simulator when the Rust targets are installed) →
# bindings/swift/CPrx.xcframework. usage: scripts/build_xcframework.sh
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
H="$R/bindings/swift/Headers"; cp "$R/include/prx.h" "$H/prx.h"
LIBS=()
build() { # target
  if rustup target list --installed | grep -qx "$1"; then
    cargo build --release -q -p prx-ffi --target "$1" --manifest-path "$R/Cargo.toml"
    LIBS+=("$R/target/$1/release/libprx.a")
  else
    echo "skip $1 (rustup target add $1)"
  fi
}
build aarch64-apple-darwin
build aarch64-apple-ios
build aarch64-apple-ios-sim
ARGS=(); for l in "${LIBS[@]}"; do ARGS+=(-library "$l" -headers "$H"); done
rm -rf "$R/bindings/swift/CPrx.xcframework"
xcodebuild -create-xcframework "${ARGS[@]}" -output "$R/bindings/swift/CPrx.xcframework" | tail -1
ls "$R/bindings/swift/CPrx.xcframework"
