#!/usr/bin/env bash
# PLAN 5.3: static libvoxrt → bindings/swift/CVoxrt.xcframework with three slices: macOS (universal: arm64 + x86_64
# for Intel Macs), iOS device (arm64), iOS simulator (arm64 + x86_64). Missing Rust targets are skipped with a note.
# usage: scripts/build_xcframework.sh
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
H="$R/bindings/swift/Headers"; cp "$R/include/voxrt.h" "$H/voxrt.h"
U="$R/target/universal"; mkdir -p "$U"
have() { rustup target list --installed | grep -qx "$1"; }
slice() { # name targets... → one (fat) static library for the slice, path on stdout
  local name="$1"; shift; local libs=()
  for t in "$@"; do
    if have "$t"; then
      cargo build --release -q -p voxrt-ffi --target "$t" --manifest-path "$R/Cargo.toml" >&2
      libs+=("$R/target/$t/release/libvoxrt.a")
    else
      echo "skip $t (rustup target add $t)" >&2
    fi
  done
  [ ${#libs[@]} -eq 0 ] && return 0
  mkdir -p "$U/$name"; lipo -create "${libs[@]}" -output "$U/$name/libvoxrt.a"
  echo "$U/$name/libvoxrt.a"
}
ARGS=()
for s in "macos aarch64-apple-darwin x86_64-apple-darwin" "ios aarch64-apple-ios" \
         "ios-sim aarch64-apple-ios-sim x86_64-apple-ios"; do
  # shellcheck disable=SC2086
  l=$(slice $s); [ -n "$l" ] && ARGS+=(-library "$l" -headers "$H")
done
rm -rf "$R/bindings/swift/CVoxrt.xcframework"
xcodebuild -create-xcframework "${ARGS[@]}" -output "$R/bindings/swift/CVoxrt.xcframework" | tail -1
ls "$R/bindings/swift/CVoxrt.xcframework"
