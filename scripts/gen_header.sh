#!/usr/bin/env bash
# Regenerate include/voxrt.h from crates/voxrt-ffi with cbindgen (cargo install cbindgen).
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
cbindgen --config "$R/crates/voxrt-ffi/cbindgen.toml" --crate voxrt-ffi --output "$R/include/voxrt.h" --lockfile "$R/Cargo.lock" "$R/crates/voxrt-ffi"
echo "wrote include/voxrt.h"
