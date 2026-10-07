#!/usr/bin/env bash
# Regenerate include/prx.h from crates/prx-ffi with cbindgen (cargo install cbindgen).
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
cbindgen --config "$R/crates/prx-ffi/cbindgen.toml" --crate prx-ffi --output "$R/include/prx.h" --lockfile "$R/Cargo.lock" "$R/crates/prx-ffi"
echo "wrote include/prx.h"
