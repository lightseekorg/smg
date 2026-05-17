#!/usr/bin/env bash
#
# Verify that every COPIED upstream Rust source file under
# `crates/vendor_auth/src/oci/` carries the Oracle copyright header required
# by UPL-1.0 (R9).
#
# `mod.rs` is SMG-authored (the module declaration / re-exports surface) and
# is NOT a copy of upstream code, so it is skipped.
#
# This script is wired into `cargo test` via
# `crates/vendor_auth/tests/license_header.rs`. It also runs as part of CI.
#
# Exit codes:
#   0 — all copied files have the header
#   1 — at least one copied file is missing the header
#
set -euo pipefail
cd "$(dirname "$0")/../src/oci"
fail=0
checked=0
for f in *.rs; do
    # Skip the SMG-authored module-declaration file.
    if [[ "$f" == "mod.rs" ]]; then
        continue
    fi
    checked=$((checked + 1))
    if ! head -1 "$f" | grep -q "Copyright (c) 2023, Oracle"; then
        echo "MISSING: $f lacks Oracle copyright header"
        fail=1
    fi
done
if [[ "$fail" -eq 0 ]]; then
    echo "OK: all $checked copied files carry the Oracle UPL-1.0 header"
fi
exit $fail
