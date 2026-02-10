#!/usr/bin/env bash
# Checks for deeply nested internal imports (5+ path segments) in Rust files.
# Only flags crate:: and super:: paths (external crate imports are ignored).
set -euo pipefail

MAX_SEGMENTS=4
violations=0

for file in "$@"; do
    [[ -f "$file" ]] || continue

    # Match: use crate::a::b::c::d  (5+ segments starting with crate/super)
    # {4,} means 4+ additional ::word segments after crate/super = 5+ total
    while IFS= read -r line_info; do
        line_num="${line_info%%:*}"
        line_content="${line_info#*:}"
        echo "  $file:$line_num:$line_content"
        ((violations++))
    done < <(grep -nE '^\s*use\s+(crate|super)(::\w+){4,}' "$file" 2>/dev/null || true)
done

if [[ $violations -gt 0 ]]; then
    echo ""
    echo "Found $violations import(s) exceeding $MAX_SEGMENTS path segments."
    echo "Consider using shorter paths or re-exports to reduce import depth."
    exit 1
fi
