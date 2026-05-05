#!/usr/bin/env python3
"""Regenerate expected/*.txt from schemas/*.json using Kimi-K2.5's reference Python encoder.

Run by hand any time the reference encoder changes:
    python3 crates/tokenizer/tests/fixtures/kimi_k25/regenerate.py

Requires the Kimi-K2.5 model snapshot on disk (default path below — override via
$KIMI_K25_DIR). Not run in CI.
"""

import json
import os
import sys
from pathlib import Path

DEFAULT_KIMI_DIR = "/raid/models/hub/models--moonshotai--Kimi-K2.5/snapshots"
KIMI_DIR = os.environ.get("KIMI_K25_DIR")
if not KIMI_DIR:
    snapshots = Path(DEFAULT_KIMI_DIR)
    if not snapshots.exists():
        sys.exit(
            f"set $KIMI_K25_DIR to a Kimi-K2.5 snapshot dir (no default at {DEFAULT_KIMI_DIR})"
        )
    KIMI_DIR = str(next(snapshots.iterdir()))

sys.path.insert(0, KIMI_DIR)
from tool_declaration_ts import encode_tools_to_typescript_style  # type: ignore  # noqa: E402

here = Path(__file__).parent
schemas_dir = here / "schemas"
expected_dir = here / "expected"
expected_dir.mkdir(exist_ok=True)

for schema_path in sorted(schemas_dir.glob("*.json")):
    tools = json.loads(schema_path.read_text())
    output = encode_tools_to_typescript_style(tools)
    out_path = expected_dir / f"{schema_path.stem}.txt"
    out_path.write_text(output)
    print(f"wrote {out_path.relative_to(here.parent.parent.parent)} ({len(output)} bytes)")
