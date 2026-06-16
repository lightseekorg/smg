#!/usr/bin/env python3
"""Register a model that ``bfcl-eval`` doesn't ship a handler for yet.

The BFCL leaderboard package pins a fixed ``MODEL_CONFIG_MAPPING``; brand-new
models (e.g. ``Qwen/Qwen3.6-27B``, released after the package was cut) aren't in
it, so ``bfcl generate --model <id>-FC`` fails with "Unknown model_name". For an
A/B that only needs FC mode against a self-hosted OpenAI endpoint, the handler
logic is identical to any other Qwen FC model (send native ``tools``, read
``tool_calls``), so we clone an existing FC entry under the new id.

This edits the *installed* ``bfcl_eval/constants/model_config.py`` in place
(idempotent). Re-running is safe. Intended for nightly "test the latest models"
flows where the bfcl release lags new releases.

    python register_bfcl_model.py --model-id Qwen/Qwen3.6-27B
    # registers "Qwen/Qwen3.6-27B-FC" cloned from "Qwen/Qwen3-32B-FC"
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

DEFAULT_ANCHOR = '    "Qwen/Qwen3-32B-FC": ModelConfig('


def find_model_config() -> Path:
    spec = importlib.util.find_spec("bfcl_eval.constants.model_config")
    if spec is None or spec.origin is None:
        raise SystemExit("bfcl_eval not importable in this interpreter")
    return Path(spec.origin)


def build_entry(model_id: str, handler: str) -> str:
    return (
        f'    "{model_id}-FC": ModelConfig(\n'
        f'        model_name="{model_id}",\n'
        f'        display_name="{model_id.split("/")[-1]} (FC)",\n'
        f'        url="https://huggingface.co/{model_id}",\n'
        f'        org="{model_id.split("/")[0]}",\n'
        f'        license="apache-2.0",\n'
        f"        model_handler={handler},\n"
        f"        input_price=None,\n"
        f"        output_price=None,\n"
        f"        is_fc_model=True,\n"
        f"        underscore_to_dot=False,\n"
        f"    ),\n"
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-id", required=True, help="HF id, e.g. Qwen/Qwen3.6-27B")
    p.add_argument(
        "--handler",
        default="QwenFCHandler",
        help="bfcl handler class already imported in model_config.py",
    )
    p.add_argument("--anchor", default=DEFAULT_ANCHOR, help="existing entry line to insert before")
    args = p.parse_args()

    path = find_model_config()
    src = path.read_text(encoding="utf-8")
    key = f'"{args.model_id}-FC":'
    if key in src:
        print(f"already registered: {args.model_id}-FC")
        return 0
    if args.anchor not in src:
        raise SystemExit(
            f"anchor not found in {path}; pass a valid --anchor (an existing entry line)"
        )
    if args.handler not in src:
        raise SystemExit(
            f"handler {args.handler} is not referenced in {path}; pick one that is imported there"
        )

    entry = build_entry(args.model_id, args.handler)
    src = src.replace(args.anchor, entry + args.anchor, 1)
    path.write_text(src, encoding="utf-8")
    print(f"registered {args.model_id}-FC (handler={args.handler}) in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
