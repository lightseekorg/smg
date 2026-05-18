"""Aggregate genai-bench output into a markdown comparison table.

Walks $RESULTS_DIR for sub-directories matching `{label}_{scenario}_c{n}/`
(label ∈ {mlx, grpc}) and renders one section per scenario. Rows for
backends that produced no JSON are omitted, so a PHASES=mlx run renders
as single-backend rather than padding with `—`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DIRNAME_RE = re.compile(r"^(?P<label>mlx|grpc)_(?P<scenario>[^_]+)_c(?P<concurrency>\d+)$")

LABEL_ORDER: tuple[str, ...] = ("mlx", "grpc")
LABEL_PRETTY = {"mlx": "mlx-lm.server", "grpc": "smg → mlx-grpc"}
LABEL_DESCRIPTION = {
    "mlx": "`mlx-lm.server` — direct HTTP (mlx-lm package)",
    "grpc": "`smg → mlx-grpc` — SMG router fronting our MLX gRPC servicer (PR #1099)",
}
WAY_NAME = {1: "Single-backend", 2: "Two-Way"}


def _find_result_json(folder: Path) -> Path | None:
    for p in folder.rglob("*.json"):
        if "experiment_metadata" in p.name or "gpu_utilization" in p.name:
            continue
        return p
    return None


def _read(folder: Path) -> dict[str, Any] | None:
    j = _find_result_json(folder)
    if j is None:
        return None
    try:
        return json.loads(j.read_text())
    except json.JSONDecodeError:
        return None


def collect(results_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    # Aggregate runs under `if: always()`; return empty instead of crashing
    # when no phase produced output.
    if not results_dir.is_dir():
        return out
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = DIRNAME_RE.match(sub.name)
        if not m:
            continue
        data = _read(sub)
        if data is None:
            continue
        out[(m.group("label"), m.group("scenario"), int(m.group("concurrency")))] = data
    return out


def _stat(d: dict[str, Any], path: str) -> float | None:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def fmt_float(v: float | None, places: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{places}f}"


def fmt_ms(v_seconds: float | None) -> str:
    """genai-bench latency stats are in seconds; render as ms."""
    if v_seconds is None:
        return "—"
    return f"{v_seconds * 1000:.0f}"


def build_section(
    results: dict[tuple[str, str, int], dict[str, Any]],
    scenario: str,
    concurrencies: list[int],
    labels: list[str],
) -> list[str]:
    lines = [
        "",
        f"### Scenario `{scenario}`",
        "",
        "| Concurrency | Backend | RPS | Output tok/s | TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) | Completed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in concurrencies:
        for label in labels:
            r = results.get((label, scenario, c))
            pretty = LABEL_PRETTY[label]
            if r is None:
                lines.append(f"| {c} | {pretty} | — | — | — | — | — | — |")
                continue
            agg = r.get("aggregated_metrics", {})
            lines.append(
                f"| {c} | {pretty} | "
                f"{fmt_float(agg.get('requests_per_second'))} | "
                f"{fmt_float(agg.get('mean_output_throughput_tokens_per_s'), 1)} | "
                f"{fmt_ms(_stat(agg, 'stats.ttft.mean'))} | "
                f"{fmt_ms(_stat(agg, 'stats.ttft.p99'))} | "
                f"{fmt_ms(_stat(agg, 'stats.tpot.mean'))} | "
                f"{agg.get('num_completed_requests', '?')} |"
            )
    lines.append("")
    return lines


def build_table(results: dict[tuple[str, str, int], dict[str, Any]], model: str) -> str:
    if not results:
        return "_No results found._"
    present = {k[0] for k in results}
    labels = [label for label in LABEL_ORDER if label in present]
    scenarios = sorted({k[1] for k in results})
    concurrencies = sorted({k[2] for k in results})
    way = WAY_NAME.get(len(labels), f"{len(labels)}-way")
    lines = [
        f"## MLX {way} Benchmark",
        "",
        f"**Model:** `{model}`",
        "",
        f"{len(labels)} backend{'s' if len(labels) != 1 else ''} serving the same MLX "
        "model on Apple Silicon, driven by `genai-bench` against synthetic "
        "deterministic scenarios:",
        "",
        "- `chat` = `D(100,256)` — short prompt + medium output (typical chat turn)",
        "- `agent` = `D(2500,256)` — ~2.5k token context + medium output "
        "(RAG / code-edit / Cursor-style local agent traffic)",
        "",
        f"**Concurrencies:** {', '.join(str(c) for c in concurrencies)}",
        "",
        "Backends:",
    ]
    lines.extend(f"- {LABEL_DESCRIPTION[label]}" for label in labels)
    for s in scenarios:
        lines.extend(build_section(results, s, concurrencies, labels))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="bench-results")
    p.add_argument("--model", default="mlx-community/gemma-3-4b-it-qat-4bit")
    p.add_argument("--out", default="bench-results/SUMMARY.md")
    args = p.parse_args()

    results = collect(Path(args.results_dir))
    table = build_table(results, args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(table)


if __name__ == "__main__":
    main()
