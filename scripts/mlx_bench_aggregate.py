"""Aggregate genai-bench output into a markdown comparison table.

Walks $RESULTS_DIR for sub-directories matching `{label}_{scenario}_c{n}/`
(label ∈ {mlx, grpc}) and renders one section per scenario. Rows for
backends that produced no JSON are omitted, so a PHASES=mlx run renders
as single-backend rather than padding with `—`.

Latency columns (TTFT / TPOT) are computed over the first N completed
requests per cell, where N = min(completed) across the backends present for
that cell. This makes the two backends comparable on an equal sample size even
when one finished more requests than the other within the backstop. Throughput
columns (RPS, output tok/s) and the `Completed` count remain whole-run values.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
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


@dataclass
class Cell:
    """One benchmarked (label, scenario, concurrency) cell."""

    agg: dict[str, Any]
    # Completed (error-free) per-request metrics, in genai-bench's order.
    completed: list[dict[str, Any]]
    # True when the cell produced no result JSON (0 completed within backstop).
    empty: bool = False


def _find_result_json(folder: Path) -> Path | None:
    for p in sorted(folder.rglob("*.json")):
        if "experiment_metadata" in p.name or "gpu_utilization" in p.name:
            continue
        return p
    return None


def _read_cell(folder: Path) -> Cell | None:
    """Return a Cell for a run dir, or None if the dir isn't a benchmark cell.

    A dir that ran but completed nothing (only experiment_metadata.json, or a
    `.empty` / `.failed` marker) yields an empty Cell so the row is rendered
    rather than silently dropped.
    """
    j = _find_result_json(folder)
    if j is None:
        if (folder / ".empty").exists() or (folder / ".failed").exists():
            return Cell(agg={}, completed=[], empty=True)
        return None
    try:
        data = json.loads(j.read_text())
    except json.JSONDecodeError:
        return None
    agg = data.get("aggregated_metrics", {})
    completed = [r for r in data.get("individual_request_metrics", []) if not r.get("error_code")]
    return Cell(agg=agg, completed=completed, empty=not completed)


def collect(results_dir: Path) -> dict[tuple[str, str, int], Cell]:
    out: dict[tuple[str, str, int], Cell] = {}
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
        cell = _read_cell(sub)
        if cell is None:
            continue
        out[(m.group("label"), m.group("scenario"), int(m.group("concurrency")))] = cell
    return out


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _vals(reqs: list[dict[str, Any]], key: str) -> list[float]:
    return [r[key] for r in reqs if isinstance(r.get(key), (int, float))]


def fmt_float(v: float | None, places: int = 2) -> str:
    return "—" if v is None else f"{v:.{places}f}"


def fmt_ms(v_seconds: float | None) -> str:
    """genai-bench latency stats are in seconds; render as ms."""
    return "—" if v_seconds is None else f"{v_seconds * 1000:.0f}"


def build_section(
    results: dict[tuple[str, str, int], Cell],
    scenario: str,
    concurrencies: list[int],
    labels: list[str],
) -> list[str]:
    lines = [
        "",
        f"### Scenario `{scenario}`",
        "",
        "| Concurrency | Backend | Completed | N (compared) | RPS | Output tok/s "
        "| TTFT mean (ms) | TTFT p99 (ms) | TPOT mean (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in concurrencies:
        # Equal-N: compare the backends present for this cell on the smaller of
        # their completed counts.
        present = [
            results[(label, scenario, c)] for label in labels if (label, scenario, c) in results
        ]
        completed_counts = [len(cell.completed) for cell in present if cell.completed]
        n = min(completed_counts) if completed_counts else 0

        for label in labels:
            pretty = LABEL_PRETTY[label]
            cell = results.get((label, scenario, c))
            if cell is None:
                lines.append(f"| {c} | {pretty} | — | — | — | — | — | — | — |")
                continue
            if cell.empty or not cell.completed:
                err = cell.agg.get("error_codes_frequency") or {}
                note = (
                    f"0 ({', '.join(f'{k}×{v}' for k, v in err.items())})"
                    if err
                    else "0 (timed out)"
                )
                lines.append(f"| {c} | {pretty} | {note} | — | 0.00 | 0.0 | — | — | — |")
                continue

            sample = cell.completed[:n] if n else cell.completed
            ttft = _vals(sample, "ttft")
            tpot = _vals(sample, "tpot")
            lines.append(
                f"| {c} | {pretty} | "
                f"{cell.agg.get('num_completed_requests', len(cell.completed))} | "
                f"{n} | "
                f"{fmt_float(cell.agg.get('requests_per_second'))} | "
                f"{fmt_float(cell.agg.get('mean_output_throughput_tokens_per_s'), 1)} | "
                f"{fmt_ms(_mean(ttft))} | "
                f"{fmt_ms(_percentile(ttft, 99))} | "
                f"{fmt_ms(_mean(tpot))} |"
            )
    lines.extend(
        [
            "",
            "> `N (compared)` = min completed across backends for the cell; TTFT/TPOT "
            "are computed over each backend's first N completed requests so the latency "
            "comparison uses an equal sample size. RPS, output tok/s and `Completed` are "
            "whole-run values.",
            "",
        ]
    )
    return lines


def build_table(results: dict[tuple[str, str, int], Cell], model: str) -> str:
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
        f"**Concurrencies:** {', '.join(str(c) for c in concurrencies)} "
        "(runs are request-bounded — each cell targets a fixed number of completed requests)",
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
