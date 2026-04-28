"""Build a markdown comparison table from bench_client.py output.

Reads experiment folders produced by run.sh:
    bench-results/{label}_{scenario}_c{concurrency}/result.json

Emits a side-by-side http vs grpc markdown table per scenario.
Cells with a `.failed` marker (or no result.json) render as "—".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DIRNAME_RE = re.compile(r"^(?P<label>http|grpc)_(?P<scenario>[^_]+)_c(?P<concurrency>\d+)$")


def parse_experiment(folder: Path) -> dict[str, Any] | None:
    result = folder / "result.json"
    if not result.exists():
        return None
    try:
        return json.loads(result.read_text())
    except json.JSONDecodeError:
        return None


def collect(results_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = DIRNAME_RE.match(sub.name)
        if not m:
            continue
        data = parse_experiment(sub)
        if data is None:
            continue
        key = (m.group("label"), m.group("scenario"), int(m.group("concurrency")))
        out[key] = data
    return out


def fmt_ms(stats: dict[str, Any] | None, key: str) -> str:
    if not stats or key not in stats:
        return "—"
    return f"{stats[key]:.0f}"


def fmt_float(x: float, places: int = 2) -> str:
    if x in (0.0, float("inf")):
        return "—"
    return f"{x:.{places}f}"


def build_section(
    results: dict[tuple[str, str, int], dict[str, Any]],
    scenario: str,
    concurrencies: list[int],
) -> list[str]:
    samples = [v for v in results.values() if v["scenario"] == scenario]
    if not samples:
        return [
            "",
            f"### Scenario `{scenario}`",
            "",
            "_No results._",
            "",
        ]
    sample = samples[0]
    title_extra = f" (max_tokens={sample['max_tokens']})"

    lines = [
        "",
        f"### Scenario `{scenario}`{title_extra}",
        "",
        "| Concurrency | Setup | RPS | Output tok/s | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | Errors / Total |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in concurrencies:
        for label in ("http", "grpc"):
            r = results.get((label, scenario, c))
            if r is None:
                lines.append(f"| {c} | {label} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {c} | {label} | {fmt_float(r['request_throughput'], 2)} | "
                f"{fmt_float(r['output_throughput'], 1)} | "
                f"{fmt_ms(r['ttft_ms'], 'p50')} | {fmt_ms(r['ttft_ms'], 'p95')} | "
                f"{fmt_ms(r['tpot_ms'], 'p50')} | "
                f"{r['error_count']} / {r['error_count'] + r['completed_requests']} |"
            )
    lines.append("")
    return lines


def build_table(results: dict[tuple[str, str, int], dict[str, Any]], model: str) -> str:
    if not results:
        return "_No results found in results dir._"
    scenarios = sorted({k[1] for k in results})
    concurrencies = sorted({k[2] for k in results})
    lines = [
        "## MLX Direct HTTP vs Router + gRPC Benchmark",
        "",
        f"**Model:** `{model}`  ",
        f"**Concurrencies:** {', '.join(str(c) for c in concurrencies)}  ",
        f"**Scenarios:** {', '.join(scenarios)}",
        "",
        "RPS = request throughput. TTFT = time to first token. "
        "TPOT = time per output token (excl. first). Lower is better for "
        "latencies; higher is better for throughput. Cells where every "
        "request failed (server overload) render as `—`.",
        "",
        "Scenarios:",
        "- `chat` — ShareGPT (short user prompts, ~50–300 tokens)",
        "- `agent` — vdaita/edit_10k_char (real local-agent code-edit prompts, ~2.5k tokens)",
    ]
    for s in scenarios:
        lines.extend(build_section(results, s, concurrencies))
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
