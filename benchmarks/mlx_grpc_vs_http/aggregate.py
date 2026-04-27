"""Build a markdown comparison table from genai-bench output JSON.

Reads the experiment folders produced by run.sh:
    bench-results/{label}_{scenario}_c{concurrency}/.../*.json

Pulls the aggregated metrics (ttft, e2e_latency, input_throughput,
output_throughput) from each and emits a side-by-side http vs grpc
markdown table per traffic scenario.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Directory name format from run.sh: e.g. "http_D-100-256_c16"
DIRNAME_RE = re.compile(r"^(?P<label>http|grpc)_(?P<scenario>.+?)_c(?P<concurrency>\d+)$")


def _stats(d: dict[str, Any], key: str) -> dict[str, float]:
    return d.get("aggregated_metrics", {}).get("stats", {}).get(key, {})


def parse_experiment(folder: Path) -> dict[str, Any] | None:
    """Pull metrics out of a genai-bench experiment folder."""
    json_files = sorted(
        p
        for p in folder.rglob("*.json")
        if "experiment_metadata" not in p.name and "gpu_utilization" not in p.name
    )
    if not json_files:
        return None
    # Each cell produces one summary JSON; sorted() makes the choice
    # deterministic if a partial rerun left a stale file.
    data = json.loads(json_files[0].read_text())
    return {
        "ttft_mean": float(_stats(data, "ttft").get("mean", float("inf"))),
        "ttft_p90": float(_stats(data, "ttft").get("p90", float("inf"))),
        "e2e_mean": float(_stats(data, "e2e_latency").get("mean", float("inf"))),
        "input_tps": float(_stats(data, "input_throughput").get("mean", 0.0)),
        "output_tps": float(_stats(data, "output_throughput").get("mean", 0.0)),
        "rps": float(_stats(data, "request_throughput").get("mean", 0.0)),
    }


def collect(results_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = DIRNAME_RE.match(sub.name)
        if not m:
            continue
        metrics = parse_experiment(sub)
        if metrics is None:
            continue
        label = m.group("label")
        scenario = m.group("scenario")
        concurrency = int(m.group("concurrency"))
        out[(label, scenario, concurrency)] = metrics
    return out


def fmt_ms(seconds: float) -> str:
    if seconds == float("inf"):
        return "—"
    return f"{seconds * 1000:.0f}"


def fmt_float(x: float, places: int = 1) -> str:
    if x in (0.0, float("inf")):
        return "—"
    return f"{x:.{places}f}"


def build_section(
    results: dict[tuple[str, str, int], dict[str, Any]],
    scenario: str,
    concurrencies: list[int],
) -> list[str]:
    lines = [
        "",
        f"### Scenario `{scenario}`",
        "",
        "| Concurrency | Setup | RPS | TTFT mean (ms) | TTFT p90 (ms) | E2E mean (ms) | Output tok/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in concurrencies:
        for label in ("http", "grpc"):
            r = results.get((label, scenario, c))
            if r is None:
                lines.append(f"| {c} | {label} | — | — | — | — | — |")
                continue
            lines.append(
                f"| {c} | {label} | {fmt_float(r['rps'], 3)} | "
                f"{fmt_ms(r['ttft_mean'])} | {fmt_ms(r['ttft_p90'])} | "
                f"{fmt_ms(r['e2e_mean'])} | {fmt_float(r['output_tps'])} |"
            )
    lines.append("")
    return lines


def build_table(
    results: dict[tuple[str, str, int], dict[str, Any]],
    model: str,
) -> str:
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
        "E2E = end-to-end request latency. Lower is better for latencies; "
        "higher is better for throughput.",
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
