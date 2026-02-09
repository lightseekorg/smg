#!/usr/bin/env python3
"""Generate nightly benchmark summary for GitHub Actions.

Produces a gRPC vs HTTP comparison report with aggregate stats, per-concurrency
breakdown, win/loss scorecard, top wins, and per-model detail tables.

Usage:
    python nightly_summarize.py [base_dir]
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Single benchmark run result."""

    scenario: str
    concurrency: int
    rps: float
    output_throughput: float
    total_throughput: float
    ttft_mean: float
    ttft_p99: float
    tpot_mean: float
    tpot_p99: float
    e2e_mean: float
    e2e_p99: float
    error_rate: float


@dataclass
class ExperimentInfo:
    """Parsed experiment metadata."""

    model: str  # short name (e.g. Llama-3.1-8B-Instruct)
    protocol: str  # http, grpc
    runtime: str  # sglang, vllm
    worker_type: str  # single, multi
    gpu_type: str
    gpu_count: int
    runs: list[RunResult] = field(default_factory=list)

    @property
    def group_key(self) -> str:
        """Key for grouping gRPC vs HTTP pairs."""
        return f"{self.model}|{self.runtime}|{self.worker_type}"

    @property
    def table_key(self) -> str:
        """Key for overview table columns."""
        return f"{self.protocol}_{self.runtime}_{self.worker_type}"


@dataclass
class ComparisonPoint:
    """A matched gRPC vs HTTP data point."""

    model: str
    runtime: str
    worker_type: str
    scenario: str
    concurrency: int
    grpc: RunResult
    http: RunResult


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _get_float(d: dict, key: str, default: float = 0.0) -> float:
    val = d.get(key)
    return float(val) if val is not None else default


_KNOWN_PROTOCOLS = {"http", "grpc"}
_KNOWN_WORKER_TYPES = {"single", "multi"}
# Runtimes recognized in folder names. Add new runtimes here.
_KNOWN_RUNTIMES = {"sglang", "vllm", "trtllm"}


def parse_folder_name(folder_name: str) -> dict:
    """Parse experiment info from folder name.

    Expected: nightly_{model}_{protocol}_{runtime}_{worker_type}
    """
    info = {
        "model": "unknown",
        "protocol": "unknown",
        "runtime": None,
        "worker_type": "single",
    }
    name = folder_name.replace("nightly_", "")
    parts = name.rsplit("_", 3)

    if len(parts) >= 4 and parts[-1] in _KNOWN_WORKER_TYPES and parts[-2] in _KNOWN_RUNTIMES:
        info["worker_type"] = parts[-1]
        info["runtime"] = parts[-2]
        info["protocol"] = parts[-3]
        info["model"] = "_".join(parts[:-3])
    elif len(parts) >= 3 and parts[-1] in _KNOWN_RUNTIMES:
        info["runtime"] = parts[-1]
        info["protocol"] = parts[-2]
        info["model"] = "_".join(parts[:-2])
    elif len(parts) >= 2 and parts[-1] in _KNOWN_PROTOCOLS:
        info["protocol"] = parts[-1]
        info["model"] = "_".join(parts[:-1])
    else:
        info["model"] = name

    return info


def parse_experiment(folder: Path) -> ExperimentInfo | None:
    """Parse experiment folder into ExperimentInfo."""
    metadata_path = folder / "experiment_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open() as f:
            meta = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse metadata in {folder}: {e}", file=sys.stderr)
        return None

    folder_info = parse_folder_name(folder.name)

    model_path = meta.get("model", "unknown")
    model = model_path.split("/")[-1] if "/" in model_path else model_path

    runtime = meta.get("server_engine")
    if not runtime or runtime == "unknown":
        runtime = folder_info.get("runtime")
    if not runtime:
        # Fallback: detect runtime from folder name
        folder_lower = folder.name.lower()
        for rt in _KNOWN_RUNTIMES:
            if rt in folder_lower:
                runtime = rt
                break
        else:
            runtime = "unknown"
    runtime = runtime.lower() if runtime else "unknown"

    worker_type = folder_info.get("worker_type", "single")
    gpu_type = meta.get("server_gpu_type") or "unknown"
    try:
        gpu_count = int(meta.get("server_gpu_count") or "1")
    except (ValueError, TypeError):
        gpu_count = 1

    info = ExperimentInfo(
        model=model,
        protocol=folder_info.get("protocol", "unknown"),
        runtime=runtime,
        worker_type=worker_type,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
    )

    for json_file in folder.glob("*.json"):
        if "experiment_metadata" in json_file.name or "gpu_utilization" in json_file.name:
            continue

        try:
            with json_file.open() as f:
                data = json.load(f)

            agg = data.get("aggregated_metrics", {})
            stats = agg.get("stats", {})
            ttft = stats.get("ttft", {})
            tpot = stats.get("tpot", {})
            e2e = stats.get("e2e_latency", {})

            run = RunResult(
                scenario=agg.get("scenario", "unknown"),
                concurrency=agg.get("num_concurrency", 0) or 0,
                rps=_get_float(agg, "requests_per_second"),
                output_throughput=_get_float(agg, "mean_output_throughput_tokens_per_s"),
                total_throughput=_get_float(agg, "mean_total_tokens_throughput_tokens_per_s"),
                ttft_mean=_get_float(ttft, "mean"),
                ttft_p99=_get_float(ttft, "p99"),
                tpot_mean=_get_float(tpot, "mean"),
                tpot_p99=_get_float(tpot, "p99"),
                e2e_mean=_get_float(e2e, "mean"),
                e2e_p99=_get_float(e2e, "p99"),
                error_rate=_get_float(agg, "error_rate"),
            )
            info.runs.append(run)
        except Exception as e:
            print(f"Warning: Failed to parse {json_file}: {e}", file=sys.stderr)

    return info if info.runs else None


def discover_experiments(base_dir: Path) -> list[ExperimentInfo]:
    """Discover and parse all nightly experiment folders."""
    experiments = []
    for folder in base_dir.rglob("nightly_*"):
        if folder.is_dir():
            exp = parse_experiment(folder)
            if exp:
                experiments.append(exp)
    return experiments


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def build_comparisons(experiments: list[ExperimentInfo]) -> list[ComparisonPoint]:
    """Match gRPC and HTTP runs for the same model/runtime/worker/scenario/concurrency."""
    groups: dict[str, dict[str, ExperimentInfo]] = defaultdict(dict)
    for exp in experiments:
        groups[exp.group_key][exp.protocol] = exp

    comparisons = []
    for protocols in groups.values():
        if "grpc" not in protocols or "http" not in protocols:
            continue

        grpc_exp = protocols["grpc"]
        http_exp = protocols["http"]

        http_runs = {(r.scenario, r.concurrency): r for r in http_exp.runs}

        for grpc_run in grpc_exp.runs:
            http_run = http_runs.get((grpc_run.scenario, grpc_run.concurrency))
            if http_run and grpc_run.error_rate == 0 and http_run.error_rate == 0:
                comparisons.append(
                    ComparisonPoint(
                        model=grpc_exp.model,
                        runtime=grpc_exp.runtime,
                        worker_type=grpc_exp.worker_type,
                        scenario=grpc_run.scenario,
                        concurrency=grpc_run.concurrency,
                        grpc=grpc_run,
                        http=http_run,
                    )
                )

    return comparisons


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _pct(grpc_val: float, http_val: float) -> float | None:
    """(gRPC - HTTP) / HTTP * 100."""
    if http_val == 0:
        return None
    return (grpc_val - http_val) / http_val * 100


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "N/A"
    return f"{pct:+.1f}%"


def _fmt_latency_s(val_s: float) -> str:
    """Format latency value (in seconds) for display."""
    ms = val_s * 1000
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{val_s:.2f}s"


def _fmt_throughput(val: float) -> str:
    if val >= 1000:
        return f"{val / 1000:.1f}K"
    return f"{val:.0f}"


def _winner(pct: float | None, lower_is_better: bool, threshold: float = 2.0) -> str:
    if pct is None:
        return ""
    if abs(pct) < threshold:
        return "~Tie"
    if lower_is_better:
        return "**gRPC**" if pct < 0 else "HTTP"
    return "**gRPC**" if pct > 0 else "HTTP"


def _safe_avg(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------


_RUNTIME_DISPLAY = {"sglang": "SGLang", "vllm": "vLLM", "trtllm": "TRT-LLM"}
_PROTOCOL_DISPLAY = {"http": "HTTP", "grpc": "gRPC"}


def _section_overview(experiments: list[ExperimentInfo]) -> list[str]:
    """Overview table with status per model/config — columns discovered dynamically."""
    by_model: dict[str, dict[str, ExperimentInfo]] = defaultdict(dict)
    for exp in experiments:
        by_model[exp.model][exp.table_key] = exp

    # Discover all (protocol, runtime, worker_type) combos that actually exist
    all_keys: set[str] = set()
    for model_exps in by_model.values():
        all_keys.update(model_exps.keys())

    # Sort: single before multi, then by runtime name, then http before grpc
    _worker_order = {"single": 0, "multi": 1}
    _protocol_order = {"http": 0, "grpc": 1}

    def _col_sort_key(key: str) -> tuple:
        protocol, runtime, worker = key.split("_")
        return (_worker_order.get(worker, 9), runtime, _protocol_order.get(protocol, 9))

    table_order = []
    for key in sorted(all_keys, key=_col_sort_key):
        protocol, runtime, worker = key.split("_")
        p_disp = _PROTOCOL_DISPLAY.get(protocol, protocol.upper())
        r_disp = _RUNTIME_DISPLAY.get(runtime, runtime)
        w_disp = worker.capitalize()
        table_order.append((key, f"{p_disp} {r_disp} {w_disp}"))

    header_cols = ["Model"] + [title for _, title in table_order]
    lines = [
        "### Overview",
        "",
        "| " + " | ".join(header_cols) + " |",
        "|" + "|".join(["---"] * len(header_cols)) + "|",
    ]

    for model in sorted(by_model.keys()):
        model_exps = by_model[model]
        row = [model]
        for table_key, _ in table_order:
            if table_key not in model_exps:
                row.append("\u2796")
            else:
                exp = model_exps[table_key]
                has_errors = any(r.rps == 0 or r.output_throughput == 0 for r in exp.runs)
                row.append("\u26a0\ufe0f" if has_errors else "\u2705")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return lines


def _section_aggregate(comparisons: list[ComparisonPoint]) -> list[str]:
    """Aggregate gRPC vs HTTP comparison table."""
    if not comparisons:
        return ["*No gRPC vs HTTP comparison data available.*", ""]

    metrics = [
        ("TTFT mean", "ttft_mean", True),
        ("TTFT p99", "ttft_p99", True),
        ("E2E mean", "e2e_mean", True),
        ("E2E p99", "e2e_p99", True),
        ("TPOT mean", "tpot_mean", True),
        ("Output throughput", "output_throughput", False),
        ("Total throughput", "total_throughput", False),
        ("RPS", "rps", False),
    ]

    lines = [
        "### Aggregate: gRPC vs HTTP",
        "",
        f"*{len(comparisons)} matched data points (error-free pairs only). "
        "Negative % = gRPC lower (better for latency). "
        "Positive % = gRPC higher (better for throughput).*",
        "",
        "| Metric | Avg % | Median % | Winner |",
        "|--------|------:|--------:|--------|",
    ]

    for label, fld, lower_better in metrics:
        diffs = []
        for cp in comparisons:
            p = _pct(getattr(cp.grpc, fld), getattr(cp.http, fld))
            if p is not None:
                diffs.append(p)
        if not diffs:
            continue
        avg = sum(diffs) / len(diffs)
        med = median(diffs)
        lines.append(
            f"| {label} | {_fmt_pct(avg)} | {_fmt_pct(med)} | {_winner(avg, lower_better)} |"
        )

    lines.append("")
    return lines


def _section_by_concurrency(comparisons: list[ComparisonPoint]) -> list[str]:
    """Per-concurrency breakdown for TTFT, E2E, and throughput."""
    if not comparisons:
        return []

    by_conc: dict[int, list[ComparisonPoint]] = defaultdict(list)
    for cp in comparisons:
        by_conc[cp.concurrency].append(cp)
    conc_levels = sorted(by_conc.keys())

    lines = ["### Performance by Concurrency", ""]

    # TTFT mean
    lines.extend(
        [
            "<details>",
            "<summary><b>TTFT Mean by Concurrency</b></summary>",
            "",
            "| Concurrency | gRPC avg | HTTP avg | Diff % | Winner |",
            "|---:|---:|---:|---:|:---|",
        ]
    )
    for conc in conc_levels:
        cps = by_conc[conc]
        g_avg = sum(cp.grpc.ttft_mean * 1000 for cp in cps) / len(cps)
        h_avg = sum(cp.http.ttft_mean * 1000 for cp in cps) / len(cps)
        pct = _pct(g_avg, h_avg)
        lines.append(
            f"| {conc} | {g_avg:.0f}ms | {h_avg:.0f}ms | {_fmt_pct(pct)} | "
            f"{_winner(pct, lower_is_better=True)} |"
        )
    lines.extend(["", "</details>", ""])

    # E2E mean
    lines.extend(
        [
            "<details>",
            "<summary><b>E2E Latency Mean by Concurrency</b></summary>",
            "",
            "| Concurrency | gRPC avg | HTTP avg | Diff % | Winner |",
            "|---:|---:|---:|---:|:---|",
        ]
    )
    for conc in conc_levels:
        cps = by_conc[conc]
        g_avg = sum(cp.grpc.e2e_mean * 1000 for cp in cps) / len(cps)
        h_avg = sum(cp.http.e2e_mean * 1000 for cp in cps) / len(cps)
        pct = _pct(g_avg, h_avg)
        lines.append(
            f"| {conc} | {g_avg:.0f}ms | {h_avg:.0f}ms | {_fmt_pct(pct)} | "
            f"{_winner(pct, lower_is_better=True)} |"
        )
    lines.extend(["", "</details>", ""])

    # Output throughput
    lines.extend(
        [
            "<details>",
            "<summary><b>Output Throughput by Concurrency</b></summary>",
            "",
            "| Concurrency | gRPC avg | HTTP avg | Diff % | Winner |",
            "|---:|---:|---:|---:|:---|",
        ]
    )
    for conc in conc_levels:
        cps = by_conc[conc]
        g_avg = sum(cp.grpc.output_throughput for cp in cps) / len(cps)
        h_avg = sum(cp.http.output_throughput for cp in cps) / len(cps)
        pct = _pct(g_avg, h_avg)
        lines.append(
            f"| {conc} | {_fmt_throughput(g_avg)} tok/s | "
            f"{_fmt_throughput(h_avg)} tok/s | {_fmt_pct(pct)} | "
            f"{_winner(pct, lower_is_better=False)} |"
        )
    lines.extend(["", "</details>", ""])

    return lines


def _section_scorecard(comparisons: list[ComparisonPoint]) -> list[str]:
    """Win/loss scorecard at different thresholds."""
    if not comparisons:
        return []

    lines = [
        "### Win/Loss Scorecard",
        "",
        "*How often does gRPC beat HTTP beyond a given threshold?*",
        "",
    ]

    metrics = [
        ("E2E mean", "e2e_mean", True),
        ("TTFT mean", "ttft_mean", True),
        ("Output throughput", "output_throughput", False),
    ]
    thresholds = [1, 2, 5, 10]

    for label, fld, lower_better in metrics:
        lines.extend(
            [
                f"**{label}:**",
                "",
                "| Threshold | gRPC wins | HTTP wins | Within |",
                "|---:|---:|---:|---:|",
            ]
        )

        for thresh in thresholds:
            grpc_w = http_w = within = 0
            for cp in comparisons:
                pct = _pct(getattr(cp.grpc, fld), getattr(cp.http, fld))
                if pct is None:
                    continue
                if lower_better:
                    if pct < -thresh:
                        grpc_w += 1
                    elif pct > thresh:
                        http_w += 1
                    else:
                        within += 1
                else:
                    if pct > thresh:
                        grpc_w += 1
                    elif pct < -thresh:
                        http_w += 1
                    else:
                        within += 1

            g_str = f"**{grpc_w}**" if grpc_w > http_w else str(grpc_w)
            h_str = f"**{http_w}**" if http_w > grpc_w else str(http_w)
            lines.append(f"| >{thresh}% | {g_str} | {h_str} | {within} |")

        lines.append("")

    return lines


def _section_top_wins(comparisons: list[ComparisonPoint], top_n: int = 15) -> list[str]:
    """Table of largest gRPC wins (>10% improvement)."""
    if not comparisons:
        return []

    metric_defs = [
        ("TTFT p99", "ttft_p99", True),
        ("TTFT mean", "ttft_mean", True),
        ("E2E mean", "e2e_mean", True),
        ("E2E p99", "e2e_p99", True),
        ("Output tput", "output_throughput", False),
        ("RPS", "rps", False),
    ]

    wins: list[tuple[float, float, str, ComparisonPoint, float, float]] = []
    for cp in comparisons:
        for label, fld, lower_better in metric_defs:
            g = getattr(cp.grpc, fld)
            h = getattr(cp.http, fld)
            pct = _pct(g, h)
            if pct is None:
                continue
            if lower_better and pct < -10:
                wins.append((abs(pct), pct, label, cp, g, h))
            elif not lower_better and pct > 10:
                wins.append((abs(pct), pct, label, cp, g, h))

    wins.sort(key=lambda x: x[0], reverse=True)
    wins = wins[:top_n]

    if not wins:
        return []

    lines = [
        "<details>",
        "<summary><b>Top gRPC Wins (&gt;10% improvement)</b></summary>",
        "",
        "| Diff | Model | Config | Scenario | C | Metric | gRPC | HTTP |",
        "|-----:|-------|--------|----------|--:|--------|-----:|-----:|",
    ]

    for _, pct, label, cp, g, h in wins:
        config = f"{cp.runtime}/{cp.worker_type}"
        is_latency = any(k in label.lower() for k in ("ttft", "e2e", "tpot"))
        if is_latency:
            g_str = _fmt_latency_s(g)
            h_str = _fmt_latency_s(h)
        elif "tput" in label.lower():
            g_str = f"{_fmt_throughput(g)} tok/s"
            h_str = f"{_fmt_throughput(h)} tok/s"
        else:
            g_str = f"{g:.1f}"
            h_str = f"{h:.1f}"
        lines.append(
            f"| {_fmt_pct(pct)} | {cp.model} | {config} | "
            f"`{cp.scenario}` | {cp.concurrency} | {label} | {g_str} | {h_str} |"
        )

    lines.extend(["", "</details>", ""])
    return lines


def _section_per_model(comparisons: list[ComparisonPoint]) -> list[str]:
    """Per-model summary table plus collapsible detail tables."""
    if not comparisons:
        return []

    by_model: dict[str, list[ComparisonPoint]] = defaultdict(list)
    for cp in comparisons:
        by_model[cp.model].append(cp)

    lines = ["### Per-Model Summary", ""]

    # Compact summary: one row per model
    lines.extend(
        [
            "| Model | TTFT mean | TTFT p99 | E2E mean | Output tput | N |",
            "|-------|--------:|--------:|--------:|--------:|---:|",
        ]
    )

    def _avg_or_na(vals: list[float]) -> str:
        return _fmt_pct(sum(vals) / len(vals)) if vals else "N/A"

    for model in sorted(by_model.keys()):
        cps = by_model[model]
        ttft = [p for cp in cps if (p := _pct(cp.grpc.ttft_mean, cp.http.ttft_mean)) is not None]
        ttft_p99 = [p for cp in cps if (p := _pct(cp.grpc.ttft_p99, cp.http.ttft_p99)) is not None]
        e2e = [p for cp in cps if (p := _pct(cp.grpc.e2e_mean, cp.http.e2e_mean)) is not None]
        tput = [
            p
            for cp in cps
            if (p := _pct(cp.grpc.output_throughput, cp.http.output_throughput)) is not None
        ]
        lines.append(
            f"| {model} | {_avg_or_na(ttft)} | {_avg_or_na(ttft_p99)} | "
            f"{_avg_or_na(e2e)} | {_avg_or_na(tput)} | {len(cps)} |"
        )

    lines.append("")

    # Detailed per-model/config tables
    for model in sorted(by_model.keys()):
        cps = by_model[model]
        by_config: dict[str, list[ComparisonPoint]] = defaultdict(list)
        for cp in cps:
            by_config[f"{cp.runtime}/{cp.worker_type}"].append(cp)

        for config in sorted(by_config.keys()):
            config_cps = sorted(by_config[config], key=lambda x: (x.scenario, x.concurrency))
            lines.extend(
                [
                    "<details>",
                    f"<summary><b>{model} ({config})</b> — {len(config_cps)} data points</summary>",
                    "",
                    "| Scenario | C | TTFT mean | TTFT p99 | E2E mean | Tput |",
                    "|----------|--:|--------:|--------:|--------:|--------:|",
                ]
            )

            for cp in config_cps:
                lines.append(
                    f"| `{cp.scenario}` | {cp.concurrency} | "
                    f"{_fmt_pct(_pct(cp.grpc.ttft_mean, cp.http.ttft_mean))} | "
                    f"{_fmt_pct(_pct(cp.grpc.ttft_p99, cp.http.ttft_p99))} | "
                    f"{_fmt_pct(_pct(cp.grpc.e2e_mean, cp.http.e2e_mean))} | "
                    f"{_fmt_pct(_pct(cp.grpc.output_throughput, cp.http.output_throughput))} |"
                )

            lines.extend(["", "</details>", ""])

    return lines


# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------


def generate_summary(base_dir: Path) -> str:
    """Generate the full markdown summary."""
    experiments = discover_experiments(base_dir)

    if not experiments:
        return "## Nightly Benchmark Summary\n\nNo benchmark results found."

    comparisons = build_comparisons(experiments)

    grpc_count = sum(1 for e in experiments if e.protocol == "grpc")
    http_count = sum(1 for e in experiments if e.protocol == "http")
    total_runs = sum(len(e.runs) for e in experiments)

    lines = [
        "## Nightly Benchmark: gRPC vs HTTP",
        "",
        f"> **{len(experiments)} experiments** ({grpc_count} gRPC, {http_count} HTTP), "
        f"**{total_runs} benchmark runs**, "
        f"**{len(comparisons)} matched comparison points**",
        "",
        "---",
        "",
    ]

    lines.extend(_section_overview(experiments))
    lines.extend(_section_aggregate(comparisons))
    lines.extend(_section_by_concurrency(comparisons))
    lines.extend(_section_scorecard(comparisons))
    lines.extend(_section_top_wins(comparisons))
    lines.extend(_section_per_model(comparisons))

    lines.append("---")
    lines.append(
        f"*Generated from {len(experiments)} experiment(s), {len(comparisons)} comparison points*"
    )

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    summary = generate_summary(base_dir)

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(summary)
            f.write("\n")
        print(f"Summary written to {summary_file}")
    else:
        print(summary)


if __name__ == "__main__":
    main()
