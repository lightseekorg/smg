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

    if (
        len(parts) >= 4
        and parts[-1] in _KNOWN_WORKER_TYPES
        and parts[-2] in _KNOWN_RUNTIMES
    ):
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
                total_throughput=_get_float(
                    agg, "mean_total_tokens_throughput_tokens_per_s"
                ),
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
#
# All percentages are shown as "gRPC advantage": positive = gRPC is better.
# For latency metrics (lower=better): advantage = (HTTP - gRPC) / HTTP
# For throughput metrics (higher=better): advantage = (gRPC - HTTP) / HTTP
# ---------------------------------------------------------------------------


def _raw_pct(grpc_val: float, http_val: float) -> float | None:
    """Raw (gRPC - HTTP) / HTTP * 100."""
    if http_val == 0:
        return None
    return (grpc_val - http_val) / http_val * 100


def _advantage(grpc_val: float, http_val: float, lower_is_better: bool) -> float | None:
    """gRPC advantage %: positive = gRPC is better, negative = HTTP is better."""
    pct = _raw_pct(grpc_val, http_val)
    if pct is None:
        return None
    return -pct if lower_is_better else pct


def _fmt_winner(pct: float | None, threshold: float = 2.0) -> str:
    """Format as 'gRPC X%' or 'HTTP X%' or '~'. pct is gRPC advantage."""
    if pct is None:
        return "N/A"
    if abs(pct) < threshold:
        return "~"
    if pct > 0:
        return f"gRPC {abs(pct):.1f}%"
    return f"HTTP {abs(pct):.1f}%"


def _fmt_winner_bold(pct: float | None, threshold: float = 2.0) -> str:
    """Like _fmt_winner but bolds the winner name."""
    if pct is None:
        return "N/A"
    if abs(pct) < threshold:
        return "~"
    if pct > 0:
        return f"**gRPC** {abs(pct):.1f}%"
    return f"**HTTP** {abs(pct):.1f}%"


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


# ---------------------------------------------------------------------------
# Chart generation (optional — requires matplotlib)
# ---------------------------------------------------------------------------

# Metrics used across charts and key findings
_CHART_METRICS = [
    ("TTFT p99", "ttft_p99", True, "ms"),
    ("TPOT p99", "tpot_p99", True, "ms"),
    ("E2E p99", "e2e_p99", True, "s"),
    ("Output Throughput", "output_throughput", False, "tok/s"),
]


def generate_charts(
    comparisons: list[ComparisonPoint],
    experiments: list[ExperimentInfo],
    output_dir: Path,
) -> list[str]:
    """Generate comparison charts as PNGs. Returns list of generated filenames."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart generation", file=sys.stderr)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    charts: list[str] = []

    by_conc: dict[int, list[ComparisonPoint]] = defaultdict(list)
    for cp in comparisons:
        by_conc[cp.concurrency].append(cp)
    conc_levels = sorted(by_conc.keys())

    if not conc_levels:
        return []

    # ---- Aggregate comparison (2x2 grid) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("gRPC vs HTTP — Aggregate Comparison", fontsize=16, fontweight="bold")

    for ax, (title, field, _lower_better, unit) in zip(axes.flat, _CHART_METRICS):
        grpc_vals, http_vals = [], []
        for conc in conc_levels:
            cps = by_conc[conc]
            g = sum(getattr(cp.grpc, field) for cp in cps) / len(cps)
            h = sum(getattr(cp.http, field) for cp in cps) / len(cps)
            if unit == "ms":
                g *= 1000
                h *= 1000
            grpc_vals.append(g)
            http_vals.append(h)

        x = range(len(conc_levels))
        ax.plot(x, grpc_vals, "o-", label="gRPC", color="#2196F3", linewidth=2)
        ax.plot(x, http_vals, "s--", label="HTTP", color="#FF9800", linewidth=2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(f"{title} ({unit})")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(c) for c in conc_levels])
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Use log scale when values span >10x
        nz = [v for v in grpc_vals + http_vals if v > 0]
        if nz and max(nz) > 10 * min(nz):
            ax.set_yscale("log")

    fig.tight_layout()
    fname = "aggregate_comparison.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    charts.append(fname)

    # ---- Per-model/config charts ----
    by_model_config: dict[str, list[ComparisonPoint]] = defaultdict(list)
    for cp in comparisons:
        by_model_config[f"{cp.model}|{cp.runtime}/{cp.worker_type}"].append(cp)

    for key, cps in sorted(by_model_config.items()):
        model, config = key.split("|", 1)
        mc_conc: dict[int, list[ComparisonPoint]] = defaultdict(list)
        for cp in cps:
            mc_conc[cp.concurrency].append(cp)
        mc_levels = sorted(mc_conc.keys())
        if len(mc_levels) < 3:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        display_name = model.split("/")[-1] if "/" in model else model
        fig.suptitle(
            f"{display_name} ({config}): gRPC vs HTTP",
            fontsize=16,
            fontweight="bold",
        )

        for ax, (title, field, _lb, unit) in zip(axes.flat, _CHART_METRICS):
            gv, hv = [], []
            for conc in mc_levels:
                g = sum(getattr(c.grpc, field) for c in mc_conc[conc]) / len(
                    mc_conc[conc]
                )
                h = sum(getattr(c.http, field) for c in mc_conc[conc]) / len(
                    mc_conc[conc]
                )
                if unit == "ms":
                    g *= 1000
                    h *= 1000
                gv.append(g)
                hv.append(h)

            x = range(len(mc_levels))
            ax.plot(x, gv, "o-", label="gRPC", color="#2196F3", linewidth=2)
            ax.plot(x, hv, "s--", label="HTTP", color="#FF9800", linewidth=2)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Concurrency")
            ax.set_ylabel(f"{title} ({unit})")
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(c) for c in mc_levels])
            ax.legend()
            ax.grid(True, alpha=0.3)
            nz = [v for v in gv + hv if v > 0]
            if nz and max(nz) > 10 * min(nz):
                ax.set_yscale("log")

        fig.tight_layout()
        safe = model.replace("/", "__")
        safe_cfg = config.replace("/", "_")
        fname = f"{safe}_{safe_cfg}_comparison.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        charts.append(fname)

    return charts


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

_RUNTIME_DISPLAY = {"sglang": "SGLang", "vllm": "vLLM", "trtllm": "TRT-LLM"}
_PROTOCOL_DISPLAY = {"http": "HTTP", "grpc": "gRPC"}


def _section_key_findings(
    comparisons: list[ComparisonPoint],
    experiments: list[ExperimentInfo],
) -> list[str]:
    """Auto-generated executive summary of the benchmark results."""
    if not comparisons:
        return []

    lines = ["### Key Findings", ""]

    # 1. Overall verdict per key metric
    for label, field, lower_better, _unit in _CHART_METRICS:
        advs = [
            a
            for cp in comparisons
            if (
                a := _advantage(
                    getattr(cp.grpc, field), getattr(cp.http, field), lower_better
                )
            )
            is not None
        ]
        if not advs:
            continue
        avg_adv = sum(advs) / len(advs)
        grpc_wins = sum(1 for a in advs if a > 2)
        http_wins = sum(1 for a in advs if a < -2)
        ties = len(advs) - grpc_wins - http_wins
        if abs(avg_adv) < 1:
            verdict = f"**{label}**: No clear winner — essentially tied across all scenarios"
        else:
            winner = "gRPC" if avg_adv > 0 else "HTTP"
            lines.append(
                f"- **{label}**: {winner} wins {max(grpc_wins, http_wins)}/{len(advs)} "
                f"comparisons (avg {abs(avg_adv):.1f}% better), "
                f"{min(grpc_wins, http_wins)} losses, {ties} ties"
            )
            continue
        lines.append(f"- {verdict}")

    # 2. Error rates
    total_runs = sum(len(e.runs) for e in experiments)
    error_runs = [
        (e, r)
        for e in experiments
        for r in e.runs
        if r.error_rate > 0
    ]
    if error_runs:
        lines.append(
            f"- **Errors**: {len(error_runs)}/{total_runs} runs had non-zero error rates"
        )
    else:
        lines.append(f"- **Errors**: All {total_runs} runs completed with 0% error rate")

    # 3. Biggest outliers
    biggest_grpc_win = biggest_http_win = None
    biggest_grpc_adv = biggest_http_adv = 0.0
    for cp in comparisons:
        for _label, field, lower_better, _ in _CHART_METRICS:
            adv = _advantage(
                getattr(cp.grpc, field), getattr(cp.http, field), lower_better
            )
            if adv is not None and adv > biggest_grpc_adv:
                biggest_grpc_adv = adv
                biggest_grpc_win = (cp, _label)
            if adv is not None and adv < biggest_http_adv:
                biggest_http_adv = adv
                biggest_http_win = (cp, _label)

    if biggest_grpc_win and biggest_grpc_adv > 10:
        cp, metric = biggest_grpc_win
        lines.append(
            f"- **Largest gRPC win**: {biggest_grpc_adv:.0f}% on {metric} "
            f"— {cp.model} `{cp.scenario}` C={cp.concurrency}"
        )
    if biggest_http_win and abs(biggest_http_adv) > 10:
        cp, metric = biggest_http_win
        lines.append(
            f"- **Largest HTTP win**: {abs(biggest_http_adv):.0f}% on {metric} "
            f"— {cp.model} `{cp.scenario}` C={cp.concurrency}"
        )

    lines.append("")
    return lines


def _section_error_rates(experiments: list[ExperimentInfo]) -> list[str]:
    """Surface any non-zero error rates."""
    errors = []
    for e in experiments:
        for r in e.runs:
            if r.error_rate > 0:
                errors.append((e, r))

    if not errors:
        return []

    lines = [
        "<details>",
        f"<summary><b>Runs with Errors</b> ({len(errors)} runs)</summary>",
        "",
        "| Model | Protocol | Runtime | Workers | Scenario | C | Error Rate |",
        "|-------|----------|---------|---------|----------|--:|-----------:|",
    ]

    for e, r in sorted(
        errors, key=lambda x: x[1].error_rate, reverse=True
    ):
        lines.append(
            f"| {e.model} | {e.protocol} | {e.runtime} | {e.worker_type} "
            f"| `{r.scenario}` | {r.concurrency} | {r.error_rate:.1%} |"
        )

    lines.extend(["", "</details>", ""])
    return lines


def _section_overview(
    experiments: list[ExperimentInfo], models_with_data: set[str]
) -> list[str]:
    """Overview table — only models that have comparison data."""
    by_model: dict[str, dict[str, ExperimentInfo]] = defaultdict(dict)
    for exp in experiments:
        by_model[exp.model][exp.table_key] = exp

    # Discover columns dynamically
    all_keys: set[str] = set()
    for model in models_with_data:
        if model in by_model:
            all_keys.update(by_model[model].keys())

    _worker_order = {"single": 0, "multi": 1}
    _protocol_order = {"http": 0, "grpc": 1}

    def _col_sort_key(key: str) -> tuple:
        protocol, runtime, worker = key.split("_")
        return (
            _worker_order.get(worker, 9),
            runtime,
            _protocol_order.get(protocol, 9),
        )

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

    for model in sorted(models_with_data):
        if model not in by_model:
            continue
        model_exps = by_model[model]
        row = [model]
        for table_key, _ in table_order:
            if table_key not in model_exps:
                row.append("\u2796")
            else:
                exp = model_exps[table_key]
                has_errors = any(
                    r.rps == 0 or r.output_throughput == 0 for r in exp.runs
                )
                row.append("\u26a0\ufe0f" if has_errors else "\u2705")
        lines.append("| " + " | ".join(row) + " |")

    # Note excluded models
    excluded = set(by_model.keys()) - models_with_data
    if excluded:
        names = ", ".join(sorted(excluded))
        lines.append("")
        lines.append(
            f"*Excluded from comparison (no matched gRPC/HTTP data): {names}*"
        )

    lines.append("")
    return lines


def _section_aggregate(comparisons: list[ComparisonPoint]) -> list[str]:
    """Aggregate gRPC vs HTTP comparison table."""
    if not comparisons:
        return ["*No gRPC vs HTTP comparison data available.*", ""]

    # (label, field, lower_is_better)
    metrics = [
        ("TTFT mean", "ttft_mean", True),
        ("TTFT p99", "ttft_p99", True),
        ("E2E mean", "e2e_mean", True),
        ("E2E p99", "e2e_p99", True),
        ("TPOT mean", "tpot_mean", True),
        ("TPOT p99", "tpot_p99", True),
        ("Output throughput", "output_throughput", False),
        ("Total throughput", "total_throughput", False),
        ("RPS", "rps", False),
    ]

    lines = [
        "### Aggregate: gRPC vs HTTP",
        "",
        f"*{len(comparisons)} matched data points. "
        "Shows which protocol is better and by how much.*",
        "",
        "| Metric | Avg | Median | Verdict |",
        "|--------|----:|-------:|---------|",
    ]

    for label, fld, lower_better in metrics:
        advs = []
        for cp in comparisons:
            a = _advantage(
                getattr(cp.grpc, fld), getattr(cp.http, fld), lower_better
            )
            if a is not None:
                advs.append(a)
        if not advs:
            continue
        avg = sum(advs) / len(advs)
        med = median(advs)
        lines.append(
            f"| {label} | {_fmt_winner(avg)} | {_fmt_winner(med)} | "
            f"{_fmt_winner_bold(avg)} |"
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

    def _conc_table(
        title: str, field: str, lower_is_better: bool, fmt_val
    ) -> list[str]:
        tbl = [
            "<details>",
            f"<summary><b>{title}</b></summary>",
            "",
            "| Concurrency | gRPC | HTTP | Faster |",
            "|---:|---:|---:|:---|",
        ]
        for conc in conc_levels:
            cps = by_conc[conc]
            g_avg = sum(getattr(cp.grpc, field) for cp in cps) / len(cps)
            h_avg = sum(getattr(cp.http, field) for cp in cps) / len(cps)
            adv = _advantage(g_avg, h_avg, lower_is_better)
            tbl.append(
                f"| {conc} | {fmt_val(g_avg)} | {fmt_val(h_avg)} | "
                f"{_fmt_winner_bold(adv)} |"
            )
        tbl.extend(["", "</details>", ""])
        return tbl

    def _fmt_ms(v: float) -> str:
        ms = v * 1000
        return f"{ms:.0f}ms" if ms < 1000 else f"{v:.2f}s"

    def _fmt_tput(v: float) -> str:
        return f"{_fmt_throughput(v)} tok/s"

    def _fmt_tpot(v: float) -> str:
        ms = v * 1000
        return f"{ms:.1f}ms" if ms >= 1 else f"{ms:.2f}ms"

    lines.extend(_conc_table("TTFT Mean", "ttft_mean", True, _fmt_ms))
    lines.extend(_conc_table("TTFT p99", "ttft_p99", True, _fmt_ms))
    lines.extend(_conc_table("TPOT Mean", "tpot_mean", True, _fmt_tpot))
    lines.extend(_conc_table("TPOT p99", "tpot_p99", True, _fmt_tpot))
    lines.extend(_conc_table("E2E Latency Mean", "e2e_mean", True, _fmt_ms))
    lines.extend(_conc_table("E2E Latency p99", "e2e_p99", True, _fmt_ms))
    lines.extend(_conc_table("Output Throughput", "output_throughput", False, _fmt_tput))

    return lines


def _section_scorecard(comparisons: list[ComparisonPoint]) -> list[str]:
    """Win/loss scorecard at different thresholds."""
    if not comparisons:
        return []

    lines = [
        "### Win/Loss Scorecard",
        "",
        "*How often is one protocol better by > N%?*",
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
                "| Threshold | gRPC better | HTTP better | Within |",
                "|---:|---:|---:|---:|",
            ]
        )

        for thresh in thresholds:
            grpc_w = http_w = within = 0
            for cp in comparisons:
                adv = _advantage(
                    getattr(cp.grpc, fld), getattr(cp.http, fld), lower_better
                )
                if adv is None:
                    continue
                if adv > thresh:
                    grpc_w += 1
                elif adv < -thresh:
                    http_w += 1
                else:
                    within += 1

            g_str = f"**{grpc_w}**" if grpc_w > http_w else str(grpc_w)
            h_str = f"**{http_w}**" if http_w > grpc_w else str(http_w)
            lines.append(f"| >{thresh}% | {g_str} | {h_str} | {within} |")

        lines.append("")

    return lines


def _section_top_wins(
    comparisons: list[ComparisonPoint], threshold: float = 30.0
) -> list[str]:
    """Table of all gRPC wins exceeding the threshold."""
    if not comparisons:
        return []

    metric_defs = [
        ("TTFT p99", "ttft_p99", True),
        ("TTFT mean", "ttft_mean", True),
        ("E2E mean", "e2e_mean", True),
        ("E2E p99", "e2e_p99", True),
        ("TPOT p99", "tpot_p99", True),
        ("Output tput", "output_throughput", False),
        ("RPS", "rps", False),
    ]

    wins: list[tuple[float, str, ComparisonPoint, float, float]] = []
    for cp in comparisons:
        for label, fld, lower_better in metric_defs:
            g = getattr(cp.grpc, fld)
            h = getattr(cp.http, fld)
            adv = _advantage(g, h, lower_better)
            if adv is not None and adv > threshold:
                wins.append((adv, label, cp, g, h))

    wins.sort(key=lambda x: x[0], reverse=True)

    if not wins:
        return []

    lines = [
        "<details>",
        f"<summary><b>Top gRPC Wins (&gt;{threshold:.0f}%)</b> — {len(wins)} entries</summary>",
        "",
        "| gRPC better by | Model | Config | Scenario | C | Metric | gRPC | HTTP |",
        "|---------------:|-------|--------|----------|--:|--------|-----:|-----:|",
    ]

    for adv, label, cp, g, h in wins:
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
            f"| {adv:.1f}% | {cp.model} | {config} | "
            f"`{cp.scenario}` | {cp.concurrency} | {label} | {g_str} | {h_str} |"
        )

    lines.extend(["", "</details>", ""])
    return lines


def _section_per_model(comparisons: list[ComparisonPoint]) -> list[str]:
    """Per-model summary table plus collapsible detail tables.

    All values show gRPC advantage: 'gRPC X%' = gRPC is X% better,
    'HTTP X%' = HTTP is X% better, '~' = within 2%.
    """
    if not comparisons:
        return []

    by_model: dict[str, list[ComparisonPoint]] = defaultdict(list)
    for cp in comparisons:
        by_model[cp.model].append(cp)

    lines = [
        "### Per-Model Summary",
        "",
        "*Each cell shows which protocol is better and by how much.*",
        "",
        "| Model | TTFT p99 | TPOT p99 | E2E p99 | Output tput | N |",
        "|-------|--------:|--------:|--------:|--------:|---:|",
    ]

    for model in sorted(by_model.keys()):
        cps = by_model[model]

        def _model_avg(field: str, lower_better: bool) -> str:
            advs = [
                a
                for cp in cps
                if (a := _advantage(getattr(cp.grpc, field), getattr(cp.http, field), lower_better))
                is not None
            ]
            if not advs:
                return "N/A"
            return _fmt_winner(sum(advs) / len(advs))

        lines.append(
            f"| {model} "
            f"| {_model_avg('ttft_p99', True)} "
            f"| {_model_avg('tpot_p99', True)} "
            f"| {_model_avg('e2e_p99', True)} "
            f"| {_model_avg('output_throughput', False)} "
            f"| {len(cps)} |"
        )

    lines.append("")

    # Detailed per-model/config tables
    for model in sorted(by_model.keys()):
        cps = by_model[model]
        by_config: dict[str, list[ComparisonPoint]] = defaultdict(list)
        for cp in cps:
            by_config[f"{cp.runtime}/{cp.worker_type}"].append(cp)

        for config in sorted(by_config.keys()):
            config_cps = sorted(
                by_config[config], key=lambda x: (x.scenario, x.concurrency)
            )
            lines.extend(
                [
                    "<details>",
                    f"<summary><b>{model} ({config})</b> — "
                    f"{len(config_cps)} points</summary>",
                    "",
                    "| Scenario | C | TTFT p99 | TPOT p99 | E2E p99 | Tput |",
                    "|----------|--:|--------:|--------:|--------:|--------:|",
                ]
            )

            for cp in config_cps:
                lines.append(
                    f"| `{cp.scenario}` | {cp.concurrency} "
                    f"| {_fmt_winner(_advantage(cp.grpc.ttft_p99, cp.http.ttft_p99, True))} "
                    f"| {_fmt_winner(_advantage(cp.grpc.tpot_p99, cp.http.tpot_p99, True))} "
                    f"| {_fmt_winner(_advantage(cp.grpc.e2e_p99, cp.http.e2e_p99, True))} "
                    f"| {_fmt_winner(_advantage(cp.grpc.output_throughput, cp.http.output_throughput, False))} |"
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

    # Only include models that have comparison data
    models_with_data = {cp.model for cp in comparisons}

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
        "<details>",
        "<summary><b>Glossary</b></summary>",
        "",
        "#### Metrics",
        "",
        "| Metric | Description |",
        "|--------|-------------|",
        "| **TTFT** | Time To First Token — latency from request sent to first token received |",
        "| **TPOT** | Time Per Output Token — average time between consecutive output tokens |",
        "| **E2E** | End-to-End latency — total time from request sent to last token received |",
        "| **Output tput** | Output throughput — tokens generated per second (tok/s) |",
        "| **Total tput** | Total throughput — input + output tokens processed per second |",
        "| **RPS** | Requests Per Second — completed requests per second |",
        "| **p99** | 99th percentile — the value below which 99% of observations fall (worst 1% of requests) |",
        "| **mean** | Arithmetic average across all requests |",
        "",
        "#### Traffic Scenarios",
        "",
        "| Pattern | Description |",
        "|---------|-------------|",
        "| **D(in, out)** | Deterministic — fixed input/output token lengths, e.g. `D(100,100)` = 100 input, 100 output tokens |",
        "| **N(μ,σ)/(μ,σ)** | Normal distribution — input and output lengths drawn from Gaussian, e.g. `N(480,240)/(300,150)` |",
        "| **E(size)** | Embedding — input of given token length, used for embedding model benchmarks |",
        "",
        "#### Comparison Columns",
        "",
        "| Value | Meaning |",
        "|-------|---------|",
        "| **gRPC X%** | gRPC is X% better than HTTP for this metric |",
        "| **HTTP X%** | HTTP is X% better than gRPC for this metric |",
        "| **~** | Difference is within 2% — essentially a tie |",
        "",
        "*Lower is better for latency metrics (TTFT, TPOT, E2E). Higher is better for throughput (tput, RPS).*",
        "",
        "</details>",
        "",
        "---",
        "",
    ]

    lines.extend(_section_key_findings(comparisons, experiments))
    lines.extend(_section_overview(experiments, models_with_data))
    lines.extend(_section_aggregate(comparisons))
    lines.extend(_section_by_concurrency(comparisons))
    lines.extend(_section_scorecard(comparisons))
    lines.extend(_section_top_wins(comparisons))
    lines.extend(_section_per_model(comparisons))
    lines.extend(_section_error_rates(experiments))

    lines.append("---")
    lines.append(
        f"*Generated from {len(experiments)} experiment(s), "
        f"{len(comparisons)} comparison points*"
    )

    return "\n".join(lines), experiments, comparisons


def main() -> None:
    """Main entry point.

    Usage: nightly_summarize.py [base_dir] [--charts-dir DIR]
    """
    args = sys.argv[1:]
    base_dir = Path.cwd()
    charts_dir: Path | None = None

    i = 0
    while i < len(args):
        if args[i] == "--charts-dir" and i + 1 < len(args):
            charts_dir = Path(args[i + 1])
            i += 2
        elif not args[i].startswith("-"):
            base_dir = Path(args[i])
            i += 1
        else:
            i += 1

    summary, experiments, comparisons = generate_summary(base_dir)

    # Generate comparison charts if requested
    if charts_dir and comparisons:
        chart_files = generate_charts(comparisons, experiments, charts_dir)
        if chart_files:
            print(
                f"Generated {len(chart_files)} chart(s) in {charts_dir}",
                file=sys.stderr,
            )

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
