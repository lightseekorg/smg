#!/usr/bin/env python3
"""Generate nightly benchmark summary for GitHub Actions.

Produces a concise report with collapsible tables per model/runtime/protocol.

Usage:
    python nightly_summarize.py [base_dir]
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    """Single benchmark run result."""

    scenario: str
    concurrency: int
    rps: float
    output_throughput: float
    ttft_mean: float
    ttft_p99: float
    tpot_mean: float
    tpot_p99: float
    e2e_mean: float
    e2e_p99: float


@dataclass
class ExperimentInfo:
    """Parsed experiment metadata."""

    model: str
    protocol: str  # http, grpc
    runtime: str  # sglang, vllm
    worker_type: str  # single, multi
    gpu_type: str
    gpu_count: int
    runs: list[RunResult] = field(default_factory=list)

    @property
    def table_key(self) -> str:
        """Key for table grouping."""
        return f"{self.protocol}_{self.runtime}_{self.worker_type}"


def _get_float(d: dict, key: str, default: float = 0.0) -> float:
    """Get float value, handling None."""
    val = d.get(key)
    return float(val) if val is not None else default


def parse_folder_name(folder_name: str) -> dict:
    """Parse experiment info from folder name.

    Expected patterns (newest to oldest):
    - nightly_llama-8b_http_sglang_single -> model=llama-8b, protocol=http, runtime=sglang, worker_type=single
    - nightly_llama-8b_grpc_vllm_multi -> model=llama-8b, protocol=grpc, runtime=vllm, worker_type=multi
    - nightly_llama-8b_http_sglang -> model=llama-8b, protocol=http, runtime=sglang (worker_type=single)
    - nightly_llama-8b_http (legacy) -> model=llama-8b, protocol=http
    """
    info = {"model": "unknown", "protocol": "unknown", "runtime": None, "worker_type": "single"}

    # Remove nightly_ prefix
    name = folder_name.replace("nightly_", "")

    # Try newest format: model_protocol_runtime_worker_type (e.g., llama-8b_grpc_sglang_single)
    parts = name.rsplit("_", 3)

    if len(parts) >= 4 and parts[-1] in ("single", "multi") and parts[-2] in ("sglang", "vllm"):
        # Newest format: model_protocol_runtime_worker_type
        info["worker_type"] = parts[-1]
        info["runtime"] = parts[-2]
        info["protocol"] = parts[-3]
        info["model"] = "_".join(parts[:-3])
    elif len(parts) >= 3 and parts[-1] in ("sglang", "vllm"):
        # Old format without worker_type: model_protocol_runtime
        info["runtime"] = parts[-1]
        info["protocol"] = parts[-2]
        info["model"] = "_".join(parts[:-2])
    elif len(parts) >= 2 and parts[-1] in ("http", "grpc"):
        # Legacy format: model_protocol
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

    # Parse folder name for protocol info
    folder_info = parse_folder_name(folder.name)

    # Extract model name (short form)
    model_path = meta.get("model", "unknown")
    model = model_path.split("/")[-1] if "/" in model_path else model_path

    # Determine runtime from metadata or folder name
    runtime = meta.get("server_engine")
    if not runtime or runtime == "unknown":
        # Use runtime from folder name if available
        runtime = folder_info.get("runtime")
    if not runtime:
        # Fallback: check folder name for vllm/sglang
        if "vllm" in folder.name.lower():
            runtime = "vllm"
        else:
            runtime = "sglang"
    # Normalize to lowercase for consistent grouping
    runtime = runtime.lower() if runtime else "sglang"

    # Determine worker type from folder name parsing
    worker_type = folder_info.get("worker_type", "single")

    # Get GPU info
    gpu_type = meta.get("server_gpu_type") or "unknown"
    gpu_count_str = meta.get("server_gpu_count") or "1"
    try:
        gpu_count = int(gpu_count_str)
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

    # Parse run results
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
                ttft_mean=_get_float(ttft, "mean"),
                ttft_p99=_get_float(ttft, "p99"),
                tpot_mean=_get_float(tpot, "mean"),
                tpot_p99=_get_float(tpot, "p99"),
                e2e_mean=_get_float(e2e, "mean"),
                e2e_p99=_get_float(e2e, "p99"),
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


def format_throughput(val: float) -> str:
    """Format throughput with K suffix."""
    if val >= 1000:
        return f"{val/1000:.1f}K"
    return f"{val:.0f}"


def format_latency(val: float) -> str:
    """Format latency in ms or s."""
    if val < 1:
        return f"{val*1000:.0f}ms"
    return f"{val:.2f}s"


def generate_table(runs: list[RunResult]) -> list[str]:
    """Generate a markdown table for runs."""
    if not runs:
        return ["*No data*", ""]

    sorted_runs = sorted(runs, key=lambda r: (r.scenario, r.concurrency))

    lines = [
        "| Scenario | Concurrency | RPS | Output (tok/s) | TTFT (mean) | TTFT (p99) | TPOT (mean) | TPOT (p99) | E2E (mean) | E2E (p99) |",
        "|----------|-------------|-----|----------------|-------------|------------|-------------|------------|------------|-----------|",
    ]

    for run in sorted_runs:
        lines.append(
            f"| {run.scenario} | {run.concurrency} | "
            f"{run.rps:.1f} | {format_throughput(run.output_throughput)} | "
            f"{format_latency(run.ttft_mean)} | {format_latency(run.ttft_p99)} | "
            f"{format_latency(run.tpot_mean)} | {format_latency(run.tpot_p99)} | "
            f"{format_latency(run.e2e_mean)} | {format_latency(run.e2e_p99)} |"
        )

    lines.append("")
    return lines


def generate_overview_table(
    by_model: dict[str, dict[str, ExperimentInfo]],
    table_order: list[tuple[str, str]],
) -> list[str]:
    """Generate overview table with status emojis."""
    # Header
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
                row.append("\u2796")  # Heavy minus sign (skipped)
            else:
                exp = model_exps[table_key]
                # Check if any run had errors (0 RPS or 0 throughput indicates failure)
                has_errors = any(r.rps == 0 or r.output_throughput == 0 for r in exp.runs)
                if has_errors:
                    row.append("\u26A0\uFE0F")  # Warning sign (partial failure)
                else:
                    row.append("\u2705")  # Green checkmark (success)

        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return lines


def generate_summary(base_dir: Path) -> str:
    """Generate the full markdown summary."""
    experiments = discover_experiments(base_dir)

    if not experiments:
        return "## Nightly Benchmark Summary\n\nNo benchmark results found."

    # Group by model
    by_model: dict[str, dict[str, ExperimentInfo]] = defaultdict(dict)
    for exp in experiments:
        by_model[exp.model][exp.table_key] = exp

    # Define table order
    table_order = [
        ("http_sglang_single", "HTTP SGLang Single"),
        ("grpc_sglang_single", "gRPC SGLang Single"),
        ("grpc_vllm_single", "gRPC vLLM Single"),
        ("http_sglang_multi", "HTTP SGLang Multi"),
        ("grpc_sglang_multi", "gRPC SGLang Multi"),
        ("grpc_vllm_multi", "gRPC vLLM Multi"),
    ]

    lines = ["## Nightly Benchmark Summary", ""]

    # Add overview table
    lines.extend(generate_overview_table(by_model, table_order))

    for model in sorted(by_model.keys()):
        model_exps = by_model[model]

        lines.append(f"### {model}")
        lines.append("")

        for table_key, table_title in table_order:
            if table_key not in model_exps:
                continue

            exp = model_exps[table_key]

            # Show GPU info per runtime/worker combination
            gpu_info = f" ({exp.gpu_count}x {exp.gpu_type})" if exp.gpu_type != "unknown" else ""

            lines.append(f"<details>")
            lines.append(f"<summary><b>{table_title}</b>{gpu_info}</summary>")
            lines.append("")
            lines.extend(generate_table(exp.runs))
            lines.append("</details>")
            lines.append("")

    lines.append("---")
    lines.append(f"*Generated from {len(experiments)} experiment(s)*")

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
