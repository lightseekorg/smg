#!/usr/bin/env python3
"""Generate nightly benchmark summary for GitHub Actions.

Processes genai-bench output folders and creates a concise markdown report
showing key performance metrics across models, scenarios, and concurrency levels.

Usage:
    python nightly_summarize.py [base_dir]

The script discovers nightly_* folders and aggregates results into:
- Model overview table with peak throughput and best latency
- Per-scenario breakdown with concurrency scaling
- Error summary if any failures occurred
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    """Single benchmark run result (one scenario × concurrency combination)."""

    scenario: str
    concurrency: int
    requests_per_second: float
    output_throughput: float  # tokens/s
    input_throughput: float  # tokens/s
    total_throughput: float  # tokens/s
    ttft_mean: float  # seconds
    ttft_p99: float
    e2e_latency_mean: float  # seconds
    e2e_latency_p99: float
    tpot_mean: float  # time per output token
    error_rate: float
    num_requests: int
    run_duration: float


@dataclass
class ExperimentSummary:
    """Aggregated summary for one experiment folder."""

    folder_name: str
    model: str
    server_engine: str
    server_version: str
    gpu_type: str
    gpu_count: str
    task: str
    scenarios: list[str] = field(default_factory=list)
    runs: list[RunResult] = field(default_factory=list)

    @property
    def peak_output_throughput(self) -> float:
        """Peak output throughput across all runs."""
        return max((r.output_throughput for r in self.runs), default=0)

    @property
    def peak_total_throughput(self) -> float:
        """Peak total throughput across all runs."""
        return max((r.total_throughput for r in self.runs), default=0)

    @property
    def peak_rps(self) -> float:
        """Peak requests per second across all runs."""
        return max((r.requests_per_second for r in self.runs), default=0)

    @property
    def best_ttft(self) -> float:
        """Best (lowest) mean TTFT across all runs."""
        return min((r.ttft_mean for r in self.runs), default=float("inf"))

    @property
    def best_e2e_latency(self) -> float:
        """Best (lowest) mean E2E latency at concurrency=1."""
        c1_runs = [r for r in self.runs if r.concurrency == 1]
        return min((r.e2e_latency_mean for r in c1_runs), default=float("inf"))

    @property
    def total_errors(self) -> int:
        """Total error count across all runs."""
        return sum(
            int(r.num_requests * r.error_rate) for r in self.runs if r.error_rate > 0
        )

    @property
    def has_errors(self) -> bool:
        """Whether any run had errors."""
        return any(r.error_rate > 0 for r in self.runs)


def parse_run_json(path: Path) -> RunResult | None:
    """Parse a single benchmark run JSON file."""
    try:
        with path.open() as f:
            data = json.load(f)

        agg = data.get("aggregated_metrics", {})
        stats = agg.get("stats", {})

        return RunResult(
            scenario=agg.get("scenario", "unknown"),
            concurrency=agg.get("num_concurrency", 0),
            requests_per_second=agg.get("requests_per_second", 0),
            output_throughput=agg.get("mean_output_throughput_tokens_per_s", 0),
            input_throughput=agg.get("mean_input_throughput_tokens_per_s", 0),
            total_throughput=agg.get("mean_total_tokens_throughput_tokens_per_s", 0),
            ttft_mean=stats.get("ttft", {}).get("mean", 0),
            ttft_p99=stats.get("ttft", {}).get("p99", 0),
            e2e_latency_mean=stats.get("e2e_latency", {}).get("mean", 0),
            e2e_latency_p99=stats.get("e2e_latency", {}).get("p99", 0),
            tpot_mean=stats.get("tpot", {}).get("mean", 0),
            error_rate=agg.get("error_rate", 0),
            num_requests=agg.get("num_requests", 0),
            run_duration=agg.get("run_duration", 0),
        )
    except Exception as e:
        print(f"Warning: Failed to parse {path}: {e}", file=sys.stderr)
        return None


def parse_experiment_folder(folder: Path) -> ExperimentSummary | None:
    """Parse all results from an experiment folder."""
    # Parse metadata
    metadata_path = folder / "experiment_metadata.json"
    if not metadata_path.exists():
        print(f"Warning: No metadata found in {folder}", file=sys.stderr)
        return None

    try:
        with metadata_path.open() as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse metadata in {folder}: {e}", file=sys.stderr)
        return None

    summary = ExperimentSummary(
        folder_name=folder.name,
        model=metadata.get("model", "unknown"),
        server_engine=metadata.get("server_engine", "unknown"),
        server_version=metadata.get("server_version", "unknown"),
        gpu_type=metadata.get("server_gpu_type", "unknown"),
        gpu_count=metadata.get("server_gpu_count", "?"),
        task=metadata.get("task", "text-to-text"),
        scenarios=metadata.get("traffic_scenario", []),
    )

    # Parse all run results
    for json_file in folder.glob("*.json"):
        if "experiment_metadata" in json_file.name:
            continue
        if "gpu_utilization" in json_file.name:
            continue
        result = parse_run_json(json_file)
        if result:
            summary.runs.append(result)

    return summary if summary.runs else None


def discover_experiments(base_dir: Path) -> list[ExperimentSummary]:
    """Discover and parse all nightly experiment folders."""
    experiments = []

    # Look for nightly_* folders (from test_nightly_perf.py)
    for folder in base_dir.rglob("nightly_*"):
        if folder.is_dir():
            summary = parse_experiment_folder(folder)
            if summary:
                experiments.append(summary)

    # Also look for genai-bench style folders (openai_*, anthropic_*)
    for pattern in ["openai_*", "anthropic_*"]:
        for folder in base_dir.rglob(pattern):
            if folder.is_dir() and (folder / "experiment_metadata.json").exists():
                summary = parse_experiment_folder(folder)
                if summary:
                    experiments.append(summary)

    return sorted(experiments, key=lambda x: x.folder_name)


def format_throughput(val: float) -> str:
    """Format throughput value with K suffix for large numbers."""
    if val >= 1000:
        return f"{val/1000:.1f}K"
    return f"{val:.0f}"


def format_latency(val: float) -> str:
    """Format latency in appropriate units."""
    if val < 0.001:
        return f"{val*1000000:.0f}μs"
    if val < 1:
        return f"{val*1000:.0f}ms"
    return f"{val:.2f}s"


def generate_overview_table(experiments: list[ExperimentSummary]) -> list[str]:
    """Generate the main overview table."""
    lines = [
        "## Nightly Benchmark Summary",
        "",
        "### Overview",
        "",
        "| Model | Runtime | GPUs | Peak Output (tok/s) | Peak Total (tok/s) | Best TTFT | Best E2E @c1 | Status |",
        "|-------|---------|------|---------------------|--------------------|-----------|--------------| -------|",
    ]

    for exp in experiments:
        status = "⚠️ Errors" if exp.has_errors else "✅ OK"
        lines.append(
            f"| {exp.model} | {exp.server_engine} {exp.server_version} | "
            f"{exp.gpu_count}×{exp.gpu_type} | "
            f"{format_throughput(exp.peak_output_throughput)} | "
            f"{format_throughput(exp.peak_total_throughput)} | "
            f"{format_latency(exp.best_ttft)} | "
            f"{format_latency(exp.best_e2e_latency)} | "
            f"{status} |"
        )

    return lines


def generate_scenario_details(experiments: list[ExperimentSummary]) -> list[str]:
    """Generate per-scenario breakdown tables."""
    lines = ["", "### Scenario Details", ""]

    for exp in experiments:
        lines.append(f"#### {exp.model} ({exp.server_engine})")
        lines.append("")

        # Group runs by scenario
        by_scenario: dict[str, list[RunResult]] = defaultdict(list)
        for run in exp.runs:
            by_scenario[run.scenario].append(run)

        for scenario in sorted(by_scenario.keys()):
            runs = sorted(by_scenario[scenario], key=lambda r: r.concurrency)
            lines.append(f"**{scenario}**")
            lines.append("")
            lines.append(
                "| Concurrency | RPS | Output (tok/s) | TTFT (mean) | TTFT (p99) | E2E (mean) | E2E (p99) | Errors |"
            )
            lines.append(
                "|-------------|-----|----------------|-------------|------------|------------|-----------|--------|"
            )

            for run in runs:
                err_str = f"{run.error_rate*100:.1f}%" if run.error_rate > 0 else "0"
                lines.append(
                    f"| {run.concurrency} | {run.requests_per_second:.1f} | "
                    f"{format_throughput(run.output_throughput)} | "
                    f"{format_latency(run.ttft_mean)} | "
                    f"{format_latency(run.ttft_p99)} | "
                    f"{format_latency(run.e2e_latency_mean)} | "
                    f"{format_latency(run.e2e_latency_p99)} | "
                    f"{err_str} |"
                )
            lines.append("")

    return lines


def generate_peak_performance_table(experiments: list[ExperimentSummary]) -> list[str]:
    """Generate table showing peak performance per scenario."""
    lines = ["", "### Peak Performance by Scenario", ""]

    for exp in experiments:
        lines.append(f"#### {exp.model} ({exp.server_engine})")
        lines.append("")
        lines.append(
            "| Scenario | Best Concurrency | Peak RPS | Peak Output (tok/s) | Latency @peak |"
        )
        lines.append(
            "|----------|------------------|----------|---------------------|---------------|"
        )

        # Group runs by scenario
        by_scenario: dict[str, list[RunResult]] = defaultdict(list)
        for run in exp.runs:
            by_scenario[run.scenario].append(run)

        for scenario in sorted(by_scenario.keys()):
            runs = by_scenario[scenario]
            # Find run with peak throughput
            peak_run = max(runs, key=lambda r: r.output_throughput)
            lines.append(
                f"| {scenario} | {peak_run.concurrency} | "
                f"{peak_run.requests_per_second:.1f} | "
                f"{format_throughput(peak_run.output_throughput)} | "
                f"{format_latency(peak_run.e2e_latency_mean)} |"
            )
        lines.append("")

    return lines


def generate_concise_summary(experiments: list[ExperimentSummary]) -> list[str]:
    """Generate a very concise summary for quick review."""
    lines = [
        "",
        "### Quick Summary",
        "",
        "<details>",
        "<summary>Expand for full details</summary>",
        "",
    ]

    for exp in experiments:
        # Find best throughput scenario and its metrics
        if not exp.runs:
            continue

        peak_run = max(exp.runs, key=lambda r: r.output_throughput)
        c1_runs = [r for r in exp.runs if r.concurrency == 1]
        best_latency_run = min(c1_runs, key=lambda r: r.e2e_latency_mean) if c1_runs else peak_run

        lines.append(f"**{exp.model}** ({exp.server_engine}, {exp.gpu_count}×{exp.gpu_type})")
        lines.append(f"- Peak: {format_throughput(peak_run.output_throughput)} tok/s @ concurrency {peak_run.concurrency} ({peak_run.scenario})")
        lines.append(f"- Latency @c1: TTFT {format_latency(best_latency_run.ttft_mean)}, E2E {format_latency(best_latency_run.e2e_latency_mean)}")
        if exp.has_errors:
            lines.append(f"- ⚠️ {exp.total_errors} total errors")
        lines.append("")

    lines.append("</details>")
    return lines


def generate_summary(base_dir: Path) -> str:
    """Generate the full markdown summary."""
    experiments = discover_experiments(base_dir)

    if not experiments:
        return "## Nightly Benchmark Summary\n\nNo benchmark results found."

    lines = []
    lines.extend(generate_overview_table(experiments))
    lines.extend(generate_concise_summary(experiments))
    lines.extend(generate_peak_performance_table(experiments))
    lines.extend(generate_scenario_details(experiments))

    # Add footer with timestamp
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated from {len(experiments)} experiment(s)*")

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    summary = generate_summary(base_dir)

    # Write to GITHUB_STEP_SUMMARY if available
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
