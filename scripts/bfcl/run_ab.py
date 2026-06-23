#!/usr/bin/env python3
"""Run the official BFCL benchmark against two "arms" and diff the scores.

This is **Track B** of the parser-verification proposal: a live, end-to-end A/B
that holds the model + engine + checkpoint + sampling fixed and varies only the
*frontend* (chat template + tokenization + tool/reasoning parsing):

  * baseline = pure vLLM  (vLLM's own parsers)
  * candidate = SMG -> vLLM gRPC  (SMG's Rust parsers)

Both expose an identical OpenAI ``/v1`` endpoint. We point the **official**
``bfcl`` CLI (FC mode, so the server's parsed ``tool_calls`` are what gets
scored) at each arm via ``LOCAL_SERVER_ENDPOINT`` / ``LOCAL_SERVER_PORT`` +
``--skip-server-setup``, run ``generate`` then ``evaluate`` into a per-arm
``BFCL_PROJECT_ROOT``, parse the per-category accuracy, and emit a comparison
table. Any score delta is attributable to the frontend — that is the number
that persuades an engine to adopt SMG's parsing layer.

The arms must already be serving (see ``launch_arm.sh``); this driver does not
launch them. Example::

    python run_ab.py \\
        --baseline   vllm=http://127.0.0.1:31199 \\
        --candidate  smg=http://127.0.0.1:31200 \\
        --bfcl-model Qwen/Qwen3-4B-Instruct-2507-FC \\
        --categories simple_python,multiple,parallel,irrelevance \\
        --bfcl /home/keyang/bfcl-env/bin/bfcl \\
        --project-root /home/keyang/bfcl_ab \\
        --out /tmp/bfcl_ab.md --json-out /tmp/bfcl_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class Arm:
    """One side of the A/B: a name and an already-serving OpenAI base URL."""

    name: str
    base_url: str
    host: str = ""
    port: str = ""
    project_root: Path = field(default_factory=Path)
    scores: dict[str, float] = field(default_factory=dict)


def parse_arm(spec: str, project_root: Path) -> Arm:
    """Parse a ``name=url`` arm spec and derive host/port + a per-arm BFCL root."""
    if "=" not in spec:
        raise ValueError(f"arm spec must be name=url, got: {spec!r}")
    name, url = spec.split("=", 1)
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 80)
    return Arm(
        name=name,
        base_url=url,
        host=host,
        port=port,
        project_root=project_root / name,
    )


def run_bfcl(
    arm: Arm,
    *,
    bfcl: str,
    model: str,
    categories: list[str],
    num_threads: int,
    temperature: float,
    skip_generate: bool,
) -> None:
    """Run ``bfcl generate`` + ``bfcl evaluate`` for one arm, in its own root."""
    arm.project_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = str(arm.project_root)
    # OpenAICompletionsHandler (the FC handler we register the models with) talks to
    # this arm's OpenAI-compatible endpoint via OPENAI_BASE_URL. LOCAL_SERVER_* is
    # kept for any model bfcl ships with a local OSS handler instead.
    env["OPENAI_BASE_URL"] = f"http://{arm.host}:{arm.port}/v1"
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    env["LOCAL_SERVER_ENDPOINT"] = arm.host
    env["LOCAL_SERVER_PORT"] = arm.port
    # NOTE: we do NOT force HF_HUB_OFFLINE. With the model cached, bfcl runs fine
    # online (~7 req/s in practice); forcing offline would only hide a genuinely
    # missing cache. Export HF_HUB_OFFLINE=1 yourself for air-gapped runs.

    cats = ",".join(categories)
    if not skip_generate:
        _run(
            [
                bfcl,
                "generate",
                "--model",
                model,
                "--test-category",
                cats,
                "--skip-server-setup",
                "--backend",
                "vllm",
                "--temperature",
                str(temperature),
                "--num-threads",
                str(num_threads),
                "--allow-overwrite",
            ],
            env,
            f"[{arm.name}] generate",
        )
    _run(
        [bfcl, "evaluate", "--model", model, "--test-category", cats],
        env,
        f"[{arm.name}] evaluate",
    )
    arm.scores = parse_scores(arm.project_root, model, categories)


def _run(cmd: list[str], env: dict[str, str], label: str) -> None:
    print(f"\n=== {label}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        print(f"WARNING: {label} exited {proc.returncode}", file=sys.stderr)


def parse_scores(project_root: Path, model: str, categories: list[str]) -> dict[str, float]:
    """Extract per-category accuracy from BFCL's score output.

    BFCL writes ``<root>/score/<sanitized-model>/<category>_score.json`` whose
    FIRST line is a summary dict containing ``accuracy``. We glob for the model
    dir (sanitization differs across versions) and read each category's summary.
    """
    score_root = project_root / "score"
    out: dict[str, float] = {}
    if not score_root.is_dir():
        print(f"WARNING: no score dir at {score_root}", file=sys.stderr)
        return out
    for cat in categories:
        acc = _find_category_accuracy(score_root, cat)
        if acc is not None:
            out[cat] = acc
    return out


def _find_category_accuracy(score_root: Path, category: str) -> float | None:
    # BFCL nests scores as <model>/<section>/BFCL_v4_<category>_score.json, so
    # match a trailing-wildcard pattern (the BFCL_v4_ prefix varies by version).
    for path in score_root.rglob(f"*{category}_score.json"):
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0]
            summary = json.loads(first)
        except (OSError, ValueError, IndexError):
            continue
        for key in ("accuracy", "acc", "score"):
            if key in summary:
                return float(summary[key])
    return None


def build_report(baseline: Arm, candidate: Arm, categories: list[str]) -> tuple[str, dict]:
    """Build a markdown comparison table + a JSON blob; candidate minus baseline."""
    rows: list[dict] = []
    for cat in categories:
        b = baseline.scores.get(cat)
        c = candidate.scores.get(cat)
        delta = (c - b) if (b is not None and c is not None) else None
        rows.append({"category": cat, "baseline": b, "candidate": c, "delta": delta})

    b_vals = [r["baseline"] for r in rows if r["baseline"] is not None]
    c_vals = [r["candidate"] for r in rows if r["candidate"] is not None]
    b_overall = sum(b_vals) / len(b_vals) if b_vals else None
    c_overall = sum(c_vals) / len(c_vals) if c_vals else None
    overall_delta = (
        (c_overall - b_overall) if (b_overall is not None and c_overall is not None) else None
    )

    def fmt(x: float | None) -> str:
        return "—" if x is None else f"{x * 100:.2f}"

    def fmt_d(x: float | None) -> str:
        return "—" if x is None else f"{x * 100:+.2f}"

    lines = [
        f"# BFCL A/B — {candidate.name} (candidate) vs {baseline.name} (baseline)",
        "",
        f"| category | {baseline.name} | {candidate.name} | Δ (cand−base) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['category']} | {fmt(r['baseline'])} | {fmt(r['candidate'])} | {fmt_d(r['delta'])} |"
        )
    lines.append(
        f"| **overall (unweighted)** | **{fmt(b_overall)}** | **{fmt(c_overall)}** | **{fmt_d(overall_delta)}** |"
    )
    lines.append("")
    lines.append(
        "_Scores are % accuracy (official BFCL, FC mode). Same model, engine, "
        "checkpoint and sampling on both arms — the only difference is the "
        "frontend, so Δ is attributable to the tokenization+parsing layer._"
    )

    payload = {
        "baseline": {
            "name": baseline.name,
            "base_url": baseline.base_url,
            "scores": baseline.scores,
        },
        "candidate": {
            "name": candidate.name,
            "base_url": candidate.base_url,
            "scores": candidate.scores,
        },
        "per_category": rows,
        "overall": {"baseline": b_overall, "candidate": c_overall, "delta": overall_delta},
    }
    return "\n".join(lines), payload


def save_scores(arm: Arm, path: Path) -> None:
    """Persist one arm's per-category scores so a later --diff can compare them.

    Used by the sequential mode: when a model needs the whole node (TP=8) the two
    arms cannot run at once, so each is scored on its own and diffed afterwards.
    """
    path.write_text(
        json.dumps({"name": arm.name, "base_url": arm.base_url, "scores": arm.scores}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def load_scores(path: Path) -> Arm:
    """Rebuild an Arm (name + scores) from a file written by save_scores."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Arm(name=data["name"], base_url=data.get("base_url", ""), scores=data["scores"])


def write_report_and_gate(
    baseline: Arm, candidate: Arm, categories: list[str], args: argparse.Namespace
) -> int:
    """Emit the markdown + JSON comparison and apply the regression gate."""
    report_md, payload = build_report(baseline, candidate, categories)
    print("\n" + report_md)
    if args.out:
        args.out.write_text(report_md + "\n", encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    overall = payload["overall"]
    if overall["delta"] is not None and overall["delta"] < -args.tolerance:
        print(
            f"\nREGRESSION: {candidate.name} overall is {overall['delta'] * 100:.2f}pp "
            f"below {baseline.name} (tolerance {args.tolerance * 100:.2f}pp)",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Concurrent mode (both arms serving at once): pass both.
    p.add_argument("--baseline", help="name=base_url, e.g. vllm=http://127.0.0.1:31199")
    p.add_argument("--candidate", help="name=base_url, e.g. smg=http://127.0.0.1:31200")
    # Sequential mode (one arm owns the whole node, TP=8): score one live arm now,
    # then later diff the two saved score files.
    p.add_argument("--score-arm", help="name=base_url of the single live arm to score")
    p.add_argument("--scores-out", type=Path, help="write this arm's scores JSON here")
    p.add_argument("--diff-baseline", type=Path, help="baseline scores JSON (from --scores-out)")
    p.add_argument("--diff-candidate", type=Path, help="candidate scores JSON (from --scores-out)")
    p.add_argument(
        "--bfcl-model",
        help="BFCL model handler name, e.g. Qwen/Qwen3-4B-Instruct-2507-FC",
    )
    p.add_argument(
        "--categories", default="simple_python", help="comma-separated BFCL test categories"
    )
    p.add_argument("--bfcl", default="bfcl", help="path to the bfcl executable")
    p.add_argument("--project-root", default="/tmp/bfcl_ab", type=Path)
    p.add_argument("--num-threads", default=16, type=int)
    p.add_argument("--temperature", default=0.001, type=float)
    p.add_argument(
        "--tolerance",
        default=0.02,
        type=float,
        help="max allowed candidate-below-baseline overall drop",
    )
    p.add_argument(
        "--skip-generate",
        action="store_true",
        help="reuse existing generation results, only evaluate",
    )
    p.add_argument("--out", type=Path, help="write the markdown report here")
    p.add_argument("--json-out", type=Path, help="write the JSON report here")
    args = p.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    # Mode: diff two previously-saved score files (sequential mode, final step).
    if args.diff_baseline or args.diff_candidate:
        if not (args.diff_baseline and args.diff_candidate):
            p.error("--diff-baseline and --diff-candidate must be given together")
        return write_report_and_gate(
            load_scores(args.diff_baseline), load_scores(args.diff_candidate), categories, args
        )

    # Mode: score a single live arm and persist its scores (sequential mode, per arm).
    if args.score_arm:
        if not (args.bfcl_model and args.scores_out):
            p.error("--score-arm requires --bfcl-model and --scores-out")
        arm = parse_arm(args.score_arm, args.project_root)
        run_bfcl(
            arm,
            bfcl=args.bfcl,
            model=args.bfcl_model,
            categories=categories,
            num_threads=args.num_threads,
            temperature=args.temperature,
            skip_generate=args.skip_generate,
        )
        save_scores(arm, args.scores_out)
        print(f"[{arm.name}] scores -> {args.scores_out}: {arm.scores}")
        return 0

    # Mode: concurrent A/B — both arms serving at once.
    if not (args.baseline and args.candidate and args.bfcl_model):
        p.error("concurrent mode requires --baseline, --candidate and --bfcl-model")
    baseline = parse_arm(args.baseline, args.project_root)
    candidate = parse_arm(args.candidate, args.project_root)
    for arm in (baseline, candidate):
        run_bfcl(
            arm,
            bfcl=args.bfcl,
            model=args.bfcl_model,
            categories=categories,
            num_threads=args.num_threads,
            temperature=args.temperature,
            skip_generate=args.skip_generate,
        )
    return write_report_and_gate(baseline, candidate, categories, args)


if __name__ == "__main__":
    raise SystemExit(main())
