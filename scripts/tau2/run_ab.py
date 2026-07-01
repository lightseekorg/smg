#!/usr/bin/env python3
"""Run τ²-bench against two serving "arms" and diff pass^k. Track B (multi-turn).

baseline = pure vLLM; candidate = SMG -> vLLM gRPC. Both expose an identical
OpenAI /v1 endpoint; the official `tau2` CLI points --agent-llm at each arm and
--user-llm at a FIXED gpt-5.2, so any score delta is attributable to the
frontend (tokenization + tool/reasoning parsing). Arms must already be serving
(see launch_arms.sh); this driver does not launch them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from math import comb
from pathlib import Path


@dataclass
class Arm:
    name: str
    base_url: str
    scores: dict[str, dict[str, float]] = field(default_factory=dict)


def passk(num_success: int, num_trials: int, k: int) -> float:
    """tau-bench pass^k unbiased estimator: C(c,k)/C(n,k); 0 if k>n or n==0."""
    if num_trials == 0 or k > num_trials:
        return 0.0
    if k <= 0:
        return 1.0
    return comb(num_success, k) / comb(num_trials, k)


def load_results(raw: dict) -> list[dict]:
    """Flatten τ²-bench results.json to [{task_id, reward}] (one record per trial).

    Validated schema (tau2-bench 1.0.0, recon d8e915f): the top-level Results
    object has a `simulations` list; each SimulationRun carries `task_id` and
    `reward_info.reward` (0.0/1.0). (`simulation_index[].reward` mirrors it.)
    """
    out: list[dict] = []
    for s in raw["simulations"]:
        out.append({"task_id": str(s["task_id"]), "reward": float(s["reward_info"]["reward"])})
    return out


def domain_scores(results: list[dict], k: int) -> dict[str, float]:
    """pass1 = mean reward over all trials; passk = mean over tasks of C(c,k)/C(n,k)."""
    by_task: dict[str, list[float]] = {}
    for r in results:
        by_task.setdefault(r["task_id"], []).append(r["reward"])
    all_rewards = [x for xs in by_task.values() for x in xs]
    pass1 = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    per_task = [passk(sum(1 for x in xs if x >= 1.0), len(xs), k) for xs in by_task.values()]
    passk_val = sum(per_task) / len(per_task) if per_task else 0.0
    return {"pass1": pass1, "passk": passk_val}


def run_tau2(
    arm: Arm,
    *,
    tau2: str,
    agent_model: str,
    domain: str,
    num_trials: int,
    num_tasks: int,
    max_concurrency: int,
    user_llm: str,
    data_dir: Path,
) -> None:
    """Run `tau2 run` for one arm+domain, then read back its results.json.

    Validated routing (tau2-bench 1.0.0): the agent uses LiteLLM's OpenAI provider
    with a per-call `api_base` pointing at this arm (via --agent-llm-args); the
    user uses the fixed gpt-5.2. Results land at
    <data_dir>/simulations/<save_to>/results.json.
    """
    save_to = f"ab_{arm.name}_{domain}"
    agent_args = json.dumps(
        {"api_base": arm.base_url.rstrip("/") + "/v1", "api_key": "smg-local", "temperature": 0.0}
    )
    user_args = json.dumps({"temperature": 0.0})
    cmd = [
        tau2,
        "run",
        "--domain",
        domain,
        "--agent-llm",
        f"openai/{agent_model}",
        "--agent-llm-args",
        agent_args,
        "--user-llm",
        user_llm,
        "--user-llm-args",
        user_args,
        "--num-trials",
        str(num_trials),
        "--save-to",
        save_to,
    ]
    if num_tasks > 0:
        cmd += ["--num-tasks", str(num_tasks)]
    if max_concurrency > 0:
        cmd += ["--max-concurrency", str(max_concurrency)]
    print(f"\n=== [{arm.name}/{domain}] {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    # Pin tau2's write dir to the same dir we read from, so results land where we
    # look for them regardless of any inherited $TAU2_DATA_DIR or whether tau2 is
    # installed editable vs into site-packages.
    env["TAU2_DATA_DIR"] = str(data_dir)
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        print(f"WARNING: [{arm.name}/{domain}] exited {proc.returncode}", file=sys.stderr)
    results_json = data_dir / "simulations" / save_to / "results.json"
    try:
        arm.scores[domain] = domain_scores(
            load_results(json.loads(results_json.read_text())), k=num_trials
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        # tau2 may have died before writing results (OOM, engine death, API auth).
        # Skip this domain rather than aborting — build_report renders "—" for a
        # missing domain, so one failure doesn't discard the rest of the run.
        print(f"WARNING: [{arm.name}/{domain}] no usable results ({e})", file=sys.stderr)


def build_report(baseline: Arm, candidate: Arm, domains: list[str], k: int):
    """Markdown + JSON; candidate − baseline; overall = unweighted mean."""

    def cell(x):
        return "—" if x is None else f"{x * 100:.2f}"

    def dcell(x):
        return "—" if x is None else f"{x * 100:+.2f}"

    rows, agg = [], {"pass1": {"b": [], "c": []}, "passk": {"b": [], "c": []}}
    for d in domains:
        b, c = baseline.scores.get(d, {}), candidate.scores.get(d, {})
        row = {"domain": d}
        for m in ("pass1", "passk"):
            bv, cv = b.get(m), c.get(m)
            row[m] = {
                "baseline": bv,
                "candidate": cv,
                "delta": (cv - bv) if (bv is not None and cv is not None) else None,
            }
            if bv is not None:
                agg[m]["b"].append(bv)
            if cv is not None:
                agg[m]["c"].append(cv)
        rows.append(row)

    overall = {}
    for m in ("pass1", "passk"):
        bo = sum(agg[m]["b"]) / len(agg[m]["b"]) if agg[m]["b"] else None
        co = sum(agg[m]["c"]) / len(agg[m]["c"]) if agg[m]["c"] else None
        overall[m] = {
            "baseline": bo,
            "candidate": co,
            "delta": (co - bo) if (bo is not None and co is not None) else None,
        }

    # At k=1, pass^k == pass^1, so show only the pass^1 columns (a duplicated
    # triple is confusing). At k>1 show both pass^1 and pass^k.
    metrics = ["pass1"] if k == 1 else ["pass1", "passk"]
    mlabel = {"pass1": "pass^1", "passk": f"pass^{k}"}

    def triple(d, bold=False):
        b, c, dl = cell(d["baseline"]), cell(d["candidate"]), dcell(d["delta"])
        return f"**{b}** | **{c}** | **{dl}**" if bold else f"{b} | {c} | {dl}"

    header = " | ".join(
        f"{baseline.name} {mlabel[m]} | {candidate.name} {mlabel[m]} | Δ" for m in metrics
    )
    lines = [
        f"# τ²-bench A/B — {candidate.name} (candidate) vs {baseline.name} (baseline)",
        "",
        f"| domain | {header} |",
        "|---" * (1 + 3 * len(metrics)) + "|",
    ]
    for r in rows:
        lines.append(f"| {r['domain']} | " + " | ".join(triple(r[m]) for m in metrics) + " |")
    lines.append(
        "| **overall** | " + " | ".join(triple(overall[m], bold=True) for m in metrics) + " |"
    )
    lines += [
        "",
        "_Same model · engine · checkpoint · sampling · user-sim (gpt-5.2) "
        "on both arms — only the frontend differs, so Δ is the parsing layer._",
    ]
    payload = {
        "baseline": {"name": baseline.name, "scores": baseline.scores},
        "candidate": {"name": candidate.name, "scores": candidate.scores},
        "per_domain": rows,
        "overall": overall,
    }
    return "\n".join(lines), payload


def score_arm(
    arm: Arm,
    *,
    tau2: str,
    agent_model: str,
    domains: list[str],
    num_trials: int,
    num_tasks: int,
    max_concurrency: int,
    user_llm: str,
    data_dir: Path,
) -> None:
    """Run tau2 for every domain against one already-serving arm, filling arm.scores."""
    for domain in domains:
        run_tau2(
            arm,
            tau2=tau2,
            agent_model=agent_model,
            domain=domain,
            num_trials=num_trials,
            num_tasks=num_tasks,
            max_concurrency=max_concurrency,
            user_llm=user_llm,
            data_dir=data_dir,
        )


def save_scores(arm: Arm, path: Path) -> None:
    """Persist one arm's per-domain scores so a later --diff can compare them.

    Sequential mode: a whole-node model (TP=8) can't run both arms at once, so
    each arm is scored on its own and the two score files are diffed afterwards.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"name": arm.name, "base_url": arm.base_url, "scores": arm.scores}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def load_scores(path: Path) -> Arm:
    """Rebuild an Arm (name + per-domain scores) from a file written by save_scores."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Arm(name=data["name"], base_url=data.get("base_url", ""), scores=data["scores"])


def write_report_and_gate(
    baseline: Arm, candidate: Arm, domains: list[str], k: int, args: argparse.Namespace
) -> int:
    """Emit the markdown + JSON comparison and apply the informational regression gate."""
    report_md, payload = build_report(baseline, candidate, domains, k)
    print("\n" + report_md)
    if args.out:
        args.out.write_text(report_md + "\n", encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    delta = payload["overall"]["passk"]["delta"]
    if delta is not None and delta < -args.tolerance:
        print(
            f"\nREGRESSION: {candidate.name} pass^{k} {delta * 100:.2f}pp "
            f"below {baseline.name} (tol {args.tolerance * 100:.2f}pp)",
            file=sys.stderr,
        )
        return 1
    return 0


def _parse_arm(spec: str) -> Arm:
    name, url = spec.split("=", 1)
    return Arm(name=name, base_url=url)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Concurrent mode (both arms serving at once on opposite GPU halves): pass both.
    p.add_argument("--baseline", help="name=base_url (concurrent mode)")
    p.add_argument("--candidate", help="name=base_url (concurrent mode)")
    # Sequential mode (one arm owns the whole node, TP=8): score one live arm now,
    # then later diff the two saved score files.
    p.add_argument("--score-arm", help="name=base_url of the single live arm to score")
    p.add_argument("--scores-out", type=Path, help="write this arm's per-domain scores JSON here")
    p.add_argument("--diff-baseline", type=Path, help="baseline scores JSON (from --scores-out)")
    p.add_argument("--diff-candidate", type=Path, help="candidate scores JSON (from --scores-out)")
    p.add_argument("--domains", default="retail,airline,telecom")
    p.add_argument("--num-trials", type=int, default=2)
    p.add_argument("--num-tasks", type=int, default=0, help="0 = all tasks")
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=0,
        help="tau2 --max-concurrency (concurrent simulations per arm; 0 = tau2 default)",
    )
    p.add_argument(
        "--agent-model",
        default="Qwen/Qwen3.6-27B",
        help="served model name on both arms (used as openai/<name>)",
    )
    p.add_argument("--user-llm", default="gpt-5.2", help="fixed user-sim model")
    p.add_argument("--tau2", default="tau2", help="path to the tau2 executable")
    p.add_argument(
        "--data-dir",
        type=Path,
        help="tau2 DATA_DIR (results written/read under <data-dir>/simulations)",
    )
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--out", type=Path)
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    # Mode: diff two previously-saved score files (sequential mode, final step).
    if args.diff_baseline or args.diff_candidate:
        if not (args.diff_baseline and args.diff_candidate):
            p.error("--diff-baseline and --diff-candidate must be given together")
        baseline = load_scores(args.diff_baseline)
        candidate = load_scores(args.diff_candidate)
        diff_domains = sorted(set(baseline.scores) | set(candidate.scores)) or domains
        return write_report_and_gate(baseline, candidate, diff_domains, args.num_trials, args)

    # Mode: score a single live arm and persist its scores (sequential mode, per arm).
    if args.score_arm:
        if not (args.scores_out and args.data_dir):
            p.error("--score-arm requires --scores-out and --data-dir")
        arm = _parse_arm(args.score_arm)
        score_arm(
            arm,
            tau2=args.tau2,
            agent_model=args.agent_model,
            domains=domains,
            num_trials=args.num_trials,
            num_tasks=args.num_tasks,
            max_concurrency=args.max_concurrency,
            user_llm=args.user_llm,
            data_dir=args.data_dir,
        )
        save_scores(arm, args.scores_out)
        print(f"[{arm.name}] scores -> {args.scores_out}: {arm.scores}")
        return 0

    # Mode: concurrent A/B — arms serve on opposite GPU halves, so score them in
    # PARALLEL (separate servers, separate save_to dirs) to roughly halve wall-clock.
    if not (args.baseline and args.candidate and args.data_dir):
        p.error("concurrent mode requires --baseline, --candidate and --data-dir")
    baseline = _parse_arm(args.baseline)
    candidate = _parse_arm(args.candidate)

    def score(arm: Arm) -> None:
        score_arm(
            arm,
            tau2=args.tau2,
            agent_model=args.agent_model,
            domains=domains,
            num_trials=args.num_trials,
            num_tasks=args.num_tasks,
            max_concurrency=args.max_concurrency,
            user_llm=args.user_llm,
            data_dir=args.data_dir,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        # list() forces both futures to complete and re-raises any exception.
        list(ex.map(score, (baseline, candidate)))
    return write_report_and_gate(baseline, candidate, domains, args.num_trials, args)


if __name__ == "__main__":
    raise SystemExit(main())
