import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "tau2_run_ab", Path(__file__).resolve().parents[1] / "tau2" / "run_ab.py"
)
run_ab = importlib.util.module_from_spec(SPEC)
sys.modules["tau2_run_ab"] = run_ab
SPEC.loader.exec_module(run_ab)


def test_passk_all_pass():
    # 2 successes out of 2 trials, k=2 -> 1.0
    assert run_ab.passk(2, 2, 2) == 1.0


def test_passk_one_fail():
    # 1 success out of 2 trials, k=2 -> C(1,2)/C(2,2) = 0
    assert run_ab.passk(1, 2, 2) == 0.0


def test_passk_pass1_is_success_rate():
    # k=1 -> c/n
    assert run_ab.passk(1, 2, 1) == 0.5


def test_passk_k_gt_trials_is_zero():
    assert run_ab.passk(1, 1, 2) == 0.0


def test_domain_scores_groups_by_task():
    results = [
        {"task_id": "t1", "reward": 1.0},
        {"task_id": "t1", "reward": 1.0},
        {"task_id": "t2", "reward": 1.0},
        {"task_id": "t2", "reward": 0.0},
    ]
    scores = run_ab.domain_scores(results, k=2)
    assert scores["pass1"] == 0.75  # (1+1+1+0)/4
    assert scores["passk"] == 0.5  # t1 passes^2, t2 does not -> 0.5


def test_build_report_delta_and_overall():
    base = run_ab.Arm(name="vllm", base_url="u")
    cand = run_ab.Arm(name="smg", base_url="u")
    base.scores = {"retail": {"pass1": 0.80, "passk": 0.60}}
    cand.scores = {"retail": {"pass1": 0.82, "passk": 0.62}}
    md, payload = run_ab.build_report(base, cand, ["retail"], k=2)
    assert "retail" in md and "overall" in md
    assert payload["overall"]["passk"]["delta"] == pytest.approx(0.02)


def test_build_report_dedups_columns_at_k1():
    base = run_ab.Arm(name="vllm", base_url="u")
    cand = run_ab.Arm(name="smg", base_url="u")
    base.scores = {"retail": {"pass1": 0.6667, "passk": 0.6667}}
    cand.scores = {"retail": {"pass1": 1.0, "passk": 1.0}}
    # k=1: pass^k == pass^1, so only the pass^1 triple renders (no duplicate).
    md1, _ = run_ab.build_report(base, cand, ["retail"], k=1)
    assert "pass^2" not in md1
    assert md1.count("pass^1") == 2  # once per arm in the header, not four times
    # k>1: both pass^1 and pass^k columns render.
    md2, _ = run_ab.build_report(base, cand, ["retail"], k=2)
    assert "pass^2" in md2


def test_domain_scores_on_real_fixture():
    fixture = Path(__file__).resolve().parent / "fixtures" / "tau2_results_sample.json"
    raw = json.loads(fixture.read_text())
    results = run_ab.load_results(raw)  # schema validated live against tau2 1.0.0 output
    # Fixture: 3 tasks x 2 trials, mixed pass/fail -> pass1 = 3/6, and at k=2 one
    # task passes both trials -> pass^2 = 1/3. Pinned so a grouping/schema regression
    # (e.g. swapping pass1/passk, or reading the wrong reward key) actually fails.
    scores = run_ab.domain_scores(results, k=2)
    assert scores["pass1"] == pytest.approx(0.5)
    assert scores["passk"] == pytest.approx(1 / 3)


def _run_tau2(arm, tmp_path, **overrides):
    """Invoke run_tau2 with sensible defaults; overrides tune the knobs under test."""
    kwargs = dict(
        tau2="tau2",
        agent_model="Qwen/Qwen3.6-27B",
        domain="airline",
        num_trials=1,
        num_tasks=0,
        max_concurrency=0,
        user_llm="gpt-5.2",
        data_dir=tmp_path,
        request_timeout=0,
        run_timeout=0,
    )
    kwargs.update(overrides)
    run_ab.run_tau2(arm, **kwargs)


def test_run_tau2_injects_request_timeout_into_agent_args(monkeypatch, tmp_path):
    # A per-request litellm timeout caps a single generate() so a degenerate task
    # (Qwen3.6-27B airline/44.1 ran 77 min on one request) fails fast instead of
    # dragging the whole leg into the 6h job limit.
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = kwargs.get("timeout")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm(name="vllm", base_url="http://x:1")
    _run_tau2(arm, tmp_path, domain="retail", request_timeout=300, run_timeout=5400)

    i = captured["cmd"].index("--agent-llm-args")
    agent_args = json.loads(captured["cmd"][i + 1])
    assert agent_args["timeout"] == 300
    assert captured["timeout"] == 5400  # per-domain subprocess wall-clock cap


def test_run_tau2_omits_timeout_when_disabled(monkeypatch, tmp_path):
    # request_timeout/run_timeout == 0 preserves the pre-fix behavior for ad-hoc runs.
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = kwargs.get("timeout")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm(name="vllm", base_url="http://x:1")
    _run_tau2(arm, tmp_path, domain="retail", request_timeout=0, run_timeout=0)

    i = captured["cmd"].index("--agent-llm-args")
    agent_args = json.loads(captured["cmd"][i + 1])
    assert "timeout" not in agent_args
    assert captured["timeout"] is None


def test_run_tau2_survives_domain_timeout(monkeypatch, tmp_path):
    # A hung tau2 must NOT propagate or block: the domain is left unscored (renders
    # "—") so one wedged simulation can't consume the entire CI budget.
    def fake_run(cmd, **kwargs):
        raise run_ab.subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(run_ab.subprocess, "run", fake_run)
    arm = run_ab.Arm(name="vllm", base_url="http://x:1")
    _run_tau2(arm, tmp_path, domain="airline", request_timeout=300, run_timeout=5400)

    assert "airline" not in arm.scores


def test_save_load_scores_roundtrip(tmp_path):
    # Sequential mode persists one arm's per-domain scores, then --diff reloads them.
    arm = run_ab.Arm(
        name="vllm", base_url="http://x:1", scores={"retail": {"pass1": 0.7, "passk": 0.5}}
    )
    path = tmp_path / "vllm.json"
    run_ab.save_scores(arm, path)
    back = run_ab.load_scores(path)
    assert back.name == "vllm"
    assert back.scores == {"retail": {"pass1": 0.7, "passk": 0.5}}
