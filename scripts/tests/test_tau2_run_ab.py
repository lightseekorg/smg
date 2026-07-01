import importlib.util
import json
import sys
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
