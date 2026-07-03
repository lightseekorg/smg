import importlib.util
import sys
from pathlib import Path

SHIM = Path(__file__).resolve().parents[1] / "tau2" / "costmap_shim" / "sitecustomize.py"


class FakeLitellm:
    """Stand-in for the litellm module the shim registers pricing against."""

    def __init__(self, raise_on_register=False):
        self.registered = []
        self._raise = raise_on_register

    def register_model(self, cost):
        if self._raise:
            raise RuntimeError("boom")
        self.registered.append(cost)


def _load_shim(monkeypatch, fake_litellm, model_env):
    # Inject a fake litellm so the shim's top-level `import litellm` binds to it,
    # set the model env it keys off of, then exec the module (registration runs
    # as an import side effect, exactly as sitecustomize does at interpreter start).
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.delenv("TAU2_COST_FREE_MODEL", raising=False)
    if model_env is None:
        monkeypatch.delenv("MODEL", raising=False)
    else:
        monkeypatch.setenv("MODEL", model_env)
    spec = importlib.util.spec_from_file_location("tau2_costmap_shim", SHIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registers_model_at_zero_cost(monkeypatch):
    fake = FakeLitellm()
    _load_shim(monkeypatch, fake, "Qwen/Qwen3.6-27B")
    assert fake.registered == [
        {
            "Qwen/Qwen3.6-27B": {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    ]


def test_no_model_env_is_noop(monkeypatch):
    fake = FakeLitellm()
    _load_shim(monkeypatch, fake, None)
    assert fake.registered == []


def test_explicit_override_wins_over_model(monkeypatch):
    fake = FakeLitellm()
    monkeypatch.setenv("TAU2_COST_FREE_MODEL", "custom/model-x")
    monkeypatch.setenv("MODEL", "Qwen/Qwen3.6-27B")
    monkeypatch.setitem(sys.modules, "litellm", fake)
    spec = importlib.util.spec_from_file_location("tau2_costmap_shim", SHIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert list(fake.registered[0].keys()) == ["custom/model-x"]


def test_register_failure_is_swallowed(monkeypatch):
    # A pricing-registration hiccup must never break a benchmark run.
    fake = FakeLitellm(raise_on_register=True)
    _load_shim(monkeypatch, fake, "Qwen/Qwen3.6-27B")  # must not raise
    assert fake.registered == []
