"""Silence LiteLLM's per-response "This model isn't mapped yet" ERROR.

tau2 asks litellm for the dollar cost of every completion. The self-hosted model
under test (served via a local vLLM/SMG OpenAI endpoint) has no entry in litellm's
public price map, so each call logs a noisy ERROR and reports cost $0 — thousands
of lines per leg, drowning the arm logs. Registering the model at zero cost makes
the lookup succeed, so the noise disappears with no effect on rewards/pass^k
(cost is already effectively 0 and not part of the A/B gate).

This file is a `sitecustomize` module: Python's `site` imports it automatically at
interpreter startup when its directory is on PYTHONPATH. The nightly puts this dir
on PYTHONPATH so the registration runs inside the tau2 subprocess (its own venv)
where litellm actually computes cost. The model is read from $TAU2_COST_FREE_MODEL
(falling back to $MODEL) so it stays leg-agnostic across the matrix, and every step
is best-effort: a missing litellm or a registration hiccup can never break a run.
"""

import contextlib
import os

try:
    import litellm
except ImportError:  # non-tau2 interpreters on the same PYTHONPATH (e.g. run_ab.py)
    litellm = None


def _register_zero_cost() -> None:
    model = os.environ.get("TAU2_COST_FREE_MODEL") or os.environ.get("MODEL")
    if not (model and litellm):
        return
    with contextlib.suppress(Exception):
        litellm.register_model(
            {
                model: {
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                    "litellm_provider": "openai",
                    "mode": "chat",
                }
            }
        )


_register_zero_cost()
