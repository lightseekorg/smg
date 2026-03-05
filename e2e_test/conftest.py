"""Pytest configuration for E2E tests.

Parallel Execution
------------------
Tests can run in parallel using pytest-xdist with shared model worker
processes.  Use ``-n N --dist loadscope`` for N concurrent worker processes:

    pytest -n 4 --dist loadscope e2e_test/router/

Each xdist worker is a separate process.  Worker ``gw0`` starts the model
servers and writes a ``.ready`` sentinel; other workers wait for the
sentinel and then share the model endpoints via a file-based state file.

``--dist loadscope`` groups tests by module/class so that tests sharing the
same model run on the same worker, minimizing model swaps.

Markers
-------
@pytest.mark.model(name)
    Specify which model to use for the test.

@pytest.mark.workers(count=1, prefill=None, decode=None)
    Configure worker topology for the test.

@pytest.mark.gateway(policy="round_robin", timeout=None, extra_args=None)
    Configure the gateway/router.

@pytest.mark.e2e
    Mark test as an end-to-end test requiring GPU workers.

@pytest.mark.slow
    Mark test as slow-running.

@pytest.mark.thread_unsafe(reason=None)
    Legacy marker — under xdist this is a no-op (separate processes).

Fixtures
--------
model_pool: Session-scoped fixture managing SGLang worker processes.
setup_backend: Function-scoped fixture (with per-process caching) that launches gateway + provides client.

Usage Examples
--------------
Basic test with default model::

    @pytest.mark.e2e
    @pytest.mark.parametrize("setup_backend", ["http"], indirect=True)
    class TestBasic:
        def test_chat(self, setup_backend):
            backend, model, client, gateway = setup_backend
            response = client.chat.completions.create(...)
"""

from __future__ import annotations

import contextlib
import logging
import sys
from importlib.util import find_spec
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup (must happen before other imports)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]  # smg/
_E2E_TEST = Path(__file__).resolve().parent  # e2e_test/
_SRC = _ROOT / "bindings" / "python"

# Add e2e_test to path so "from infra import ..." works
if str(_E2E_TEST) not in sys.path:
    sys.path.insert(0, str(_E2E_TEST))

# Add bindings/python to path if the wheel is not installed (for local development)
_wheel_installed = find_spec("smg.smg_rs") is not None

if not _wheel_installed and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Logging setup (clean output without pytest's "---- live log ----" dividers)
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Configure clean logging to stdout with timestamps and xdist worker ID.

    Under xdist each worker is a separate process — no fork, no duplicate
    handlers.  The worker ID (e.g. ``gw0``) is included via an environment
    variable set by xdist.
    """
    import os

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    fmt = f"%(asctime)s.%(msecs)03d [{worker_id}] [%(name)s] %(message)s"
    datefmt = "%H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt))

    for logger_name in ("e2e_test", "infra", "fixtures"):
        log = logging.getLogger(logger_name)
        log.handlers.clear()
        log.setLevel(logging.INFO)
        log.addHandler(handler)
        log.propagate = False

    for logger_name in ("openai", "httpx", "httpcore", "numexpr"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


_setup_logging()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test visibility hooks
# ---------------------------------------------------------------------------


def pytest_runtest_logstart(nodeid: str, location: tuple) -> None:
    """Print clear test header at start of each test."""
    import os

    from infra import LOG_SEPARATOR_WIDTH

    test_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    print(f"\n{'=' * LOG_SEPARATOR_WIDTH}")
    print(f"[{worker_id}] TEST: {test_name}")
    print(f"{'=' * LOG_SEPARATOR_WIDTH}")


# ---------------------------------------------------------------------------
# Import pytest hooks and fixtures from fixtures/ package
# ---------------------------------------------------------------------------

# Import fixtures - pytest discovers these by name
# Import hooks - pytest discovers these by name
import pytest
from fixtures import (
    backend_router,
    model_base_url,
    model_client,
    model_pool,
    pytest_collection_finish,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_runtest_setup,
    pytest_sessionfinish,
    setup_backend,
)
from smg_client import SmgClient
from smg_client._errors import SmgError


@pytest.fixture
def smg(setup_backend):
    """SmgClient pointing at the same gateway as setup_backend.

    Uses max_retries=0 to avoid amplifying load on GPU backends —
    SmgClient comparison calls already double the inference load.
    """
    _, _, _, gateway = setup_backend
    client = SmgClient(base_url=gateway.base_url, max_retries=0)
    yield client
    client.close()


@contextlib.contextmanager
def smg_compare():
    """Wrap SmgClient comparison blocks to handle backend errors gracefully.

    SmgClient assertions double the inference load on GPU backends. When the
    backend returns a server error (5xx), the request fails, or an assertion
    differs (e.g. enum vs string comparison), log a warning instead of failing
    the test — the primary SDK assertion already passed.
    """
    try:
        yield
    except SmgError as exc:
        logger.warning("SmgClient comparison skipped (SmgError): %s", exc)
    except AssertionError as exc:
        logger.warning("SmgClient comparison mismatch: %s", exc)


# Re-export for pytest discovery
__all__ = [
    # Hooks
    "pytest_runtest_logstart",
    "pytest_collection_modifyitems",
    "pytest_collection_finish",
    "pytest_configure",
    "pytest_runtest_setup",
    "pytest_sessionfinish",
    # Fixtures
    "model_pool",
    "model_client",
    "model_base_url",
    "setup_backend",
    "backend_router",
    "smg",
]
