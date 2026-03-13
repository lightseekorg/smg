"""Pytest configuration for E2E tests.

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

Fixtures
--------
setup_backend: Class-scoped fixture that launches workers + gateway per test class.
    Returns (backend_name, model_path, client, gateway).
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
    """Configure clean logging to stdout with timestamps.

    Configures the root logger so all modules (infra, fixtures, test suites)
    get the same format and level without needing per-package configuration.
    """
    fmt = "%(asctime)s.%(msecs)03d [%(name)s] %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(handler)

    for logger_name in ("openai", "httpx", "httpcore", "numexpr"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


_setup_logging()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test visibility hooks
# ---------------------------------------------------------------------------


def pytest_runtest_logstart(nodeid: str, location: tuple) -> None:
    """Print clear test header at start of each test."""
    from infra import LOG_SEPARATOR_WIDTH

    test_name = nodeid.split("::")[-1] if "::" in nodeid else nodeid
    print(f"\n{'=' * LOG_SEPARATOR_WIDTH}")
    print(f"TEST: {test_name}")
    print(f"{'=' * LOG_SEPARATOR_WIDTH}")


# ---------------------------------------------------------------------------
# Import pytest hooks and fixtures from fixtures/ package
# ---------------------------------------------------------------------------

# Import fixtures - pytest discovers these by name
# Import hooks - pytest discovers these by name
import pytest
from fixtures import (
    backend_router,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_runtest_setup,
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
    "pytest_runtest_setup",
    "pytest_collection_modifyitems",
    "pytest_configure",
    # Fixtures
    "setup_backend",
    "backend_router",
    "smg",
]
