"""Pytest configuration for smg-grpc-servicer unit tests.

Adds the parent directory to ``sys.path`` so editable installs work
without needing ``pip install -e``, and declares an asyncio-mode default.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent

if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "tokenspeed: tests that require TokenSpeed")
