"""Unit tests for E2E connection-mode env parsing (no GPU)."""

from __future__ import annotations

import pytest
from infra.constants import (
    ENV_CONNECTION_MODE,
    ConnectionMode,
    get_connection_mode_override,
)


def test_unset_returns_none(monkeypatch):
    monkeypatch.delenv(ENV_CONNECTION_MODE, raising=False)
    assert get_connection_mode_override() is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("zmq", ConnectionMode.ZMQ),
        ("ZMQ", ConnectionMode.ZMQ),
        ("Grpc", ConnectionMode.GRPC),
        ("  http  ", ConnectionMode.HTTP),
    ],
)
def test_valid_values_are_case_insensitive(monkeypatch, value, expected):
    monkeypatch.setenv(ENV_CONNECTION_MODE, value)
    assert get_connection_mode_override() == expected


@pytest.mark.parametrize("value", ["", "   "])
def test_set_but_empty_raises(monkeypatch, value):
    # A set-but-blank var is a misconfiguration, not "unset".
    monkeypatch.setenv(ENV_CONNECTION_MODE, value)
    with pytest.raises(ValueError, match="set but empty"):
        get_connection_mode_override()


def test_invalid_value_raises(monkeypatch):
    monkeypatch.setenv(ENV_CONNECTION_MODE, "bogus")
    with pytest.raises(ValueError, match="not a valid connection mode"):
        get_connection_mode_override()
