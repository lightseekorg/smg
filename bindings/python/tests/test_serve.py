"""
Unit tests for the serve command's two-pass argument parsing.

Tests cover parse_serve_args, add_serve_args, _import_backend_args,
and the backend registry (BACKEND_ARG_ADDERS, BACKEND_CHOICES, DEFAULT_BACKEND).
"""

import argparse
from unittest.mock import patch

import pytest
from smg.serve import (
    BACKEND_ARG_ADDERS,
    BACKEND_CHOICES,
    DEFAULT_BACKEND,
    _add_trtllm_stub_args,
    _import_backend_args,
    add_serve_args,
    parse_serve_args,
)


class TestBackendRegistry:
    """Test the backend registry constants."""

    def test_default_backend_is_sglang(self):
        assert DEFAULT_BACKEND == "sglang"

    def test_backend_choices_match_registry(self):
        assert BACKEND_CHOICES == list(BACKEND_ARG_ADDERS.keys())

    def test_all_backends_registered(self):
        assert "sglang" in BACKEND_ARG_ADDERS
        assert "vllm" in BACKEND_ARG_ADDERS
        assert "trtllm" in BACKEND_ARG_ADDERS

    def test_registry_values_are_callable(self):
        for name, adder in BACKEND_ARG_ADDERS.items():
            assert callable(adder), f"Backend {name} adder is not callable"


class TestAddServeArgs:
    """Test add_serve_args populates parser correctly."""

    def test_adds_backend_arg(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--backend", "vllm"])
        assert args.backend == "vllm"

    def test_backend_default_is_sglang(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.backend == "sglang"

    def test_backend_rejects_invalid_choice(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--backend", "nonexistent"])

    def test_adds_dp_size(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--dp-size", "4"])
        assert args.dp_size == 4

    def test_dp_size_default(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.dp_size == 1

    def test_adds_worker_host(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--worker-host", "0.0.0.0"])
        assert args.worker_host == "0.0.0.0"

    def test_worker_host_default(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.worker_host == "127.0.0.1"

    def test_adds_worker_base_port(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--worker-base-port", "40000"])
        assert args.worker_base_port == 40000

    def test_worker_base_port_default(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.worker_base_port == 31000

    def test_adds_worker_startup_timeout(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--worker-startup-timeout", "600"])
        assert args.worker_startup_timeout == 600

    def test_worker_startup_timeout_default(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.worker_startup_timeout == 300


class TestImportBackendArgs:
    """Test _import_backend_args for each backend."""

    def test_trtllm_adds_model_arg(self):
        parser = argparse.ArgumentParser()
        _import_backend_args("trtllm", parser)
        args = parser.parse_args(["--model", "/path/to/model"])
        assert args.model == "/path/to/model"

    def test_sglang_import_error(self):
        """sglang is not installed in test env, so parser.error should be called."""
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit) as exc_info:
            _import_backend_args("sglang", parser)
        assert exc_info.value.code == 2

    def test_vllm_import_error(self):
        """vllm is not installed in test env, so parser.error should be called."""
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit) as exc_info:
            _import_backend_args("vllm", parser)
        assert exc_info.value.code == 2


class TestAddTrtllmStubArgs:
    """Test the TRT-LLM stub argument group."""

    def test_adds_model_arg(self):
        parser = argparse.ArgumentParser()
        _add_trtllm_stub_args(parser)
        args = parser.parse_args(["--model", "/tmp/model"])
        assert args.model == "/tmp/model"

    def test_model_default_is_none(self):
        parser = argparse.ArgumentParser()
        _add_trtllm_stub_args(parser)
        args = parser.parse_args([])
        assert args.model is None


class TestParseServeArgs:
    """Test the two-pass parse_serve_args function."""

    def test_trtllm_basic(self):
        backend, args = parse_serve_args(["--backend", "trtllm", "--model", "/tmp/m"])
        assert backend == "trtllm"
        assert args.backend == "trtllm"
        assert args.model == "/tmp/m"

    def test_trtllm_defaults(self):
        backend, args = parse_serve_args(["--backend", "trtllm"])
        assert backend == "trtllm"
        assert args.dp_size == 1
        assert args.worker_host == "127.0.0.1"
        assert args.worker_base_port == 31000
        assert args.worker_startup_timeout == 300

    def test_trtllm_with_serve_args(self):
        backend, args = parse_serve_args([
            "--backend", "trtllm",
            "--dp-size", "8",
            "--worker-host", "0.0.0.0",
            "--worker-base-port", "35000",
            "--worker-startup-timeout", "600",
        ])
        assert backend == "trtllm"
        assert args.dp_size == 8
        assert args.worker_host == "0.0.0.0"
        assert args.worker_base_port == 35000
        assert args.worker_startup_timeout == 600

    def test_trtllm_includes_router_args(self):
        """Router args should be included with --router- prefix."""
        backend, args = parse_serve_args([
            "--backend", "trtllm",
            "--router-policy", "round_robin",
        ])
        assert args.router_policy == "round_robin"

    def test_trtllm_router_args_defaults(self):
        """Router args should have sensible defaults."""
        _, args = parse_serve_args(["--backend", "trtllm"])
        assert args.router_policy == "cache_aware"
        assert args.router_pd_disaggregation is False
        assert args.router_disable_retries is False

    def test_default_backend_is_sglang_exits(self):
        """Default backend (sglang) is not installed, so should exit with error."""
        with pytest.raises(SystemExit) as exc_info:
            parse_serve_args([])
        assert exc_info.value.code == 2

    def test_sglang_explicit_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_serve_args(["--backend", "sglang"])
        assert exc_info.value.code == 2

    def test_vllm_explicit_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            parse_serve_args(["--backend", "vllm"])
        assert exc_info.value.code == 2

    def test_invalid_backend_exits(self):
        with pytest.raises(SystemExit):
            parse_serve_args(["--backend", "nonexistent"])

    def test_none_argv_uses_default(self):
        """parse_serve_args(None) should behave like empty list (default backend)."""
        with pytest.raises(SystemExit) as exc_info:
            parse_serve_args(None)
        assert exc_info.value.code == 2

    def test_help_exits_zero(self):
        """--help should display help and exit with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            parse_serve_args(["--backend", "trtllm", "--help"])
        assert exc_info.value.code == 0

    def test_two_pass_extracts_backend_first(self):
        """Backend-specific args should not cause errors during pass 1."""
        # --model is only valid for trtllm; pass 1 should ignore it
        backend, args = parse_serve_args([
            "--backend", "trtllm",
            "--model", "/some/path",
        ])
        assert backend == "trtllm"
        assert args.model == "/some/path"

    def test_unknown_arg_rejected_in_pass2(self):
        """Unknown args should be rejected by the full parser in pass 2."""
        with pytest.raises(SystemExit):
            parse_serve_args(["--backend", "trtllm", "--totally-unknown-flag"])
