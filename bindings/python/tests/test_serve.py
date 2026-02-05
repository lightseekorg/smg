"""
Unit tests for the serve command's two-pass argument parsing.

Tests cover parse_serve_args, add_serve_args, _import_backend_args,
and the backend registry (BACKEND_ARG_ADDERS, BACKEND_CHOICES, DEFAULT_BACKEND).
"""

import argparse
import signal
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from smg.serve import (
    BACKEND_ARG_ADDERS,
    BACKEND_CHOICES,
    BACKEND_LAUNCHERS,
    DEFAULT_BACKEND,
    ServeOrchestrator,
    SglangWorkerLauncher,
    TrtllmWorkerLauncher,
    VllmWorkerLauncher,
    _add_trtllm_stub_args,
    _find_available_ports,
    _http_health_check,
    _import_backend_args,
    _is_port_available,
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


class TestBackendLauncherRegistry:
    """Test the BACKEND_LAUNCHERS registry."""

    def test_all_backends_have_launchers(self):
        assert "sglang" in BACKEND_LAUNCHERS
        assert "vllm" in BACKEND_LAUNCHERS
        assert "trtllm" in BACKEND_LAUNCHERS

    def test_launcher_classes(self):
        assert BACKEND_LAUNCHERS["sglang"] is SglangWorkerLauncher
        assert BACKEND_LAUNCHERS["vllm"] is VllmWorkerLauncher
        assert BACKEND_LAUNCHERS["trtllm"] is TrtllmWorkerLauncher


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

    def test_adds_data_parallel_size(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--dp-size", "4"])
        assert args.data_parallel_size == 4

    def test_data_parallel_size_default(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.data_parallel_size == 1

    def test_adds_connection_mode(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args(["--connection-mode", "http"])
        assert args.connection_mode == "http"

    def test_connection_mode_default_is_grpc(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.connection_mode == "grpc"

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

    def test_host_default_is_localhost(self):
        parser = argparse.ArgumentParser()
        add_serve_args(parser)
        args = parser.parse_args([])
        assert args.host == "127.0.0.1"


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
        assert args.data_parallel_size == 1
        assert args.worker_host == "127.0.0.1"
        assert args.worker_base_port == 31000
        assert args.worker_startup_timeout == 300
        assert args.connection_mode == "grpc"

    def test_trtllm_with_serve_args(self):
        backend, args = parse_serve_args(
            [
                "--backend",
                "trtllm",
                "--dp-size",
                "8",
                "--worker-host",
                "0.0.0.0",
                "--worker-base-port",
                "35000",
                "--worker-startup-timeout",
                "600",
            ]
        )
        assert backend == "trtllm"
        assert args.data_parallel_size == 8
        assert args.worker_host == "0.0.0.0"
        assert args.worker_base_port == 35000
        assert args.worker_startup_timeout == 600

    def test_trtllm_includes_router_args(self):
        """Router args should be included with --router- prefix."""
        backend, args = parse_serve_args(
            [
                "--backend",
                "trtllm",
                "--router-policy",
                "round_robin",
            ]
        )
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
        backend, args = parse_serve_args(
            [
                "--backend",
                "trtllm",
                "--model",
                "/some/path",
            ]
        )
        assert backend == "trtllm"
        assert args.model == "/some/path"

    def test_unknown_arg_rejected_in_pass2(self):
        """Unknown args should be rejected by the full parser in pass 2."""
        with pytest.raises(SystemExit):
            parse_serve_args(["--backend", "trtllm", "--totally-unknown-flag"])


# ---------------------------------------------------------------------------
# Worker launcher tests
# ---------------------------------------------------------------------------


class TestWorkerLauncherGpuEnv:
    """Test GPU assignment via _get_tp_size() and gpu_env()."""

    @pytest.mark.parametrize(
        "launcher_class, args, expected_tp",
        [
            (SglangWorkerLauncher, argparse.Namespace(tp_size=4), 4),
            (SglangWorkerLauncher, argparse.Namespace(), 1),
            (VllmWorkerLauncher, argparse.Namespace(tensor_parallel_size=2), 2),
            (VllmWorkerLauncher, argparse.Namespace(), 1),
            (TrtllmWorkerLauncher, argparse.Namespace(tp_size=8), 8),
            (TrtllmWorkerLauncher, argparse.Namespace(), 1),
        ],
    )
    def test_get_tp_size(self, launcher_class, args, expected_tp):
        """_get_tp_size returns the correct value from args or defaults to 1."""
        launcher = launcher_class()
        assert launcher._get_tp_size(args) == expected_tp

    def test_gpu_env_dp_rank_0_tp_2(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tp_size=2)
        env = launcher.gpu_env(args, dp_rank=0, env={})
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_gpu_env_dp_rank_1_tp_2(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tp_size=2)
        env = launcher.gpu_env(args, dp_rank=1, env={})
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"

    def test_gpu_env_dp_rank_2_tp_4(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(tensor_parallel_size=4)
        env = launcher.gpu_env(args, dp_rank=2, env={})
        assert env["CUDA_VISIBLE_DEVICES"] == "8,9,10,11"

    def test_gpu_env_tp_1_default(self):
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace()  # no tp_size → default 1
        env = launcher.gpu_env(args, dp_rank=3, env={})
        assert env["CUDA_VISIBLE_DEVICES"] == "3"

    def test_gpu_env_preserves_existing_env(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tp_size=1)
        base_env = {"PATH": "/usr/bin", "HOME": "/root"}
        env = launcher.gpu_env(args, dp_rank=0, env=base_env)
        assert env["PATH"] == "/usr/bin"
        assert env["HOME"] == "/root"
        assert env["CUDA_VISIBLE_DEVICES"] == "0"

    def test_gpu_env_does_not_mutate_input(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tp_size=1)
        base_env = {"FOO": "bar"}
        env = launcher.gpu_env(args, dp_rank=0, env=base_env)
        assert "CUDA_VISIBLE_DEVICES" not in base_env
        assert "CUDA_VISIBLE_DEVICES" in env

    def test_gpu_env_none_copies_os_environ(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(tp_size=1)
        with patch.dict("os.environ", {"TEST_VAR": "123"}, clear=False):
            env = launcher.gpu_env(args, dp_rank=0)
        assert env["TEST_VAR"] == "123"
        assert env["CUDA_VISIBLE_DEVICES"] == "0"


class TestSglangWorkerLauncher:
    """Test SglangWorkerLauncher.build_command()."""

    def test_build_command_grpc_mode_default(self):
        """Default connection_mode is grpc, so --grpc-mode should be present."""
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(model_path="/tmp/model", connection_mode="grpc")
        cmd = launcher.build_command(args, "127.0.0.1", 31000)
        assert "--model-path" in cmd
        assert "/tmp/model" in cmd
        assert "--host" in cmd
        assert "127.0.0.1" in cmd
        assert "--port" in cmd
        assert "31000" in cmd
        assert "--grpc-mode" in cmd

    def test_build_command_http_mode(self):
        """When connection_mode is http, --grpc-mode should not be present."""
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(model_path="/tmp/model", connection_mode="http")
        cmd = launcher.build_command(args, "127.0.0.1", 31000)
        assert "--grpc-mode" not in cmd

    def test_worker_url_grpc_mode(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        assert launcher.worker_url(args, "127.0.0.1", 31000) == "grpc://127.0.0.1:31000"

    def test_worker_url_http_mode(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(connection_mode="http")
        assert launcher.worker_url(args, "127.0.0.1", 31000) == "http://127.0.0.1:31000"

    def test_health_check_grpc_mode(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        with patch("smg.serve._grpc_health_check", return_value=True) as mock:
            result = launcher.health_check(args, "127.0.0.1", 31000, 5.0)
        assert result is True
        mock.assert_called_once_with("127.0.0.1", 31000, 5.0)

    def test_health_check_http_mode(self):
        launcher = SglangWorkerLauncher()
        args = argparse.Namespace(connection_mode="http")
        with patch("smg.serve._http_health_check", return_value=True) as mock:
            result = launcher.health_check(args, "127.0.0.1", 31000, 5.0)
        assert result is True
        mock.assert_called_once_with("http://127.0.0.1:31000/health", 5.0)


class TestVllmWorkerLauncher:
    """Test VllmWorkerLauncher.build_command()."""

    def test_build_command(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="grpc")
        cmd = launcher.build_command(args, "0.0.0.0", 32000)
        assert "vllm.entrypoints.grpc_server" in cmd
        assert "--model" in cmd
        assert "/tmp/model" in cmd
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert "--port" in cmd
        assert "32000" in cmd

    def test_build_command_rejects_http_mode(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="http")
        with pytest.raises(ValueError, match="vLLM backend only supports grpc"):
            launcher.build_command(args, "0.0.0.0", 32000)

    def test_worker_url(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        assert launcher.worker_url(args, "127.0.0.1", 32000) == "grpc://127.0.0.1:32000"

    def test_health_check_delegates_to_grpc(self):
        launcher = VllmWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        with patch("smg.serve._grpc_health_check", return_value=True) as mock:
            result = launcher.health_check(args, "127.0.0.1", 32000, 5.0)
        assert result is True
        mock.assert_called_once_with("127.0.0.1", 32000, 5.0)


class TestTrtllmWorkerLauncher:
    """Test TrtllmWorkerLauncher.build_command()."""

    def test_build_command(self):
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="grpc")
        cmd = launcher.build_command(args, "0.0.0.0", 50051)
        assert "tensorrt_llm.commands.serve" in cmd
        assert "/tmp/model" in cmd
        assert "--grpc" in cmd
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert "--port" in cmd
        assert "50051" in cmd

    def test_build_command_rejects_http_mode(self):
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(model="/tmp/model", connection_mode="http")
        with pytest.raises(ValueError, match="TensorRT-LLM backend only supports grpc"):
            launcher.build_command(args, "0.0.0.0", 50051)

    def test_worker_url(self):
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        assert launcher.worker_url(args, "127.0.0.1", 50051) == "grpc://127.0.0.1:50051"

    def test_health_check_delegates_to_grpc(self):
        launcher = TrtllmWorkerLauncher()
        args = argparse.Namespace(connection_mode="grpc")
        with patch("smg.serve._grpc_health_check", return_value=True) as mock:
            result = launcher.health_check(args, "127.0.0.1", 50051, 5.0)
        assert result is True
        mock.assert_called_once_with("127.0.0.1", 50051, 5.0)


# ---------------------------------------------------------------------------
# Health check utility tests
# ---------------------------------------------------------------------------


class TestHttpHealthCheck:
    """Test _http_health_check with mocked urllib."""

    def test_returns_true_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert _http_health_check("http://localhost:31000/health", 5.0) is True

    def test_returns_false_on_error(self):
        with patch("urllib.request.urlopen", side_effect=ConnectionError):
            assert _http_health_check("http://localhost:31000/health", 5.0) is False

    def test_returns_false_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status = 503
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert _http_health_check("http://localhost:31000/health", 5.0) is False


# ---------------------------------------------------------------------------
# Port discovery tests
# ---------------------------------------------------------------------------


class TestFindAvailablePorts:
    """Test _find_available_ports returns correct count of ports."""

    def test_returns_requested_count(self):
        with patch("smg.serve._is_port_available", return_value=True):
            ports = _find_available_ports(31000, 3)
        assert len(ports) == 3

    def test_first_port_is_base(self):
        with patch("smg.serve._is_port_available", return_value=True):
            ports = _find_available_ports(31000, 1)
        assert ports[0] == 31000

    def test_skips_unavailable_ports(self):
        # First call: port 31000 unavailable, then 31001 available
        call_count = 0

        def mock_available(port):
            nonlocal call_count
            call_count += 1
            return port != 31000

        with patch("smg.serve._is_port_available", side_effect=mock_available):
            ports = _find_available_ports(31000, 1)
        assert ports[0] == 31001

    def test_ports_are_unique(self):
        with patch("smg.serve._is_port_available", return_value=True):
            ports = _find_available_ports(31000, 5)
        assert len(set(ports)) == 5

    def test_raises_when_no_ports_available(self):
        with patch("smg.serve._is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="Could not find"):
                _find_available_ports(65530, 10)


class TestIsPortAvailable:
    """Test _is_port_available socket check."""

    def test_available_port(self):
        # Port 0 lets the OS pick a free port; we test a high ephemeral port
        # that is almost certainly free
        assert isinstance(_is_port_available(39999), bool)

    def test_occupied_port(self):
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        try:
            assert _is_port_available(port) is False
        finally:
            s.close()


# ---------------------------------------------------------------------------
# ServeOrchestrator tests
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    """Create a minimal argparse.Namespace for orchestrator tests."""
    defaults = {
        "backend": "sglang",
        "data_parallel_size": 2,
        "connection_mode": "grpc",
        "worker_host": "127.0.0.1",
        "worker_base_port": 31000,
        "worker_startup_timeout": 10,
        "model_path": "/tmp/model",
        # router args with router_ prefix
        "router_policy": "cache_aware",
        "router_pd_disaggregation": False,
        "router_disable_retries": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestServeOrchestrator:
    """Test ServeOrchestrator lifecycle methods."""

    def test_build_router_args_injects_worker_urls_grpc(self):
        args = _make_args(data_parallel_size=2, connection_mode="grpc")
        orch = ServeOrchestrator("sglang", args)
        # Simulate workers already launched
        mock_proc1 = MagicMock()
        mock_proc2 = MagicMock()
        orch.workers = [(mock_proc1, 31000), (mock_proc2, 31003)]

        with patch("smg.serve.RouterArgs.from_cli_args") as mock_from_cli:
            mock_router_args = MagicMock()
            mock_from_cli.return_value = mock_router_args
            result = orch._build_router_args()

        mock_from_cli.assert_called_once_with(args, use_router_prefix=True)
        assert mock_router_args.worker_urls == [
            "grpc://127.0.0.1:31000",
            "grpc://127.0.0.1:31003",
        ]
        assert result is mock_router_args

    def test_build_router_args_http_mode(self):
        args = _make_args(data_parallel_size=2, connection_mode="http")
        orch = ServeOrchestrator("sglang", args)
        mock_proc = MagicMock()
        orch.workers = [(mock_proc, 31000), (mock_proc, 31003)]

        with patch("smg.serve.RouterArgs.from_cli_args") as mock_from_cli:
            mock_router_args = MagicMock()
            mock_from_cli.return_value = mock_router_args
            orch._build_router_args()

        assert mock_router_args.worker_urls == [
            "http://127.0.0.1:31000",
            "http://127.0.0.1:31003",
        ]

    def test_build_router_args_vllm_grpc_urls(self):
        args = _make_args(
            backend="vllm", data_parallel_size=2, model="/tmp/m", connection_mode="grpc"
        )
        orch = ServeOrchestrator("vllm", args)
        mock_proc = MagicMock()
        orch.workers = [(mock_proc, 32000), (mock_proc, 32003)]

        with patch("smg.serve.RouterArgs.from_cli_args") as mock_from_cli:
            mock_router_args = MagicMock()
            mock_from_cli.return_value = mock_router_args
            orch._build_router_args()

        assert mock_router_args.worker_urls == [
            "grpc://127.0.0.1:32000",
            "grpc://127.0.0.1:32003",
        ]

    def test_cleanup_workers_handles_already_dead_process(self):
        args = _make_args()
        orch = ServeOrchestrator("sglang", args)
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.wait.return_value = 0
        orch.workers = [(mock_proc, 31000)]

        # killpg raises ProcessLookupError (process already dead)
        with patch("os.killpg", side_effect=ProcessLookupError):
            orch._cleanup_workers()  # should not raise

        mock_proc.wait.assert_called_once()

    def test_cleanup_workers_sigkill_on_timeout(self):
        args = _make_args()
        orch = ServeOrchestrator("sglang", args)
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=30)
        orch.workers = [(mock_proc, 31000)]

        with patch("os.killpg") as mock_killpg:
            orch._cleanup_workers()

        # First call is SIGTERM, second is SIGKILL after timeout
        assert mock_killpg.call_count == 2
        mock_killpg.assert_any_call(12345, signal.SIGTERM)
        mock_killpg.assert_any_call(12345, signal.SIGKILL)

    def test_cleanup_workers_empty_list(self):
        args = _make_args()
        orch = ServeOrchestrator("sglang", args)
        orch.workers = []
        # Should be a no-op
        orch._cleanup_workers()

    def test_signal_handler_sets_guard_flag(self):
        args = _make_args()
        orch = ServeOrchestrator("sglang", args)
        orch.workers = []

        assert orch._shutting_down is False
        with pytest.raises(SystemExit) as exc_info:
            orch._signal_handler(signal.SIGINT, None)
        assert exc_info.value.code == 128 + signal.SIGINT
        assert orch._shutting_down is True

    def test_signal_handler_guard_prevents_reentry(self):
        args = _make_args()
        orch = ServeOrchestrator("sglang", args)
        orch._shutting_down = True

        # Should return immediately without calling sys.exit
        orch._signal_handler(signal.SIGINT, None)

    def test_trtllm_orchestrator_launches_grpc_workers(self):
        args = _make_args(
            backend="trtllm", data_parallel_size=1, model="/tmp/m", connection_mode="grpc"
        )
        orch = ServeOrchestrator("trtllm", args)

        with patch("smg.serve._find_available_ports", return_value=[50051]):
            with patch.object(orch.launcher, "launch") as mock_launch:
                mock_launch.return_value = MagicMock(pid=1234)
                orch._launch_workers()

        assert len(orch.workers) == 1
        assert orch.workers[0][1] == 50051

    def test_launch_workers_passes_gpu_env(self):
        """_launch_workers passes CUDA_VISIBLE_DEVICES via gpu_env for each dp_rank."""
        args = _make_args(data_parallel_size=2, tp_size=2, connection_mode="grpc")
        orch = ServeOrchestrator("sglang", args)

        launched_envs = []

        def capture_launch(a, host, port, env):
            launched_envs.append(env)
            mock_proc = MagicMock()
            mock_proc.pid = 1000 + port
            return mock_proc

        with patch("smg.serve._find_available_ports", return_value=[31000, 31003]):
            with patch.object(orch.launcher, "launch", side_effect=capture_launch):
                orch._launch_workers()

        assert len(launched_envs) == 2
        assert launched_envs[0]["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert launched_envs[1]["CUDA_VISIBLE_DEVICES"] == "2,3"
