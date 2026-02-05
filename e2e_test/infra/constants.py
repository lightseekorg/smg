"""Constants and enums for E2E test infrastructure."""

from enum import StrEnum


class ConnectionMode(StrEnum):
    """Worker connection protocol."""

    HTTP = "http"
    GRPC = "grpc"


class WorkerType(StrEnum):
    """Worker specialization type."""

    REGULAR = "regular"
    PREFILL = "prefill"
    DECODE = "decode"


class Runtime(StrEnum):
    """Inference runtime/backend."""

    SGLANG = "sglang"
    VLLM = "vllm"
    TRTLLM = "trtllm"
    OPENAI = "openai"
    XAI = "xai"
    GEMINI = "gemini"


# Convenience sets
LOCAL_MODES = frozenset({ConnectionMode.HTTP, ConnectionMode.GRPC})
LOCAL_RUNTIMES = frozenset({Runtime.SGLANG, Runtime.VLLM, Runtime.TRTLLM})
CLOUD_RUNTIMES = frozenset({Runtime.OPENAI, Runtime.XAI, Runtime.GEMINI})

# Fixture parameter names (used in @pytest.mark.parametrize)
PARAM_SETUP_BACKEND = "setup_backend"
PARAM_BACKEND_ROUTER = "backend_router"
PARAM_MODEL = "model"

# Default model
DEFAULT_MODEL = "llama-8b"

# Default runtime for gRPC tests
DEFAULT_RUNTIME = "sglang"

# Environment variable names
ENV_MODELS = "E2E_MODELS"
ENV_BACKENDS = "E2E_BACKENDS"
ENV_MODEL = "E2E_MODEL"
ENV_RUNTIME = "E2E_RUNTIME"  # Runtime for gRPC tests: "sglang", "vllm", or "trtllm"
ENV_STARTUP_TIMEOUT = "E2E_STARTUP_TIMEOUT"
ENV_SKIP_MODEL_POOL = "SKIP_MODEL_POOL"
ENV_SKIP_BACKEND_SETUP = "SKIP_BACKEND_SETUP"


# Runtime detection helpers
_RUNTIME_CACHE = None


def get_runtime() -> str:
    """Get the current test runtime (sglang or vllm).

    Returns:
        Runtime name from E2E_RUNTIME environment variable, defaults to "sglang".
    """
    global _RUNTIME_CACHE
    if _RUNTIME_CACHE is None:
        import os

        _RUNTIME_CACHE = os.environ.get(ENV_RUNTIME, DEFAULT_RUNTIME)
    return _RUNTIME_CACHE


def is_vllm() -> bool:
    """Check if tests are running with vLLM runtime.

    Returns:
        True if E2E_RUNTIME is "vllm", False otherwise.
    """
    return get_runtime() == "vllm"


def is_sglang() -> bool:
    """Check if tests are running with SGLang runtime.

    Returns:
        True if E2E_RUNTIME is "sglang", False otherwise.
    """
    return get_runtime() == "sglang"


def is_trtllm() -> bool:
    """Check if tests are running with TensorRT-LLM runtime.

    Returns:
        True if E2E_RUNTIME is "trtllm", False otherwise.
    """
    return get_runtime() == "trtllm"


# Runtime display labels
RUNTIME_LABELS = {
    "sglang": "SGLang",
    "vllm": "vLLM",
    "trtllm": "TensorRT-LLM",
}

ENV_SHOW_ROUTER_LOGS = "SHOW_ROUTER_LOGS"
ENV_SHOW_WORKER_LOGS = "SHOW_WORKER_LOGS"

# Network
DEFAULT_HOST = "127.0.0.1"

# Timeouts (seconds)
DEFAULT_STARTUP_TIMEOUT = 300
DEFAULT_ROUTER_TIMEOUT = 60
HEALTH_CHECK_INTERVAL = 2  # Check every 2s (was 5s)

# Model loading configuration
INITIAL_GRACE_PERIOD = 30  # Wait before first health check (model loading time)
LAUNCH_STAGGER_DELAY = 10  # Delay between launching multiple workers (avoid I/O contention)

# Retry configuration
MAX_RETRY_ATTEMPTS = 6  # Max retries with exponential backoff (total ~63s: 1+2+4+8+16+32)

# Display formatting
LOG_SEPARATOR_WIDTH = 60  # Width for log separator lines (e.g., "="*60)
