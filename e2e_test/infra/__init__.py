"""Infrastructure for parallel GPU test execution."""

from .constants import (  # Enums; Convenience sets; Fixture parameters; Defaults; Environment variables
    BRAVE_MCP_HOST,
    BRAVE_MCP_PORT,
    BRAVE_MCP_URL,
    CLOUD_RUNTIMES,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_ROUTER_TIMEOUT,
    DEFAULT_RUNTIME,
    DEFAULT_STARTUP_TIMEOUT,
    ENV_BACKENDS,
    ENV_MODEL,
    ENV_MODELS,
    ENV_RUNTIME,
    ENV_SHOW_ROUTER_LOGS,
    ENV_SHOW_WORKER_LOGS,
    ENV_SKIP_BACKEND_SETUP,
    ENV_SKIP_MODEL_POOL,
    ENV_STARTUP_TIMEOUT,
    HEALTH_CHECK_INTERVAL,
    LOCAL_MODES,
    LOCAL_RUNTIMES,
    LOG_SEPARATOR_WIDTH,
    MAX_RETRY_ATTEMPTS,
    PARAM_BACKEND_ROUTER,
    PARAM_MODEL,
    PARAM_SETUP_BACKEND,
    RUNTIME_LABELS,
    ConnectionMode,
    Runtime,
    WorkerType,
    get_runtime,
    is_sglang,
    is_trtllm,
    is_vllm,
)
from .gateway import Gateway, WorkerInfo, launch_cloud_gateway
from .gpu_monitor import GPUMonitor
from .gpu_monitor import should_monitor as should_monitor_gpu
from .model_specs import (  # Default model paths; Model groups
    CHAT_MODELS,
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_ENABLE_THINKING_MODEL_PATH,
    DEFAULT_GPT_OSS_MODEL_PATH,
    DEFAULT_MISTRAL_FUNCTION_CALLING_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_QWEN_FUNCTION_CALLING_MODEL_PATH,
    DEFAULT_REASONING_MODEL_PATH,
    DEFAULT_SMALL_MODEL_PATH,
    EMBEDDING_MODELS,
    FUNCTION_CALLING_MODELS,
    MODEL_SPECS,
    REASONING_MODELS,
    THIRD_PARTY_MODELS,
)
from .process_utils import (
    detect_ib_device,
    get_open_port,
    kill_process_tree,
    release_port,
    terminate_process,
    wait_for_health,
    wait_for_workers_ready,
)
from .run_eval import run_eval
from .worker import Worker, start_workers, stop_workers

__all__ = [
    # Enums
    "ConnectionMode",
    "WorkerType",
    "Runtime",
    # Convenience sets
    "LOCAL_MODES",
    "LOCAL_RUNTIMES",
    "CLOUD_RUNTIMES",
    # Fixture params
    "PARAM_SETUP_BACKEND",
    "PARAM_BACKEND_ROUTER",
    "PARAM_MODEL",
    # Defaults
    "DEFAULT_MODEL",
    "DEFAULT_HOST",
    "BRAVE_MCP_HOST",
    "BRAVE_MCP_PORT",
    "BRAVE_MCP_URL",
    "DEFAULT_RUNTIME",
    "DEFAULT_STARTUP_TIMEOUT",
    "DEFAULT_ROUTER_TIMEOUT",
    "HEALTH_CHECK_INTERVAL",
    "MAX_RETRY_ATTEMPTS",
    "LOG_SEPARATOR_WIDTH",
    "RUNTIME_LABELS",
    # Env vars
    "ENV_MODELS",
    "ENV_BACKENDS",
    "ENV_MODEL",
    "ENV_RUNTIME",
    "ENV_STARTUP_TIMEOUT",
    "ENV_SKIP_MODEL_POOL",
    "ENV_SKIP_BACKEND_SETUP",
    "ENV_SHOW_ROUTER_LOGS",
    "ENV_SHOW_WORKER_LOGS",
    # Runtime helpers
    "get_runtime",
    "is_vllm",
    "is_sglang",
    "is_trtllm",
    # Port utilities
    "get_open_port",
    "release_port",
    # Process utilities
    "kill_process_tree",
    "terminate_process",
    "wait_for_health",
    "wait_for_workers_ready",
    "detect_ib_device",
    # GPU monitoring
    "GPUMonitor",
    "should_monitor_gpu",
    # Worker management
    "Worker",
    "start_workers",
    "stop_workers",
    "MODEL_SPECS",
    # Gateway
    "Gateway",
    "WorkerInfo",
    "launch_cloud_gateway",
    # Default model paths
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SMALL_MODEL_PATH",
    "DEFAULT_REASONING_MODEL_PATH",
    "DEFAULT_ENABLE_THINKING_MODEL_PATH",
    "DEFAULT_QWEN_FUNCTION_CALLING_MODEL_PATH",
    "DEFAULT_MISTRAL_FUNCTION_CALLING_MODEL_PATH",
    "DEFAULT_GPT_OSS_MODEL_PATH",
    "DEFAULT_EMBEDDING_MODEL_PATH",
    # Model groups
    "CHAT_MODELS",
    "EMBEDDING_MODELS",
    "REASONING_MODELS",
    "FUNCTION_CALLING_MODELS",
    "THIRD_PARTY_MODELS",
    # Evaluation
    "run_eval",
]
