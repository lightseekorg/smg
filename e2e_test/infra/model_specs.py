"""Model specifications for E2E tests.

Each model spec defines:
- model: HuggingFace model path or local path
- tp: Tensor parallelism size (number of GPUs needed)
- features: List of features this model supports (for test filtering)
"""

from __future__ import annotations

import json
import os

# Environment variable for local model paths (CI uses local copies for speed)
ROUTER_LOCAL_MODEL_PATH = os.environ.get("ROUTER_LOCAL_MODEL_PATH", "")
# Nightly benchmarks skip --enforce-eager for performance measurement
_is_nightly = os.environ.get("E2E_NIGHTLY") == "1"


def _resolve_model_path(hf_path: str) -> str:
    """Resolve model path, preferring local path if available."""
    if ROUTER_LOCAL_MODEL_PATH:
        local_path = os.path.join(ROUTER_LOCAL_MODEL_PATH, hf_path)
        if os.path.exists(local_path):
            return local_path
    return hf_path


MODEL_SPECS: dict[str, dict] = {
    # Primary chat model - used for most tests
    "meta-llama/Llama-3.1-8B-Instruct": {
        "model": _resolve_model_path("meta-llama/Llama-3.1-8B-Instruct"),
        "tp": 1,
        "features": ["chat", "streaming", "function_calling"],
    },
    # Small model for quick tests
    "meta-llama/Llama-3.2-1B-Instruct": {
        "model": _resolve_model_path("meta-llama/Llama-3.2-1B-Instruct"),
        "tp": 1,
        "features": ["chat", "streaming", "tool_choice"],
    },
    # Function calling specialist
    "Qwen/Qwen2.5-7B-Instruct": {
        "model": _resolve_model_path("Qwen/Qwen2.5-7B-Instruct"),
        "tp": 1,
        "features": ["chat", "streaming", "function_calling", "pythonic_tools"],
    },
    # Function calling specialist (larger, for Response API tests)
    "Qwen/Qwen2.5-14B-Instruct": {
        "model": _resolve_model_path("Qwen/Qwen2.5-14B-Instruct"),
        "tp": 2,
        "features": ["chat", "streaming", "function_calling", "pythonic_tools"],
        "worker_args": ["--context-length=16384"],  # Faster startup, prevents memory issues
    },
    # Reasoning model
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "model": _resolve_model_path("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        "tp": 1,
        "features": ["chat", "streaming", "reasoning"],
    },
    # Thinking/reasoning model (larger)
    "Qwen/Qwen3-30B-A3B": {
        "model": _resolve_model_path("Qwen/Qwen3-30B-A3B"),
        "tp": 1,
        "features": ["chat", "streaming", "thinking", "reasoning"],
        "vllm_args": [] if _is_nightly else ["--enforce-eager"],
        "trtllm_extra_config": {"kv_cache_config": {"free_gpu_memory_fraction": 0.8}},
    },
    # Mistral for function calling
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "model": _resolve_model_path("mistralai/Mistral-7B-Instruct-v0.3"),
        "tp": 1,
        "features": ["chat", "streaming", "function_calling"],
        "worker_args": ["--constrained-json-disable-any-whitespace"],
        "vllm_args": [
            "--structured-outputs-config",
            '{"disable_any_whitespace": true, "backend": "xgrammar"}',
        ],
    },
    # Embedding model
    "intfloat/e5-mistral-7b-instruct": {
        "model": _resolve_model_path("intfloat/e5-mistral-7b-instruct"),
        "tp": 1,
        "features": ["embedding"],
    },
    # GPT-OSS models (Harmony)
    "openai/gpt-oss-20b": {
        "model": _resolve_model_path("openai/gpt-oss-20b"),
        "tp": 2,
        "features": ["chat", "streaming", "reasoning", "harmony"],
        "vllm_args": [
            "--structured-outputs-config",
            '{"enable_in_reasoning": true}',
        ],
    },
    "openai/gpt-oss-120b": {
        "model": _resolve_model_path("openai/gpt-oss-120b"),
        "tp": 4,
        "features": ["chat", "streaming", "reasoning", "harmony"],
        "vllm_args": [
            "--structured-outputs-config",
            '{"enable_in_reasoning": true}',
        ],
    },
    # MiniMax M2 - nightly benchmarks
    "minimaxai/minimax-m2": {
        "model": _resolve_model_path("minimaxai/minimax-m2"),
        "tp": 4,
        "features": ["chat", "streaming", "function_calling", "reasoning"],
        "worker_args": ["--trust-remote-code"],
        "vllm_args": ["--trust-remote-code"],
    },
    # Vision-language model for multimodal benchmarks (MMMU)
    "Qwen/Qwen3-VL-8B-Instruct": {
        "model": _resolve_model_path("Qwen/Qwen3-VL-8B-Instruct"),
        "tp": 1,
        "features": ["chat", "streaming", "multimodal"],
    },
    # Llama-4-Maverick (17B with 128 experts, FP8) - Nightly benchmarks
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "model": _resolve_model_path("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"),
        "tp": 8,  # Tensor parallelism across 8 GPUs
        "features": ["chat", "streaming", "function_calling", "moe"],
        "worker_args": [
            "--trust-remote-code",
            "--context-length=163840",  # 160K context length (SGLang)
            "--attention-backend=fa3",  # fa3 attention backend
            "--mem-fraction-static=0.82",  # 82% GPU memory for static allocation
        ],
        "vllm_args": [
            "--trust-remote-code",
            "--max-model-len=163840",  # 160K context length (vLLM)
            "--attention-backend=FLASHINFER",  # FLASHINFER attention backend
        ],
        "startup_timeout": 1200,  # Large MoE model may need extra download/load time
    },
    # Llama-4-Scout (17B with 16 experts) - Nightly benchmarks and Multimodal tests
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "model": _resolve_model_path("meta-llama/Llama-4-Scout-17B-16E-Instruct"),
        "tp": 4,
        "features": ["chat", "streaming", "function_calling", "multimodal", "moe"],
        "worker_args": [
            "--context-length=196608",
            "--attention-backend=fa3",
            "--cuda-graph-max-bs=256",
            "--max-running-requests=300",
            "--mem-fraction-static=0.85",
        ],
        "vllm_args": [
            "--max-model-len=196608",
        ],
        "startup_timeout": 1200,  # Large MoE model may need extra download/load time
    },
    # Llama-3.3-70B - Nightly benchmarks
    "meta-llama/Llama-3.3-70B-Instruct": {
        "model": _resolve_model_path("meta-llama/Llama-3.3-70B-Instruct"),
        "tp": 4,
        "features": ["chat", "streaming", "function_calling"],
        "worker_args": [
            "--mem-fraction-static=0.9",
        ],
        "vllm_args": [
            "--max-model-len=131072",
            "--gpu-memory-utilization=0.9",
            "--enable-chunked-prefill",
        ],
    },
    # Llama-3.3-70B FP8 - Nightly benchmarks
    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic": {
        "model": _resolve_model_path("RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"),
        "tp": 4,
        "features": ["chat", "streaming", "function_calling"],
        "worker_args": [
            "--mem-fraction-static=0.9",
        ],
        "vllm_args": [
            "--max-model-len=131072",
            "--gpu-memory-utilization=0.9",
            "--enable-chunked-prefill",
        ],
    },
}


def get_models_with_feature(feature: str) -> list[str]:
    """Get list of model IDs that support a specific feature."""
    return [
        model_id for model_id, spec in MODEL_SPECS.items() if feature in spec.get("features", [])
    ]


def _parse_tp_overrides() -> dict | None:
    """Parse E2E_MODEL_TP_OVERRIDES env var once at import time."""
    raw = os.environ.get("E2E_MODEL_TP_OVERRIDES")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


_TP_OVERRIDES = _parse_tp_overrides()


def get_model_spec(model_id: str) -> dict:
    """Get spec for a specific model, raising KeyError if not found."""
    if model_id not in MODEL_SPECS:
        raise KeyError(f"Unknown model: {model_id}. Available: {list(MODEL_SPECS.keys())}")
    spec = dict(MODEL_SPECS[model_id])
    if _TP_OVERRIDES is not None:
        override = _TP_OVERRIDES.get(model_id)
        if isinstance(override, int) and override > 0:
            spec["tp"] = override
    return spec


# Convenience groupings for test parametrization
CHAT_MODELS = get_models_with_feature("chat")
EMBEDDING_MODELS = get_models_with_feature("embedding")
REASONING_MODELS = get_models_with_feature("reasoning")
FUNCTION_CALLING_MODELS = get_models_with_feature("function_calling")


# =============================================================================
# Default model path constants (for backward compatibility with existing tests)
# =============================================================================

DEFAULT_MODEL_PATH = MODEL_SPECS["meta-llama/Llama-3.1-8B-Instruct"]["model"]
DEFAULT_SMALL_MODEL_PATH = MODEL_SPECS["meta-llama/Llama-3.2-1B-Instruct"]["model"]
DEFAULT_REASONING_MODEL_PATH = MODEL_SPECS["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"]["model"]
DEFAULT_ENABLE_THINKING_MODEL_PATH = MODEL_SPECS["Qwen/Qwen3-30B-A3B"]["model"]
DEFAULT_QWEN_FUNCTION_CALLING_MODEL_PATH = MODEL_SPECS["Qwen/Qwen2.5-7B-Instruct"]["model"]
DEFAULT_MISTRAL_FUNCTION_CALLING_MODEL_PATH = MODEL_SPECS["mistralai/Mistral-7B-Instruct-v0.3"][
    "model"
]
DEFAULT_GPT_OSS_MODEL_PATH = MODEL_SPECS["openai/gpt-oss-20b"]["model"]
DEFAULT_EMBEDDING_MODEL_PATH = MODEL_SPECS["intfloat/e5-mistral-7b-instruct"]["model"]


# =============================================================================
# Third-party model configurations (cloud APIs)
# =============================================================================

THIRD_PARTY_MODELS: dict[str, dict] = {
    "openai": {
        "description": "OpenAI API",
        "model": "gpt-5-nano",
        "api_key_env": "OPENAI_API_KEY",
    },
    "xai": {
        "description": "xAI API",
        "model": "grok-4-fast",
        "api_key_env": "XAI_API_KEY",
    },
    "anthropic": {
        "description": "Anthropic API",
        "model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "client_type": "anthropic",
    },
}
