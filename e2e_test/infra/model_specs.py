"""Model specifications for E2E tests.

Each model spec defines:
- model: HuggingFace model path or local path
- memory_gb: Estimated GPU memory required
- tp: Tensor parallelism size (number of GPUs needed)
- features: List of features this model supports (for test filtering)
"""

from __future__ import annotations

import json
import os

# Environment variable for local model paths (CI uses local copies for speed)
ROUTER_LOCAL_MODEL_PATH = os.environ.get("ROUTER_LOCAL_MODEL_PATH", "")


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
        "memory_gb": 16,
        "tp": 1,
        "features": ["chat", "streaming", "function_calling"],
    },
    # Small model for quick tests
    "meta-llama/Llama-3.2-1B-Instruct": {
        "model": _resolve_model_path("meta-llama/Llama-3.2-1B-Instruct"),
        "memory_gb": 4,
        "tp": 1,
        "features": ["chat", "streaming", "tool_choice"],
    },
    # Function calling specialist
    "Qwen/Qwen2.5-7B-Instruct": {
        "model": _resolve_model_path("Qwen/Qwen2.5-7B-Instruct"),
        "memory_gb": 14,
        "tp": 1,
        "features": ["chat", "streaming", "function_calling", "pythonic_tools"],
    },
    # Function calling specialist (larger, for Response API tests)
    "Qwen/Qwen2.5-14B-Instruct": {
        "model": _resolve_model_path("Qwen/Qwen2.5-14B-Instruct"),
        "memory_gb": 28,
        "tp": 2,
        "features": ["chat", "streaming", "function_calling", "pythonic_tools"],
        "worker_args": ["--context-length=16384"],  # Faster startup, prevents memory issues
    },
    # Reasoning model
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "model": _resolve_model_path("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
        "memory_gb": 14,
        "tp": 1,
        "features": ["chat", "streaming", "reasoning"],
    },
    # Thinking/reasoning model (larger)
    "Qwen/Qwen3-30B-A3B": {
        "model": _resolve_model_path("Qwen/Qwen3-30B-A3B"),
        "memory_gb": 60,
        "tp": 4,
        "features": ["chat", "streaming", "thinking", "reasoning"],
    },
    # Mistral for function calling
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "model": _resolve_model_path("mistralai/Mistral-7B-Instruct-v0.3"),
        "memory_gb": 14,
        "tp": 1,
        "features": ["chat", "streaming", "function_calling"],
    },
    # Embedding model
    "embedding": {
        "model": _resolve_model_path("intfloat/e5-mistral-7b-instruct"),
        "memory_gb": 14,
        "tp": 1,
        "features": ["embedding"],
    },
    # GPT-OSS model (Harmony)
    "openai/gpt-oss-20b": {
        "model": _resolve_model_path("openai/gpt-oss-20b"),
        "memory_gb": 40,
        "tp": 2,
        "features": ["chat", "streaming", "reasoning", "harmony"],
    },
    # Llama-4-Maverick (17B with 128 experts, FP8) - Nightly benchmarks
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": {
        "model": _resolve_model_path("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"),
        "memory_gb": 34,  # ~1 byte per parameter for FP8 quantized 17B params
        "tp": 8,  # Tensor parallelism across 8 GPUs
        "features": ["chat", "streaming", "function_calling", "moe"],
        "worker_args": [
            "--trust-remote-code",
            "--context-length=163840",  # 160K context length (SGLang)
            "--prefill-attention-backend=flashinfer",  # MLA attention backend
            "--decode-attention-backend=flashinfer",  # MLA attention backend
            "--flashinfer-mla-disable-ragged",  # Disable ragged attention for MLA
            "--mem-fraction-static=0.9",  # 90% GPU memory for static allocation
            "--cuda-graph-max-bs=256",  # CUDA graph batch size optimization
        ],
        "vllm_args": [
            "--trust-remote-code",
            "--max-model-len=163840",  # 160K context length (vLLM)
        ],
    },
}


def get_models_with_feature(feature: str) -> list[str]:
    """Get list of model IDs that support a specific feature."""
    return [
        model_id for model_id, spec in MODEL_SPECS.items() if feature in spec.get("features", [])
    ]


def get_model_spec(model_id: str) -> dict:
    """Get spec for a specific model, raising KeyError if not found."""
    if model_id not in MODEL_SPECS:
        raise KeyError(f"Unknown model: {model_id}. Available: {list(MODEL_SPECS.keys())}")
    spec = dict(MODEL_SPECS[model_id])
    tp_overrides_json = os.environ.get("E2E_MODEL_TP_OVERRIDES")
    if tp_overrides_json:
        try:
            tp_overrides = json.loads(tp_overrides_json)
            if isinstance(tp_overrides, dict):
                override = tp_overrides.get(model_id)
                if isinstance(override, int) and override > 0:
                    spec["tp"] = override
        except json.JSONDecodeError:
            # Ignore malformed override config and fall back to canonical specs.
            pass
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
DEFAULT_EMBEDDING_MODEL_PATH = MODEL_SPECS["embedding"]["model"]


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
        "model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
        "client_type": "anthropic",
    },
}
