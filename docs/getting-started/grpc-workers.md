---
title: gRPC Workers
---

# gRPC Workers

When workers connect via gRPC instead of HTTP, SMG becomes a full OpenAI-compatible server — handling tokenization, chat templates, reasoning extraction, and tool calling at the gateway level. Workers run raw inference only.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- A gRPC-capable inference worker (vLLM with gRPC entrypoint)
- Access to the model weights or a HuggingFace model path (for tokenizer loading)

</div>

---

## What gRPC Mode Enables

| Capability | HTTP Mode (worker handles) | gRPC Mode (gateway handles) |
|------------|---------------------------|----------------------------|
| Chat templates | Worker | Gateway |
| Tokenization | Worker | Gateway (with caching) |
| Load balancing | Request-level | Token-aware |
| Reasoning extraction | Worker | Gateway |
| Tool call parsing | Worker | Gateway |
| MCP tool execution (Responses API) | N/A | Gateway |

In HTTP mode, SMG is a smart proxy — routing and failover only. In gRPC mode, SMG takes over the full request processing pipeline.

---

## Start a gRPC Worker

=== "SGLang"

    ```bash
    python -m sglang.launch_server \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 50051 \
      --grpc-mode
    ```

=== "vLLM"

    ```bash
    python -m vllm.entrypoints.grpc_server \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --host 0.0.0.0 \
      --port 50051
    ```

=== "TensorRT-LLM"

    ```bash
    python -m tensorrt_llm.commands.serve serve \
      meta-llama/Llama-3.1-8B-Instruct \
      --grpc \
      --host 0.0.0.0 \
      --port 50051 \
      --backend pytorch
    ```

---

## Connect SMG

Point SMG at the gRPC worker using `grpc://` URLs and provide `--model-path` so the gateway can load the tokenizer:

```bash
smg \
  --worker-urls grpc://localhost:50051 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

!!! note "`--model-path` is required"
    The gateway needs the tokenizer to apply chat templates, count tokens for load balancing, and parse tool calls. This can be a HuggingFace model ID or a local path.

The API is still OpenAI-compatible — clients send the same requests as with HTTP workers:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

---

## Multiple gRPC Workers

```bash
smg \
  --worker-urls grpc://worker1:50051 grpc://worker2:50052 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --policy round_robin
```

---

## Reasoning Extraction

For thinking models (DeepSeek-R1, Qwen3, etc.), SMG can extract chain-of-thought content into a separate field:

```bash
smg \
  --worker-urls grpc://worker:50051 \
  --model-path deepseek-ai/DeepSeek-R1 \
  --reasoning-parser deepseek_r1
```

The parser is auto-detected from the model name by default. Override with `--reasoning-parser` if needed.

Request with `separate_reasoning: true`:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [{"role": "user", "content": "What is 25 * 37?"}],
    "separate_reasoning": true
  }'
```

Response includes both fields:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "925",
      "reasoning_content": "Let me calculate 25 * 37 step by step..."
    }
  }]
}
```

### Supported Reasoning Parsers

Auto-detected from the model name. Override with `--reasoning-parser` if needed.

| Parser | Models |
|--------|--------|
| `deepseek_r1` | DeepSeek-R1 |
| `qwen3` | Qwen3, Nemotron Nano |
| `qwen_thinking` | Qwen3-Thinking |
| `kimi` | Kimi |
| `glm45` | GLM-4.5, GLM-4.7 |
| `step3` | Step-3 |
| `minimax` | MiniMax, MiniMax-M2 |
| `cohere_cmd` | Command-R, Command-A, C4AI |

---

## Tool Calling

In gRPC mode, SMG parses function calls from model output:

```bash
smg \
  --worker-urls grpc://worker:50051 \
  --model-path meta-llama/Llama-3.2-70B-Instruct \
  --tool-call-parser llama
```

For MCP tool execution in Responses API, see the dedicated guide:

```bash
# See:
#   Getting Started → MCP in Responses API
#   /v1/responses + --mcp-config-path
```

### Supported Tool Call Parsers

Auto-detected from the model name. Override with `--tool-call-parser` if needed.

| Parser | Models |
|--------|--------|
| `json` | GPT-4/4o, Claude, Gemini, Gemma, Llama (generic) |
| `llama` | Llama 3.2 |
| `pythonic` | Llama 4, DeepSeek (generic) |
| `deepseek` | DeepSeek-V3 |
| `mistral` | Mistral, Mixtral |
| `qwen` | Qwen |
| `qwen_coder` | Qwen3-Coder, Qwen2.5-Coder |
| `glm45_moe` | GLM-4.5, GLM-4.6 |
| `glm47_moe` | GLM-4.7 |
| `step3` | Step-3 |
| `kimik2` | Kimi-K2 |
| `minimax_m2` | MiniMax |
| `cohere` | Command-R, Command-A, C4AI |

---

## HTTP vs gRPC: When to Use Which

| Use Case | Recommended Mode |
|----------|-----------------|
| Workers already run OpenAI servers (SGLang, vLLM HTTP) | HTTP |
| You need gateway-level tool parsing or Responses MCP | gRPC |
| You want token-aware load balancing | gRPC |
| You use thinking models and want reasoning extraction | gRPC |
| Simplest possible setup | HTTP |

---

## Next Steps

- [gRPC Pipeline Concepts](../concepts/architecture/grpc-pipeline.md) — Full pipeline architecture, all supported parsers
- [Tokenizer Caching](../concepts/performance/tokenizer-caching.md) — Two-level cache for reduced CPU overhead
- [MCP in Responses API](mcp.md) — Configure Model Context Protocol servers for `/v1/responses`
- [PD Disaggregation](pd-disaggregation.md) — Separate prefill and decode with gRPC workers
