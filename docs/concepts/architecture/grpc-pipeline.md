---
title: gRPC Pipeline
---

# gRPC Pipeline

When workers communicate via gRPC, SMG becomes a complete OpenAI-compatible server with a sophisticated request processing pipeline for reasoning extraction, tool call parsing, and MCP execution.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-comment-processing: Chat Templates

Apply model-specific chat templates with full Jinja2 support for all major model families.

</div>

<div class="card" markdown>

### :material-memory: Tokenization Caching

Two-level tokenization cache reduces CPU overhead by 60-90% for repeated content.

</div>

<div class="card" markdown>

### :material-brain: Reasoning Extraction

Extract chain-of-thought content from thinking models (DeepSeek-R1, Qwen3, etc.).

</div>

<div class="card" markdown>

### :material-function: Tool Call Parsing

Parse function calls and execute MCP tools with automatic result injection.

</div>

</div>

---

## Pipeline Architecture

<div class="architecture-diagram">
  <img src="../../../assets/images/grpc-pipeline.svg" alt="gRPC Pipeline Architecture">
</div>

<div class="grid" markdown>

<div class="card" markdown>

### :material-lightning-bolt: gRPC Mode

**Gateway = Full Server**

SMG handles tokenization, chat templates, tool parsing, MCP loops, and detokenization. Workers run raw inference.

</div>

<div class="card" markdown>

### :material-swap-horizontal: HTTP Mode

**Gateway = Smart Proxy**

SMG handles routing, load balancing, and failover. Workers run full OpenAI-compatible servers.

</div>

</div>

### Responsibility Comparison

| Capability | gRPC Mode (Gateway) | HTTP Mode (Worker) |
|------------|--------------------|--------------------|
| Chat template | Gateway | Worker |
| Tokenization | Gateway (cached) | Worker |
| Load balancing | Token-aware | Request count |
| Reasoning extraction | Gateway | Worker |
| Tool call parsing | Gateway | Worker |
| MCP execution | Gateway | N/A |

---

## Reasoning Parsers

Reasoning parsers extract chain-of-thought content from model outputs. Essential for models that produce thinking tokens before their final response.

### Configuration

| Option | `--reasoning-parser` |
|--------|---------------------|
| Environment | `SMG_REASONING_PARSER` |
| Default | Auto-detected from model name |

### Supported Parsers

<div class="grid" markdown>

<div class="card" markdown>

**DeepSeek-R1**

- Pattern: `*deepseek-r1*`
- Initial state: In reasoning
- Tokens: `</think>` to exit

```bash
smg --reasoning-parser deepseek_r1
```

</div>

<div class="card" markdown>

**Qwen3**

- Pattern: `*qwen3*`
- Initial state: Not in reasoning
- Tokens: `<think>` / `</think>`

```bash
smg --reasoning-parser qwen3
```

</div>

<div class="card" markdown>

**Kimi**

- Pattern: `*kimi*`
- Initial state: Not in reasoning
- Tokens: Unicode markers

```bash
smg --reasoning-parser kimi
```

</div>

<div class="card" markdown>

**GLM-4.5**

- Pattern: `*glm45*`, `*glm47*`
- Initial state: Not in reasoning
- Tokens: `<think>` / `</think>`

```bash
smg --reasoning-parser glm45
```

</div>

</div>

### Complete Parser Reference

| Parser | Model Pattern | Initial State | Tokens |
|--------|--------------|---------------|--------|
| `deepseek_r1` | `*deepseek-r1*` | In reasoning | `</think>` |
| `qwen3` | `*qwen3*` | Not in reasoning | `<think>` / `</think>` |
| `qwen3_thinking` | `*qwen-thinking*` | In reasoning | `<think>` / `</think>` |
| `kimi` | `*kimi*` | Not in reasoning | Unicode markers |
| `glm45` | `*glm45*`, `*glm47*` | Not in reasoning | `<think>` / `</think>` |
| `step3` | `*step3*` | In reasoning | `<think>` / `</think>` |
| `minimax` | `*minimax*`, `*mm-m2*` | In reasoning | `<think>` appended |

### Output Format

When `separate_reasoning: true` is set in the request:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The answer is 42.",
      "reasoning_content": "Let me think step by step..."
    }
  }]
}
```

---

## Tool Call Parsers

Tool call parsers extract function calls from model output and validate arguments against schemas.

### Configuration

| Option | `--tool-call-parser` |
|--------|----------------|
| Environment | `SMG_TOOL_CALL_PARSER` |
| Default | Auto-detected from model name |

### Supported Parsers

<div class="grid" markdown>

<div class="card" markdown>

**Llama**

Native Llama 3.2 function calling format.

```json
<|python_tag|>{"name": "get_weather", "parameters": {"location": "NYC"}}
```

</div>

<div class="card" markdown>

**DeepSeek**

DeepSeek V3 tool format.

```xml
<tool_call>
get_weather(location="NYC")
</tool_call>
```

</div>

<div class="card" markdown>

**Qwen**

Qwen model JSON tool calling format.

```json
{"name": "get_weather", "arguments": {"location": "NYC"}}
```

</div>

<div class="card" markdown>

**Qwen Coder**

Qwen Coder XML format with parameter tags.

```xml
<tool_call><function=get_weather><parameter=location>NYC</parameter></function></tool_call>
```

</div>

</div>

### Complete Parser Reference

| Parser | Model Pattern | Format |
|--------|--------------|--------|
| `passthrough` | Default fallback | No parsing (returns text unchanged) |
| `json` | `gpt-*`, `claude-*`, `gemini-*` | Standard JSON function calls |
| `mistral` | `mistral-*`, `mixtral-*` | Mistral-specific format |
| `qwen` | `qwen*`, `Qwen*` | JSON tool calls |
| `qwen_coder` | `Qwen*-Coder*`, `qwen*-coder*` | XML with parameter tags |
| `pythonic` | `llama-4*`, `deepseek-*` | Python-style function syntax |
| `llama` | `llama-3.2*` | Python tag with JSON |
| `deepseek` | `deepseek-v3*` | XML with function syntax |
| `glm45_moe` | `glm-4.5*`, `glm-4.6*` | GLM 4.5/4.6 MoE format |
| `glm47_moe` | `glm-4.7*` | GLM 4.7 MoE format |
| `step3` | `step3*`, `Step-3*` | Step-3 model format |
| `kimik2` | `kimi-k2*`, `Kimi-K2*` | Kimi K2 model format |
| `minimax_m2` | `minimax*`, `MiniMax*` | MiniMax M2 model format |

### Tool Execution Flow

1. **Parse**: Extract tool calls from model output
2. **Validate**: Check arguments against tool schema
3. **Execute**: Run MCP tools or return to client
4. **Inject**: Add tool results back to conversation
5. **Continue**: Resume generation if needed

---

## Configuration

### Parser CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--reasoning-parser` | Auto | Reasoning parser type to use |
| `--tool-call-parser` | Auto | Tool call parser type to use |
| `--mcp-config-path` | None | Path to MCP server configuration file |

### MCP Integration

When MCP is configured, tool calls can be executed automatically:

```bash
smg \
  --mcp-config-path /path/to/mcp.json \
  --tool-call-parser llama
```

See the [MCP Guide](../extensibility/mcp.md) for detailed configuration.

---

## Recommended Configurations

<div class="grid" markdown>

<div class="card" markdown>

### :material-brain: Thinking Model

DeepSeek-R1 with reasoning extraction.

```bash
smg \
  --model-path deepseek-ai/DeepSeek-R1 \
  --reasoning-parser deepseek_r1 \
  --grpc-workers grpc://worker1:50051
```

</div>

<div class="card" markdown>

### :material-function: Tool Calling Model

Llama with MCP tool execution.

```bash
smg \
  --model-path meta-llama/Llama-3.2-70B-Instruct \
  --tool-call-parser llama \
  --mcp-config-path /config/mcp.json
```

</div>

<div class="card" markdown>

### :material-all-inclusive: Full Pipeline

Complete configuration with all features.

```bash
smg \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen \
  --mcp-config-path /config/mcp.json \
  --tokenizer-cache-enable-l0 \
  --tokenizer-cache-enable-l1 \
  --grpc-workers grpc://worker:50051
```

</div>

</div>

---

## Monitoring

### Pipeline Metrics

| Metric | Description |
|--------|-------------|
| `smg_pipeline_stage_duration_seconds` | Time spent in each pipeline stage |
| `smg_reasoning_extractions_total` | Reasoning tokens extracted |
| `smg_tool_calls_total` | Tool calls parsed by type |
| `smg_tool_execution_duration_seconds` | Tool execution time |
| `smg_mcp_tool_calls_total` | MCP tool invocations |

### Debug Logging

```bash
# Enable pipeline debug logging
RUST_LOG=smg::pipeline=debug smg ...

# Enable parser debug logging
RUST_LOG=smg::parsers=debug smg ...
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| Reasoning not extracted | Wrong parser | Check model and parser match |
| Tool calls not parsed | Format mismatch | Verify tool parser selection |
| MCP tools timeout | Slow tool execution | Check MCP server configuration |
| Empty reasoning_content | Model not thinking | Enable `separate_reasoning: true` in request |

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-memory: Tokenizer Caching

Learn about two-level tokenizer caching for performance.

[Tokenizer Caching →](../performance/tokenizer-caching.md)

</div>

<div class="card" markdown>

### :material-puzzle: MCP Integration

Configure Model Context Protocol servers for tool execution.

[MCP →](../extensibility/mcp.md)

</div>

<div class="card" markdown>

### :material-cached: Cache-Aware Routing

Maximize KV cache hits with prefix-based routing.

[Cache-Aware Routing →](../routing/cache-aware.md)

</div>

</div>
