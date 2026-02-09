---
title: Tokenizer Caching
---

# Tokenizer Caching

SMG provides a two-level tokenizer cache that reduces tokenization overhead for repeated content. In typical production workloads, this achieves 60-90% cache hit rates.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- Using gRPC workers (tokenization happens at the gateway)
- `--model-path` configured so SMG can load the tokenizer

</div>

---

## How It Works

| Cache Level | Strategy | Best For |
|-------------|----------|----------|
| **L0** (Exact Match) | Hash-based O(1) lookup for identical strings | Repeated system prompts, batch inference |
| **L1** (Prefix Match) | Boundary-aligned prefix matching, tokenizes only the suffix | Multi-turn conversations, growing contexts |

On a multi-turn conversation, L1 avoids re-tokenizing the entire history — only new messages are tokenized.

---

## Enable Caching

Both cache levels are disabled by default. Enable them with CLI flags:

### L0 Only (Exact Match)

Best for workloads with many identical prompts (system prompts, batch processing):

```bash
smg \
  --worker-urls grpc://worker:50051 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer-cache-enable-l0 \
  --tokenizer-cache-l0-max-entries 10000
```

### L0 + L1 (Exact + Prefix Match)

Best for multi-turn chat applications:

```bash
smg \
  --worker-urls grpc://worker:50051 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer-cache-enable-l0 \
  --tokenizer-cache-l0-max-entries 20000 \
  --tokenizer-cache-enable-l1 \
  --tokenizer-cache-l1-max-memory 104857600
```

---

## Configuration Reference

### L0 Cache

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tokenizer-cache-enable-l0` | `false` | Enable exact match cache |
| `--tokenizer-cache-l0-max-entries` | `10000` | Maximum number of cached entries |

Each entry uses ~2.2 KB of memory.

### L1 Cache

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tokenizer-cache-enable-l1` | `false` | Enable prefix match cache |
| `--tokenizer-cache-l1-max-memory` | `52428800` (50 MB) | Maximum memory in bytes |

---

## Memory Planning

### L0 Sizing

| Entries | Memory | Recommended For |
|---------|--------|-----------------|
| 1,000 | ~2.2 MB | Development, testing |
| 10,000 | ~22 MB | Standard production |
| 25,000 | ~55 MB | High-repetition workloads |
| 50,000 | ~110 MB | Large-scale deployments |

Set L0 entries to 1-2x the number of unique system prompt variants in your workload.

### L1 Sizing

| Memory | Recommended For |
|--------|-----------------|
| 25 MB | Memory-constrained environments |
| 50 MB | Standard deployments (default) |
| 100 MB | Multi-turn conversation heavy |
| 200 MB | Long context applications |

Estimate ~1 KB per active conversation context for L1 sizing.

---

## Recommended Configurations

=== "High-Throughput Chat"

    For workloads with repeated system prompts:

    ```bash
    smg \
      --worker-urls grpc://worker:50051 \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --tokenizer-cache-enable-l0 \
      --tokenizer-cache-l0-max-entries 50000
    ```

=== "Multi-Turn Conversations"

    For chat applications with growing conversation history:

    ```bash
    smg \
      --worker-urls grpc://worker:50051 \
      --model-path Qwen/Qwen2.5-7B-Instruct \
      --tokenizer-cache-enable-l0 \
      --tokenizer-cache-l0-max-entries 20000 \
      --tokenizer-cache-enable-l1 \
      --tokenizer-cache-l1-max-memory 104857600
    ```

=== "Memory-Constrained"

    Moderate benefit with minimal memory:

    ```bash
    smg \
      --worker-urls grpc://worker:50051 \
      --model-path meta-llama/Llama-3.1-8B-Instruct \
      --tokenizer-cache-enable-l0 \
      --tokenizer-cache-l0-max-entries 5000
    ```

---

## Next Steps

- [Tokenizer Caching Concepts](../concepts/performance/tokenizer-caching.md) — Cache architecture, special token boundaries, monitoring metrics, PromQL queries
- [gRPC Workers](grpc-workers.md) — Enable gateway-level tokenization with gRPC mode
- [Load Balancing](load-balancing.md) — Choose a routing policy (cache-aware routing uses tokenizer results)
