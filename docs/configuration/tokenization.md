---
title: Tokenization Configuration
---

# Tokenization Configuration

SMG includes a built-in tokenizer for efficient text processing, chat template rendering, and token counting. This guide covers how to configure the tokenizer, chat templates, and caching for optimal performance.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-comment-text: Chat Template Rendering

Convert chat messages into model-specific prompt formats. Supports Jinja2 templates for Llama, Qwen, Mistral, and more.

</div>

<div class="card" markdown>

### :material-counter: Token Counting

Calculate input/output token usage for accurate metrics, rate limiting, and billing. Cached for efficiency.

</div>

<div class="card" markdown>

### :material-memory: Two-Level Caching

L0 exact-match and L1 prefix-match caches reduce tokenization overhead by 60-90% for repeated content.

</div>

<div class="card" markdown>

### :material-hash: Prefix Hashing

Support cache-aware routing policies that hash prompt prefixes for optimal worker selection.

</div>

</div>

---

## Cache Architecture

<div class="architecture-diagram">
  <img src="../../assets/images/tokenization-cache.svg" alt="Tokenization Cache Architecture">
</div>

<div class="grid" markdown>

<div class="card" markdown>

### :material-lightning-bolt: L0 Cache (Exact Match)

**Router-level cache** storing complete tokenization results for exact string matches.

- Hash-based O(1) lookup
- ~2.2KB per entry
- 60-90% hit rate for repeated prompts
- Simple eviction policy (removes oldest entry when full)

</div>

<div class="card" markdown>

### :material-layers: L1 Cache (Prefix Match)

**Worker-level cache** storing tokens at special token boundaries for prefix reuse.

- Tokenize only the suffix on hit
- Cross-request deduplication
- Memory-bounded (configurable)
- Automatic boundary detection

</div>

</div>

---

## Model & Tokenizer Paths

### `--model-path`

HuggingFace model ID or local path to load the tokenizer from.

| Option | `--model-path` |
|--------|----------------|
| Default | None |

**Usage**:

```bash
# HuggingFace model ID (downloads automatically)
smg --model-path meta-llama/Llama-3.1-8B-Instruct ...

# Local path to model directory
smg --model-path /models/llama-3.1-8b-instruct ...

# Local path to tokenizer.json file
smg --model-path /models/llama-3.1-8b-instruct/tokenizer.json ...
```

When pointing to a directory, SMG automatically searches for:

1. `tokenizer.json` (HuggingFace fast tokenizer format)
2. `tokenizer_config.json` (fallback)
3. `vocab.json` (fallback)

### `--tokenizer-path`

Explicit path to a tokenizer file. Overrides `--model-path` for tokenizer loading.

| Option | `--tokenizer-path` |
|--------|-------------------|
| Default | None |

**When to use**:

- When the tokenizer is stored separately from the model
- When using a custom tokenizer with a standard model
- When the model directory structure is non-standard

```bash
# Use model for metadata but separate tokenizer
smg \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer-path /custom/tokenizers/llama3-tokenizer.json \
  ...
```

---

## Chat Templates

Chat templates convert structured messages (system, user, assistant roles) into the prompt format expected by specific models. SMG uses Jinja2 templates, the same format used by HuggingFace Transformers.

### `--chat-template`

Path to a Jinja2 chat template file.

| Option | `--chat-template` |
|--------|-------------------|
| Default | Auto-discovered from model |

**Template discovery priority**:

1. Explicit `--chat-template` path (highest priority)
2. `chat_template.json` in model directory
3. `chat_template.jinja` in model directory
4. Any `.jinja` file in model directory
5. `chat_template` field in `tokenizer_config.json`

### Template Variables

Chat templates use Jinja2 syntax with access to:

| Variable | Description |
|----------|-------------|
| `messages` | Array of message objects with `role` and `content` |
| `add_generation_prompt` | Boolean to add assistant prompt prefix |
| `tools` | Optional array of tool definitions |
| `documents` | Optional array of document context |

### Template Examples

<div class="grid" markdown>

<div class="card" markdown>

**ChatML** (Qwen, Yi)

```jinja
{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{% endif %}
```

</div>

<div class="card" markdown>

**Llama 3**

```jinja
<|begin_of_text|>{% for message in messages %}
<|start_header_id|>{{ message.role }}<|end_header_id|>

{{ message.content }}<|eot_id|>
{% endfor %}
{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|}

{% endif %}
```

</div>

</div>

### Template File Formats

**`.jinja` files**: Plain Jinja2 template text

```bash
smg --chat-template /templates/chatml.jinja ...
```

**`.json` files**: JSON containing the template string

```json
{
  "chat_template": "{%- for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}"
}
```

---

## Tokenizer Caching

### L0 Cache Configuration

The L0 cache stores complete tokenization results for exact string matches.

<div class="grid" markdown>

<div class="card" markdown>

#### `--tokenizer-cache-enable-l0`

Enable the L0 exact match cache.

| Option | `--tokenizer-cache-enable-l0` |
|--------|-------------------------------|
| Default | `false` |

</div>

<div class="card" markdown>

#### `--tokenizer-cache-l0-max-entries`

Maximum number of entries in the L0 cache.

| Option | `--tokenizer-cache-l0-max-entries` |
|--------|-----------------------------------|
| Default | `10000` |

</div>

</div>

**Memory estimation**: Each entry uses approximately 2.2KB.

| Entries | Estimated Memory |
|---------|------------------|
| 1,000 | ~2.2 MB |
| 10,000 | ~22 MB |
| 100,000 | ~220 MB |

### L1 Cache Configuration

The L1 cache stores tokenization results at special token boundaries.

<div class="grid" markdown>

<div class="card" markdown>

#### `--tokenizer-cache-enable-l1`

Enable the L1 prefix matching cache.

| Option | `--tokenizer-cache-enable-l1` |
|--------|-------------------------------|
| Default | `false` |

</div>

<div class="card" markdown>

#### `--tokenizer-cache-l1-max-memory`

Maximum memory for the L1 cache in bytes.

| Option | `--tokenizer-cache-l1-max-memory` |
|--------|----------------------------------|
| Default | `52428800` (50 MB) |

</div>

</div>

**Special token boundaries** used for cache points:

| Model Family | Special Tokens |
|--------------|----------------|
| ChatML (Qwen, Yi) | `<\|im_start\|>`, `<\|im_end\|>` |
| Llama 3 | `<\|begin_of_text\|>`, `<\|end_of_text\|>`, `<\|eot_id\|>` |
| GPT | `<\|endoftext\|>` |

---

## Recommended Configurations

<div class="grid" markdown>

<div class="card" markdown>

### :material-flash: High-Throughput Chat

For workloads with repeated system prompts.

```bash
smg \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer-cache-enable-l0 \
  --tokenizer-cache-l0-max-entries 50000
```

**Expected**: 60-90% cache hit rate

</div>

<div class="card" markdown>

### :material-forum: Multi-Turn Conversations

For chat applications with varying conversation lengths.

```bash
smg \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tokenizer-cache-enable-l0 \
  --tokenizer-cache-l0-max-entries 20000 \
  --tokenizer-cache-enable-l1 \
  --tokenizer-cache-l1-max-memory 104857600
```

**Expected**: L0 catches exact repeats, L1 accelerates prefix sharing

</div>

<div class="card" markdown>

### :material-memory: Memory-Constrained

For deployments with limited memory.

```bash
smg \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --tokenizer-cache-enable-l0 \
  --tokenizer-cache-l0-max-entries 5000
```

**Expected**: Moderate benefit with minimal memory

</div>

<div class="card" markdown>

### :material-close-circle: No Caching

For stateless deployments or when memory is critical.

```bash
smg \
  --model-path meta-llama/Llama-3.1-8B-Instruct
# Caching is disabled by default
```

**Use when**: Diverse, unique requests dominate

</div>

</div>

---

## Performance Monitoring

SMG exposes tokenizer cache metrics via Prometheus:

| Metric | Description |
|--------|-------------|
| `smg_tokenizer_cache_l0_hits_total` | L0 cache hit count |
| `smg_tokenizer_cache_l0_misses_total` | L0 cache miss count |
| `smg_tokenizer_cache_l0_entries` | Current L0 cache size |
| `smg_tokenizer_cache_l1_hits_total` | L1 cache hit count |
| `smg_tokenizer_cache_l1_misses_total` | L1 cache miss count |
| `smg_tokenizer_cache_l1_memory_bytes` | Current L1 memory usage |

**Calculate hit rate**:

```promql
rate(smg_tokenizer_cache_l0_hits_total[5m]) /
(rate(smg_tokenizer_cache_l0_hits_total[5m]) + rate(smg_tokenizer_cache_l0_misses_total[5m]))
```

### Cache Sizing Guidelines

| Metric | Guideline |
|--------|-----------|
| L0 entries | 1-2x unique system prompt variants |
| L0 memory | ~2.2KB per entry |
| L1 memory | 50-100 MB for multi-turn conversations |
| Total cache | Keep under 500 MB for typical deployments |

---

## Complete Example

Production configuration with tokenizer and caching:

```bash
smg \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy cache_aware \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --chat-template /templates/llama3.jinja \
  --tokenizer-cache-enable-l0 \
  --tokenizer-cache-l0-max-entries 25000 \
  --tokenizer-cache-enable-l1 \
  --tokenizer-cache-l1-max-memory 104857600 \
  --host 0.0.0.0 \
  --port 8080
```

