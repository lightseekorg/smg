---
title: Load Balancing
---

# Load Balancing

SMG provides multiple load balancing policies to distribute requests across workers. Choosing the right policy depends on your workload characteristics.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-cached: Cache-Aware

**Production default.** Maintains radix tree mirroring backend KV cache for optimal prefix routing with load balancing fallback.

</div>

<div class="card" markdown>

### :material-tray-full: Bucket

Request-length-based routing with adaptive boundaries. Designed for PD disaggregation workloads.

</div>

<div class="card" markdown>

### :material-scale-balance: Power of Two

Load-aware selection without global state. Samples two workers, routes to the lighter one.

</div>

<div class="card" markdown>

### :material-link-variant: Consistent Hashing

Header-based routing with minimal redistribution on scaling. Ideal for session affinity.

</div>

</div>

---

## Policy Comparison

| Policy | Load Aware | Cache Affinity | Session Affinity | Complexity | Best For |
|--------|:----------:|:--------------:|:----------------:|:----------:|----------|
| `cache_aware` | :material-check: | :material-check: | :material-close: | O(prefix) | **Production LLM** |
| `bucket` | :material-check: | :material-close: | :material-close: | O(n) | PD disaggregation |
| `power_of_two` | :material-check: | :material-close: | :material-close: | O(1) | Load balancing |
| `consistent_hashing` | :material-close: | :material-close: | :material-check: | O(log n) | Session affinity |
| `prefix_hash` | :material-check: | Partial | :material-close: | O(log n) | Lightweight caching |
| `manual` | :material-close: | :material-close: | :material-check: | O(1) | Stateful chat |
| `round_robin` | :material-close: | :material-close: | :material-close: | O(1) | Even distribution |
| `random` | :material-close: | :material-close: | :material-close: | O(1) | Testing |

---

## Cache-Aware

The **recommended policy** for production LLM inference. Maintains a multi-tenant radix tree that mirrors backend KV cache state, enabling perfect cache prediction with integrated load balancing.

```bash
smg --policy cache_aware --worker-urls http://w1:8000 http://w2:8000
```

<div class="grid" markdown>

<div class="card" markdown>

#### :material-check-circle: Advantages

- Maximizes KV cache hits (60-90% hit rate)
- Reduces TTFT by 70-75%
- Integrated load balancing fallback
- 100% accurate prefix matching

</div>

<div class="card" markdown>

#### :material-close-circle: Limitations

- Higher memory usage (radix tree per worker)
- O(prefix) selection time
- Requires tokenization

</div>

</div>

**Use when:** Production workloads with repeated prefixes—multi-turn conversations, RAG applications, batch processing with templates.

[**Learn more about Cache-Aware Routing →**](cache-aware.md)

---

## Bucket

Routes requests based on request text length using adaptive boundaries. Periodically adjusts boundaries based on observed load distribution.

```bash
smg --policy bucket --worker-urls http://w1:8000 http://w2:8000 http://w3:8000
```

<div class="grid" markdown>

<div class="card" markdown>

#### :material-check-circle: Advantages

- Request-length awareness
- Adaptive boundary adjustment
- Falls back to load balancing when imbalanced

</div>

<div class="card" markdown>

#### :material-close-circle: Limitations

- O(n) complexity
- No cache locality
- Requires understanding of length distribution

</div>

</div>

**Use when:** PD disaggregation where prefill workers handle different request sizes, or workloads with bimodal request length distribution.

---

## Power of Two Choices

Samples two random workers and selects the one with lower load. Provides good load distribution with minimal coordination overhead—a proven algorithm from distributed systems research.

```bash
smg --policy power_of_two --worker-urls http://w1:8000 http://w2:8000
```

<div class="grid" markdown>

<div class="card" markdown>

#### :material-check-circle: Advantages

- Load-aware without global state
- O(1) selection time
- Exponentially better than random

</div>

<div class="card" markdown>

#### :material-close-circle: Limitations

- No cache locality
- Requires load metrics from workers
- May not find optimal worker

</div>

</div>

**Use when:** Heterogeneous workers with varying response times, or when cache locality doesn't matter.

---

## Consistent Hashing

Provides header-based consistent routing using a hash ring. Minimizes redistribution when workers scale—only ~1/N keys move when adding/removing workers.

```bash
smg --policy consistent_hashing --worker-urls http://w1:8000 http://w2:8000
```

<div class="grid" markdown>

<div class="card" markdown>

#### :material-check-circle: Advantages

- Minimal redistribution on scaling
- Automatic failover to next healthy worker
- O(log n) lookup time

</div>

<div class="card" markdown>

#### :material-close-circle: Limitations

- No load awareness
- No cache locality
- Requires routing key header

</div>

</div>

### Routing Headers

| Header | Description |
|--------|-------------|
| `X-SMG-Target-Worker` | Direct routing by worker index (0-based) |
| `X-SMG-Routing-Key` | Consistent hash routing for session affinity |

**Priority order:** `X-SMG-Target-Worker` → `X-SMG-Routing-Key` → Implicit keys (`Authorization`, `X-Forwarded-For`, `Cookie`) → Random fallback

**Use when:** Session affinity needed, user-to-worker pinning, or consistent routing for stateful applications.

---

## Prefix Hash

A lightweight alternative to full cache-aware routing. Routes requests based on a hash of the first N tokens, using consistent hashing with load factor override.

```bash
smg --policy prefix_hash --prefix-token-count 256 --worker-urls http://w1:8000 http://w2:8000
```

<div class="grid" markdown>

<div class="card" markdown>

#### :material-check-circle: Advantages

- Predictable O(log n) performance
- Lower memory than cache_aware
- Groups similar prefixes together

</div>

<div class="card" markdown>

#### :material-close-circle: Limitations

- Prefix grouping, not exact matching
- Less precise than cache_aware
- Load factor can cause redistribution

</div>

</div>

### Comparison with Cache-Aware

| Aspect | prefix_hash | cache_aware |
|--------|-------------|-------------|
| Lookup | O(log n) | O(prefix_len) |
| Memory | O(workers × virtual_nodes) | O(total_tokens) |
| Precision | Prefix grouping | Exact matching |

**Use when:** Need some cache locality with predictable performance and lower memory footprint.

---

## Manual

Provides sticky session routing with explicit routing key mapping. Unlike consistent hashing, sessions stay with their assigned worker even when new workers are added.

```bash
smg --policy manual --assignment-mode min_load --worker-urls http://w1:8000 http://w2:8000
```

<div class="grid" markdown>

<div class="card" markdown>

#### :material-check-circle: Advantages

- Strong session stickiness
- Automatic failover with recovery
- TTL-based eviction prevents memory growth

</div>

<div class="card" markdown>

#### :material-close-circle: Limitations

- No load balancing for existing sessions
- Requires `X-SMG-Routing-Key` header
- Memory grows with active sessions

</div>

</div>

### Assignment Modes

| Mode | Description |
|------|-------------|
| `random` | Randomly select from healthy workers |
| `min_load` | Select worker with fewest active requests |
| `min_group` | Select worker with fewest routing keys assigned |

**Use when:** Stateful chat sessions where context is stored on workers, or when session continuity is critical.

---

## Round Robin

Rotates through workers sequentially, guaranteeing even distribution over time. Skips unhealthy workers automatically.

```bash
smg --policy round_robin --worker-urls http://w1:8000 http://w2:8000
```

<div class="grid" markdown>

<div class="card" markdown>

#### :material-check-circle: Advantages

- Guaranteed even distribution
- Predictable routing pattern
- Minimal state (counter only)

</div>

<div class="card" markdown>

#### :material-close-circle: Limitations

- No load awareness
- No cache locality
- Ignores request characteristics

</div>

</div>

**Use when:** All workers have equal capacity and you want predictable, even distribution.

---

## Random

The simplest policy—each healthy worker has equal probability of selection. Zero state overhead.

```bash
smg --policy random --worker-urls http://w1:8000 http://w2:8000
```

<div class="grid" markdown>

<div class="card" markdown>

#### :material-check-circle: Advantages

- Zero state overhead
- O(1) selection time
- Completely stateless

</div>

<div class="card" markdown>

#### :material-close-circle: Limitations

- No load awareness
- No cache locality
- Can create hot spots

</div>

</div>

**Use when:** Testing environments or completely homogeneous workloads where simplicity is preferred.

---

## Choosing a Policy

### Decision Guide

| Requirement | Recommended Policy |
|-------------|-------------------|
| Production LLM inference | `cache_aware` |
| Session affinity (sticky sessions) | `manual` or `consistent_hashing` |
| PD disaggregation | `bucket` |
| Load balancing without cache | `power_of_two` |
| Lightweight cache locality | `prefix_hash` |
| Even distribution | `round_robin` |
| Testing/development | `random` |

### Scenario Guide

<div class="grid" markdown>

<div class="card" markdown>

#### :material-message-text: Conversational AI

**Recommended:** `cache_aware`

Maximizes KV cache reuse for multi-turn conversations with shared system prompts.

</div>

<div class="card" markdown>

#### :material-file-search: RAG Applications

**Recommended:** `cache_aware`

Exploits common document prefixes for faster Time to First Token.

</div>

<div class="card" markdown>

#### :material-account-group: Multi-Tenant Platform

**Recommended:** `consistent_hashing` or `manual`

User-to-worker affinity for tenant isolation or stateful sessions.

</div>

<div class="card" markdown>

#### :material-server-network: PD Disaggregation

**Recommended:** `bucket` (prefill) + `power_of_two` (decode)

Length-based routing for prefill, load-based for decode workers.

</div>

</div>

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-cached: Cache-Aware Routing

Deep dive into the radix tree architecture and routing algorithm.

[Cache-Aware Routing →](cache-aware.md)

</div>

<div class="card" markdown>

### :material-cog: Configuration Reference

Detailed configuration options for all routing policies.

[Routing Configuration →](../../configuration/routing.md)

</div>

<div class="card" markdown>

### :material-shield: Circuit Breakers

How SMG handles worker failures gracefully.

[Circuit Breakers →](../reliability/circuit-breakers.md)

</div>

</div>
