---
title: Routing Policies
---

# Routing Policies

SMG provides multiple routing policies for distributing requests across workers. This guide covers all available policies, their configuration options, and when to use each one.

---

## Overview

### Available Routing Policies

| Policy | Description | Best For |
|--------|-------------|----------|
| `cache_aware` | Routes based on KV cache affinity with load balancing fallback | Default choice for LLM inference |
| `random` | Uniform random distribution | Simple deployments, testing |
| `round_robin` | Sequential rotation through workers | Even distribution without state |
| `power_of_two` | Samples two workers, picks least loaded | Load-aware without full state |
| `prefix_hash` | Hash-based routing on token prefixes | Lightweight cache locality |
| `consistent_hashing` | Header-based consistent routing | Session affinity, sticky sessions |
| `manual` | Explicit routing key mapping | Stateful chat sessions |
| `bucket` | Request-length-based bucket assignment | PD disaggregation workloads |

### Selecting a Policy

```bash
# Set the routing policy
smg --policy cache_aware --worker-urls http://worker1:8000 http://worker2:8000
```

---

## Basic Policies

### Random Policy

Selects workers randomly with uniform distribution among healthy workers. Simple and stateless.

```bash
smg --policy random --worker-urls http://worker1:8000 http://worker2:8000
```

**Characteristics:**

- No state maintained
- O(1) selection time
- Good for homogeneous workloads

**Use Cases:**

- Testing and development
- Stateless workloads with similar request patterns
- When simplicity is preferred over optimization

---

### Round Robin Policy

Selects workers in sequential order, cycling through all healthy workers.

```bash
smg --policy round_robin --worker-urls http://worker1:8000 http://worker2:8000
```

**Characteristics:**

- Maintains a counter for rotation
- Guarantees even distribution over time
- Skips unhealthy workers automatically

**Use Cases:**

- Even load distribution without considering worker state
- Predictable routing patterns
- Testing load distribution

---

### Power of Two Choices Policy

Randomly selects two workers and routes to the one with lower load. Provides good load distribution with minimal coordination overhead.

```bash
smg --policy power_of_two --worker-urls http://worker1:8000 http://worker2:8000
```

**Characteristics:**

- O(1) selection time
- Load-aware without global coordination
- Uses either cached token loads or request counts

**Load Metrics:**

- Uses high-fidelity token loads when available from monitoring
- Falls back to request counts when token data is missing
- Ensures fairness by comparing same metric types

**Use Cases:**

- Load-sensitive routing without full state tracking
- Large deployments where global state is expensive
- When cache locality is less important than load balancing

---

## Cache-Aware Policy

The default and recommended policy for LLM inference. Combines cache-aware routing with load balancing to optimize both KV cache hit rates and request distribution.

```bash
smg --policy cache_aware \
    --cache-threshold 0.3 \
    --balance-abs-threshold 64 \
    --balance-rel-threshold 1.5 \
    --eviction-interval 120 \
    --max-tree-size 67108864 \
    --worker-urls http://worker1:8000 http://worker2:8000
```

### Multi-Tenant Radix Tree Architecture

<div class="architecture-diagram">
  <img src="../../assets/images/radix-tree.svg" alt="Multi-Tenant Radix Tree Architecture">
</div>

SMG maintains a **multi-tenant radix tree** that mirrors the KV cache state on each backend worker. This enables true cache-aware routing with 100% prefix match accuracy.

<div class="grid" markdown>

<div class="card" markdown>

### :material-sync: 100% Backend Synchronization

The gateway's radix tree uses the **exact same parameters** as backend schedulers:

- **Same tokens**: Pre-tokenized input matches backend representation
- **Same page size**: Aligned to kernel page boundaries (e.g., 16 tokens for FlashInfer)
- **Same eviction policy**: LRU, LFU, FIFO, MRU, FILO, or Priority

</div>

<div class="card" markdown>

### :material-tree: Two Tree Implementations

| Tree Type | Mode | Input | Use Case |
|-----------|------|-------|----------|
| **StringTree** | HTTP | Raw text | OpenAI-compatible endpoints |
| **TokenTree** | gRPC | Token IDs | Direct worker communication |

</div>

</div>

**Why This Matters:**

- **Perfect Cache Prediction**: Since the gateway tree mirrors backend behavior exactly, prefix match calculations are 100% accurate
- **Kernel-Aware**: Honors page boundaries used by inference kernels (FlashInfer, Mamba, etc.)
- **Eviction Parity**: Tracks the same eviction policy as SGLang/vLLM/TensorRT-LLM schedulers

### How It Works

1. **Cache-Aware Routing (Balanced State)**
   - Maintains an approximate radix tree for each worker
   - Stores request text/tokens to track cache affinity
   - Routes to worker with highest prefix match if match rate > threshold
   - Routes to worker with smallest tree (most cache capacity) if match rate <= threshold

2. **Load Balancing (Imbalanced State)**
   - Monitors load across workers
   - Switches to shortest-queue routing when imbalanced
   - System is imbalanced when both conditions are met:
     - `(max_load - min_load) > balance_abs_threshold`
     - `max_load > balance_rel_threshold * min_load`

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cache-threshold` | `0.3` | Minimum prefix match ratio (0.0-1.0) to use cache-aware routing |
| `--balance-abs-threshold` | `64` | Absolute load difference to trigger load balancing |
| `--balance-rel-threshold` | `1.5` | Relative load ratio to trigger load balancing |
| `--eviction-interval` | `120` | Seconds between LRU cache eviction cycles |
| `--max-tree-size` | `67108864` | Maximum nodes per tree before eviction |

### Tuning Guidelines

**Cache Threshold (`--cache-threshold`):**

- Lower values (0.1-0.3): More aggressive cache routing, better hit rates
- Higher values (0.5-0.8): More conservative, better load distribution
- Recommended: 0.3 for most workloads

**Balance Thresholds:**

- `--balance-abs-threshold`: Set based on expected request queue depth
- `--balance-rel-threshold`: Set based on acceptable load imbalance (1.5 = 50% difference)

**Tree Size:**

- Larger trees = better cache tracking but more memory
- Default (67M nodes) suitable for most deployments
- Reduce for memory-constrained environments

---

## Consistent Hashing Policy

Provides header-based consistent routing for session affinity. Uses a hash ring to minimize redistribution when workers scale.

```bash
smg --policy consistent_hashing --worker-urls http://worker1:8000 http://worker2:8000
```

### HTTP Headers

| Header | Description |
|--------|-------------|
| `X-SMG-Target-Worker` | Direct routing by worker index (0-based). Returns error if unavailable. |
| `X-SMG-Routing-Key` | Consistent hash routing for session affinity |

### Priority Order

1. `X-SMG-Target-Worker` (explicit worker selection)
2. `X-SMG-Routing-Key` (consistent hash routing)
3. Implicit keys (`Authorization`, `X-Forwarded-For`, `Cookie` headers)
4. Random fallback (truly anonymous clients)

### Example Usage

```bash
# Route by explicit routing key
curl -H "X-SMG-Routing-Key: user-123" http://localhost:30000/v1/chat/completions ...

# Route to specific worker (index 0)
curl -H "X-SMG-Target-Worker: 0" http://localhost:30000/v1/chat/completions ...
```

### Characteristics

- **Minimal Redistribution**: When workers scale, only ~1/N keys move (N = worker count)
- **Automatic Failover**: Routes to next healthy worker on ring when target is unhealthy
- **Recovery**: Returns to original worker when it recovers
- **Complexity**: O(log n) lookup + O(k) walk for k consecutive unhealthy workers

### Use Cases

- Session affinity for stateful applications
- User-to-worker pinning
- Consistent routing for cache-sensitive workloads

---

## Prefix Hash Policy

A lightweight alternative to full cache-aware routing. Routes requests based on a hash of prefix tokens.

```bash
smg --policy prefix_hash \
    --prefix-token-count 256 \
    --prefix-hash-load-factor 1.25 \
    --worker-urls http://worker1:8000 http://worker2:8000
```

### How It Works

1. Extract first N tokens from the request
2. Hash the token sequence using xxhash
3. Use consistent hash ring to find target worker
4. If worker is overloaded (load > avg * load_factor), find least loaded

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--prefix-token-count` | `256` | Number of prefix tokens to hash |
| `--prefix-hash-load-factor` | `1.25` | Load factor threshold for walking the ring |

### Comparison with Cache-Aware

| Aspect | prefix_hash | cache_aware |
|--------|-------------|-------------|
| Lookup | O(log n) | O(prefix_len) |
| Memory | O(workers * virtual_nodes) | O(total_tokens) |
| Update | O(1) | O(prefix_len) |
| Precision | Prefix grouping | Exact matching |

### Use Cases

- When predictable O(log n) performance is needed
- Lower memory environments
- Workloads with common prefixes (system prompts, etc.)

---

## Manual Policy

Provides sticky session routing with explicit routing key mapping. Unlike consistent hashing, this policy does NOT redistribute sessions when workers are added.

```bash
smg --policy manual \
    --assignment-mode random \
    --max-idle-secs 14400 \
    --eviction-interval 60 \
    --worker-urls http://worker1:8000 http://worker2:8000
```

### HTTP Header

| Header | Description |
|--------|-------------|
| `X-SMG-Routing-Key` | Routing key for sticky session mapping |

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--assignment-mode` | `random` | How to assign new routing keys: `random`, `min_load`, `min_group` |
| `--max-idle-secs` | `14400` | Seconds before idle sessions are evicted (4 hours) |
| `--eviction-interval` | `60` | Seconds between eviction checks |

### Assignment Modes

| Mode | Description |
|------|-------------|
| `random` | Randomly select from healthy workers |
| `min_load` | Select worker with fewest active requests |
| `min_group` | Select worker with fewest routing keys assigned |

### Behavior

- **Strong Stickiness**: Sessions stay with their assigned worker even when new workers are added
- **Automatic Failover**: Routes to another healthy worker if assigned worker becomes unhealthy
- **Recovery**: Can return to original worker after recovery (maintains up to 2 candidate workers)
- **TTL Eviction**: Removes idle sessions to prevent unbounded memory growth

### Use Cases

- Stateful chat sessions with context stored on workers
- When session continuity is more important than optimal distribution
- Multi-turn conversations requiring consistent worker assignment

---

## Bucket Policy

Routes requests based on request text length using adaptive boundaries. Primarily used for PD (prefill-decode) disaggregation.

### How It Works

1. Measures request character count
2. Assigns request to bucket based on length boundaries
3. Periodically adjusts boundaries based on observed load distribution
4. Falls back to load balancing when system is imbalanced

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--balance-abs-threshold` | `32` | Absolute load difference threshold |
| `--balance-rel-threshold` | `1.0001` | Relative load ratio threshold |
| `--bucket-adjust-interval-secs` | `5` | Seconds between boundary adjustments |

### Use Cases

- PD disaggregation where prefill workers handle different request sizes
- Workloads with bimodal request length distribution

---

## PD Disaggregation

Prefill-Decode (PD) disaggregation separates the prefill (prompt processing) and decode (token generation) phases across different workers.

```bash
smg --pd-disaggregation \
    --worker-urls http://prefill1:8000 http://prefill2:8000 \
    --decode http://decode1:8000 http://decode2:8000 \
    --prefill-policy cache_aware \
    --decode-policy power_of_two
```

### Configuration Options

| Option | Description |
|--------|-------------|
| `--pd-disaggregation` | Enable PD disaggregated mode |
| `--worker-urls` | Prefill server URLs |
| `--decode` | Decode server URLs (can be specified multiple times) |
| `--prefill-policy` | Routing policy for prefill workers |
| `--decode-policy` | Routing policy for decode workers |

### Supported Policies

Both prefill and decode policies support:

- `random`
- `round_robin`
- `cache_aware`
- `power_of_two`
- `prefix_hash`
- `manual`

### Example Configurations

**Cache-aware prefill with power-of-two decode:**

```bash
smg --pd-disaggregation \
    --worker-urls http://prefill1:8000 http://prefill2:8000 \
    --decode http://decode1:8000 http://decode2:8000 \
    --prefill-policy cache_aware \
    --decode-policy power_of_two
```

**With bucket policy for prefill:**

```bash
smg --pd-disaggregation \
    --worker-urls http://prefill1:8000 http://prefill2:8000 http://prefill3:8000 \
    --decode http://decode1:8000 http://decode2:8000 \
    --balance-abs-threshold 32 \
    --balance-rel-threshold 1.0001
```

### Kubernetes Service Discovery

```bash
smg --pd-disaggregation \
    --service-discovery \
    --prefill-selector role=prefill \
    --decode-selector role=decode \
    --service-discovery-port 8000
```

---

## Advanced Options

### Data Parallelism Aware Scheduling

Enable DP-aware scheduling for workers using data parallelism (e.g., tensor parallelism across multiple GPUs).

```bash
smg --dp-aware --worker-urls http://worker1:8000 http://worker2:8000
```

When enabled:

- Workers are identified by URL including DP rank
- Proper routing to specific DP ranks within a worker group

### IGW (Inference Gateway) Mode

Enable multi-model support where the router manages multiple models dynamically.

```bash
smg --enable-igw --worker-urls http://worker1:8000 http://worker2:8000
```

**Characteristics:**

- Routes requests based on model ID in the request
- Creates separate routing contexts per model
- Automatically enabled when using service discovery

**Single Router Mode (default, `--enable-igw=false`):**

- Router owns workers directly
- Single model deployment
- Simpler configuration

**Multi-Router Mode (`--enable-igw=true`):**

- RouterManager coordinates multiple model routers
- Dynamic model registration
- Per-model worker pools

---

## Policy Selection Guidelines

### Decision Tree

1. **Need session affinity?**
   - Strong stickiness required → `manual`
   - Minimal redistribution acceptable → `consistent_hashing`

2. **LLM inference with KV cache?**
   - Full optimization → `cache_aware` (default)
   - Lightweight alternative → `prefix_hash`

3. **Load-sensitive routing?**
   - Minimal coordination → `power_of_two`

4. **Simple distribution?**
   - Even distribution → `round_robin`
   - Random distribution → `random`

5. **PD disaggregation?**
   - Request-length based → `bucket` (for prefill)
   - Load-based → `power_of_two` (for decode)

### Performance Characteristics

| Policy | Memory | CPU | State |
|--------|--------|-----|-------|
| `random` | O(1) | O(1) | None |
| `round_robin` | O(1) | O(1) | Counter |
| `power_of_two` | O(n) | O(1) | Load cache |
| `cache_aware` | O(tokens) | O(prefix) | Radix trees |
| `prefix_hash` | O(workers) | O(log n) | Hash ring |
| `consistent_hashing` | O(workers) | O(log n) | Hash ring |
| `manual` | O(sessions) | O(1) | Session map |

---

## Monitoring

All routing policies emit Prometheus metrics for monitoring:

- `smg_worker_selection_policy_branch_total` - Selection path taken
- `smg_request_latency_seconds` - Request latency by policy
- `smg_worker_load` - Current load per worker

See [Prometheus Metrics](../reference/metrics.md) for the complete list.
