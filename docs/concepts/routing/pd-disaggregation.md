---
title: PD Disaggregation
---

# PD Disaggregation

Prefill-Decode (PD) disaggregation separates the two phases of LLM inference onto specialized workers, optimizing Time to First Token (TTFT) and Time Per Output Token (TPOT) independently.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-lightning-bolt: Optimized TTFT

Dedicated prefill workers process prompts with maximum throughput.

</div>

<div class="card" markdown>

### :material-speedometer: Optimized TPOT

Dedicated decode workers generate tokens with minimal latency.

</div>

<div class="card" markdown>

### :material-arrow-expand-all: Independent Scaling

Scale prefill and decode workers based on their specific resource needs.

</div>

<div class="card" markdown>

### :material-memory: KV Cache Transfer

Automatic coordination of KV cache transfer between worker types.

</div>

</div>

---

## Why Disaggregate?

Traditional LLM inference has two distinct phases with different characteristics:

| Phase | Compute Pattern | Bottleneck | Optimization |
|-------|-----------------|------------|--------------|
| **Prefill** | Compute-bound, parallel | GPU compute | Batch similar-length prompts |
| **Decode** | Memory-bound, sequential | Memory bandwidth | Maximize batch size |

Running both phases on the same worker creates inefficiencies:

- Prefill batches are delayed waiting for decode slots
- Decode batches are small due to prefill memory pressure
- Neither phase is optimally configured

**PD disaggregation solves this** by dedicating workers to each phase.

---

## Supported Runtimes

SMG supports PD disaggregation with two inference backends:

| Runtime | Protocol | Dispatch | KV Transfer | Best For |
|---------|----------|----------|-------------|----------|
| **SGLang** | HTTP | Parallel | Bootstrap-based coordination | Production deployments with SGLang |
| **vLLM** | gRPC | Sequential | NIXL (RDMA-based) | High-performance with RDMA networking |

---

## How It Works

### SGLang PD (Parallel Dispatch)

```mermaid
sequenceDiagram
    participant C as Client
    participant S as SMG
    participant P as Prefill Worker
    participant D as Decode Worker

    C->>S: Chat completion request
    S->>S: Find P/D pair
    par Parallel dispatch
        S->>P: Forward request (with bootstrap metadata)
        S->>D: Forward request (with bootstrap metadata)
    end
    P->>P: Process prompt
    P->>D: Transfer KV cache (via bootstrap)
    P-->>S: Prefill complete

    loop Token generation
        D->>D: Generate token
        D-->>S: Stream token
        S-->>C: Stream token
    end

    D-->>S: Generation complete
    S-->>C: Final response
```

SGLang uses **parallel dispatch** with bootstrap-based coordination:

1. SMG sends the request to both prefill and decode workers simultaneously
2. Metadata (bootstrap host/port) enables workers to coordinate
3. Prefill completes and transfers KV cache to decode
4. Decode streams tokens back to client

### vLLM PD (Sequential Dispatch)

```mermaid
sequenceDiagram
    participant C as Client
    participant S as SMG
    participant P as Prefill Worker
    participant D as Decode Worker

    C->>S: Chat completion request
    S->>S: Find P/D pair
    S->>P: Forward request (max_tokens=1)
    P->>P: Process prompt, compute KV cache
    P-->>S: Prefill complete (discarded)

    S->>D: Forward original request
    Note over P,D: NIXL auto-discovers KV cache

    loop Token generation
        D->>D: Generate token
        D-->>S: Stream token
        S-->>C: Stream token
    end

    D-->>S: Generation complete
    S-->>C: Final response
```

vLLM uses **sequential dispatch** with NIXL KV transfer:

1. SMG sends request to prefill worker with `max_tokens=1`
2. Prefill computes KV cache and returns (response discarded)
3. SMG sends original request to decode worker
4. NIXL transparently transfers KV cache via RDMA
5. Decode streams tokens back to client

### Request Flow

1. **Request arrives** at SMG gateway
2. **Find P/D pair**: Select a prefill worker and decode worker
3. **Prefill phase**: Prefill worker processes the prompt
4. **KV transfer**: KV cache is transferred to decode worker
5. **Decode phase**: Decode worker generates tokens
6. **Stream response**: Tokens are streamed back to client

---

## Configuration

### SGLang PD Setup

SGLang workers use HTTP and require a bootstrap port for coordination:

```bash
smg \
  --pd-disaggregation \
  --prefill http://prefill1:8000 9001 \
  --prefill http://prefill2:8000 9002 \
  --decode http://decode1:8000 \
  --decode http://decode2:8000
```

### vLLM PD Setup

vLLM workers use gRPC and NIXL for KV transfer (no bootstrap port needed):

```bash
smg \
  --pd-disaggregation \
  --prefill grpc://prefill1:50051 \
  --prefill grpc://prefill2:50052 \
  --decode grpc://decode1:50053 \
  --decode grpc://decode2:50054
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--pd-disaggregation` | Enable PD disaggregated mode |
| `--prefill` | Prefill worker URL (and optional bootstrap port for SGLang) |
| `--decode` | Decode worker URLs |
| `--prefill-policy` | Routing policy for prefill workers |
| `--decode-policy` | Routing policy for decode workers |

### Per-Phase Policies

Configure different routing policies for each phase:

```bash
smg \
  --pd-disaggregation \
  --worker-urls http://prefill1:8000 http://prefill2:8000 \
  --decode http://decode1:8000 http://decode2:8000 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

### Supported Policies

Both prefill and decode support these policies:

| Policy | Prefill Use Case | Decode Use Case |
|--------|------------------|-----------------|
| `cache_aware` | Maximize prompt cache hits | Less beneficial |
| `power_of_two` | Balance prefill load | Balance decode load |
| `round_robin` | Even distribution | Even distribution |
| `random` | Simple distribution | Simple distribution |

**Recommended**: `cache_aware` for prefill, `power_of_two` for decode.

---

## Kubernetes Service Discovery

Use label selectors to automatically discover prefill and decode workers.

### Configuration

```bash
smg \
  --service-discovery \
  --pd-disaggregation \
  --prefill-selector "app=sglang,role=prefill" \
  --decode-selector "app=sglang,role=decode" \
  --service-discovery-namespace inference
```

### Worker Deployments

```yaml
# Prefill workers
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: sglang-prefill
  namespace: inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sglang
      role: prefill
  template:
    metadata:
      labels:
        app: sglang
        role: prefill
    spec:
      containers:
        - name: sglang
          image: lmsysorg/sglang:latest
          args:
            - --model-path=meta-llama/Llama-3.1-70B-Instruct
            - --port=8000
            - --prefill-only
          resources:
            limits:
              nvidia.com/gpu: 4

---
# Decode workers
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: sglang-decode
  namespace: inference
spec:
  replicas: 4
  selector:
    matchLabels:
      app: sglang
      role: decode
  template:
    metadata:
      labels:
        app: sglang
        role: decode
    spec:
      containers:
        - name: sglang
          image: lmsysorg/sglang:latest
          args:
            - --model-path=meta-llama/Llama-3.1-70B-Instruct
            - --port=8000
            - --decode-only
          resources:
            limits:
              nvidia.com/gpu: 2
```

---

## P/D Pair Selection

SMG maintains awareness of which prefill and decode workers can communicate.

### Pairing Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Any-to-Any** | Any prefill can send to any decode | Network with uniform latency |
| **Affinity** | Prefer co-located pairs | Reduce KV transfer latency |
| **Load-Based** | Select least loaded pair | Maximize throughput |

### KV Cache Transfer

The KV cache is transferred between workers using the backend's native mechanism:

| Backend | Transfer Method | Coordination |
|---------|-----------------|--------------|
| SGLang | NCCL/Gloo over network | Bootstrap metadata (host/port/room) |
| vLLM | NIXL (RDMA-based) | Automatic prefix matching |

**SGLang**: SMG injects bootstrap metadata (`DisaggregatedParams`) into requests, enabling workers to coordinate KV transfer through a shared "room".

**vLLM**: SMG uses the simple proxy pattern—sends `max_tokens=1` to prefill to trigger KV cache computation, then NIXL automatically discovers and transfers the cache to decode via RDMA prefix matching. No protocol changes required.

---

## Sizing Guidelines

### Prefill Workers

Prefill is **compute-bound**:

- More GPUs per worker = faster prefill
- Fewer workers with more GPUs is often better
- Size for your longest prompts

| Prompt Length | Recommended GPUs |
|---------------|------------------|
| < 4K tokens | 1-2 GPUs |
| 4K - 16K tokens | 2-4 GPUs |
| 16K - 64K tokens | 4-8 GPUs |
| > 64K tokens | 8+ GPUs |

### Decode Workers

Decode is **memory-bandwidth-bound**:

- More workers = higher throughput
- Smaller workers can batch more requests
- Size for your target concurrency

| Concurrent Users | Recommended Setup |
|------------------|-------------------|
| < 50 | 2 decode workers |
| 50 - 200 | 4 decode workers |
| 200 - 500 | 8 decode workers |
| > 500 | 16+ decode workers |

### Ratio Guidelines

| Workload Type | Prefill:Decode Ratio |
|---------------|----------------------|
| Short prompts, long outputs | 1:4 |
| Balanced prompts/outputs | 1:2 |
| Long prompts, short outputs | 1:1 or 2:1 |
| RAG with large context | 2:1 |

---

## Monitoring

### Metrics

| Metric | Description |
|--------|-------------|
| `smg_pd_prefill_duration_seconds` | Prefill phase duration |
| `smg_pd_decode_duration_seconds` | Decode phase duration |
| `smg_pd_kv_transfer_duration_seconds` | KV cache transfer time |
| `smg_pd_pair_selections_total` | P/D pair selection count |

### Key Performance Indicators

| KPI | Target | Indicates |
|-----|--------|-----------|
| TTFT | < 500ms | Prefill performance |
| TPOT | < 50ms | Decode performance |
| KV transfer time | < 100ms | Network performance |

### PromQL Queries

<div class="grid" markdown>

<div class="card" markdown>

#### Phase Duration

```promql
# Average prefill time
rate(smg_pd_prefill_duration_seconds_sum[5m]) /
rate(smg_pd_prefill_duration_seconds_count[5m])

# Average decode time per token
rate(smg_pd_decode_duration_seconds_sum[5m]) /
rate(smg_pd_decode_duration_seconds_count[5m])
```

</div>

<div class="card" markdown>

#### Worker Utilization

```promql
# Prefill worker utilization
smg_worker_requests_active{role="prefill"} /
smg_worker_max_concurrent{role="prefill"}

# Decode worker utilization
smg_worker_requests_active{role="decode"} /
smg_worker_max_concurrent{role="decode"}
```

</div>

</div>

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| High TTFT | Prefill workers overloaded | Add prefill workers or GPUs |
| High TPOT | Decode workers overloaded | Add decode workers |
| KV transfer timeout | Network congestion | Check network bandwidth |
| Uneven load | Poor pairing | Adjust routing policy |
| Decode queue buildup | Prefill too fast | Balance P:D ratio |

### Debug Logging

```bash
RUST_LOG=smg::pd=debug smg --pd-disaggregation ...
```

### Verify Configuration

```bash
# Check discovered workers
curl http://smg:3001/workers | jq

# Check worker roles
curl http://smg:3001/workers | jq '.[] | {url, role}'
```

---

## Complete Example

### SGLang PD (Kubernetes Service Discovery)

```bash
smg \
  --service-discovery \
  --pd-disaggregation \
  --prefill-selector "app=sglang,role=prefill" \
  --decode-selector "app=sglang,role=decode" \
  --prefill-policy cache_aware \
  --decode-policy power_of_two \
  --cb-failure-threshold 3 \
  --health-check-interval-secs 10 \
  --host 0.0.0.0 \
  --port 8000
```

### SGLang PD (Static Workers)

```bash
smg \
  --pd-disaggregation \
  --prefill http://prefill-0:8000 9001 \
  --prefill http://prefill-1:8000 9002 \
  --decode http://decode-0:8000 \
  --decode http://decode-1:8000 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

### vLLM PD (Static Workers with NIXL)

```bash
smg \
  --pd-disaggregation \
  --prefill grpc://prefill-0:50051 \
  --prefill grpc://prefill-1:50052 \
  --decode grpc://decode-0:50053 \
  --decode grpc://decode-1:50054 \
  --prefill-policy cache_aware \
  --decode-policy round_robin
```

### Launching vLLM PD Workers

vLLM workers require NIXL configuration for KV transfer:

```bash
# Prefill worker (kv_producer)
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
python -m vllm.entrypoints.grpc_server \
  --model /path/to/model \
  --port 50051 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Decode worker (kv_consumer)
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
python -m vllm.entrypoints.grpc_server \
  --model /path/to/model \
  --port 50052 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

Or use the provided helper script:

```bash
./scripts/launch-pd-workers.sh vllm /path/to/model
```

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-kubernetes: Service Discovery

Automatic worker discovery in Kubernetes.

[Service Discovery →](../architecture/service-discovery.md)

</div>

<div class="card" markdown>

### :material-cached: Cache-Aware Routing

Optimize prefill with cache-aware routing.

[Cache-Aware Routing →](cache-aware.md)

</div>

<div class="card" markdown>

### :material-scale-balance: Load Balancing

All available routing policies.

[Load Balancing →](load-balancing.md)

</div>

</div>
