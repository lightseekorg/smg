---
title: PD Disaggregation
---

# PD Disaggregation

Prefill-Decode (PD) disaggregation separates the two phases of LLM inference — prompt processing (prefill) and token generation (decode) — onto specialized workers. This optimizes Time to First Token (TTFT) and throughput independently.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- At least one prefill worker and one decode worker
- For vLLM PD: workers started with gRPC entrypoint and KV transfer backend

</div>

---

## Why Disaggregate?

| Phase | Compute Pattern | Bottleneck |
|-------|-----------------|------------|
| **Prefill** | Compute-bound, parallel | GPU compute |
| **Decode** | Memory-bound, sequential | Memory bandwidth |

Running both on the same worker creates contention — prefill batches wait for decode slots, and decode batches stay small due to memory pressure. Dedicating workers to each phase removes this conflict.

---

## SGLang PD

SMG sends the request to both prefill and decode workers simultaneously, and they coordinate KV cache transfer through a bootstrap mechanism.

### Start SGLang Workers

```bash
# Prefill worker
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --port 8000 \
  --prefill-only

# Decode worker
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --port 8001 \
  --decode-only
```

### Start SMG

Each prefill worker needs a bootstrap port for coordination:

```bash
smg \
  --pd-disaggregation \
  --prefill http://prefill:8000 9001 \
  --decode http://decode:8001 \
  --host 0.0.0.0 \
  --port 30000
```

### Multiple Workers

```bash
smg \
  --pd-disaggregation \
  --prefill http://prefill1:8000 9001 \
  --prefill http://prefill2:8000 9002 \
  --decode http://decode1:8001 \
  --decode http://decode2:8001 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

---

## vLLM PD

SMG sends to prefill first with `max_tokens=1`, then sends the original request to decode. The KV backend (NIXL or Mooncake) transfers the cache transparently.

### Start vLLM Workers with NIXL

```bash
# Prefill worker
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
python -m vllm.entrypoints.grpc_server \
  --model /path/to/model \
  --port 50051 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Decode worker
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
python -m vllm.entrypoints.grpc_server \
  --model /path/to/model \
  --port 50052 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
```

### Start SMG

vLLM workers use `grpc://` URLs and require `--model-path` for tokenizer loading:

```bash
smg \
  --pd-disaggregation \
  --prefill grpc://prefill:50051 \
  --decode grpc://decode:50052 \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30000
```

### Alternative: Mooncake Backend

Mooncake supports TCP transport (no RDMA required). Each prefill worker needs a unique bootstrap port:

```bash
# Prefill worker
VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
python -m vllm.entrypoints.grpc_server \
  --model /path/to/model \
  --port 50051 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'

# Decode worker
python -m vllm.entrypoints.grpc_server \
  --model /path/to/model \
  --port 50052 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

```bash
smg \
  --pd-disaggregation \
  --prefill grpc://prefill:50051 8998 \
  --decode grpc://decode:50052 \
  --model-path /path/to/model
```

### Helper Script

Use the provided script to launch workers with either backend:

```bash
# NIXL (default)
./scripts/launch-pd-workers.sh vllm /path/to/model

# Mooncake
KV_BACKEND=mooncake ./scripts/launch-pd-workers.sh vllm /path/to/model
```

---

## Verify

```bash
# Check workers and their roles
curl http://localhost:30000/workers | jq

# Send a request
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## SGLang vs vLLM PD at a Glance

| | SGLang PD | vLLM PD |
|---|-----------|---------|
| **Protocol** | HTTP | gRPC |
| **Dispatch** | Both workers receive request simultaneously | Prefill first, then decode |
| **KV Transfer** | Bootstrap-based coordination | NIXL (RDMA) or Mooncake (TCP/RDMA) |
| **SMG flags** | `--prefill http://... <bootstrap_port>` | `--prefill grpc://...` + `--model-path` |

---

## Next Steps

For sizing guidelines, per-phase routing policies, Kubernetes service discovery, and monitoring, see the full [PD Disaggregation Concepts](../concepts/routing/pd-disaggregation.md) page.
