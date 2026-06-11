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

SMG sends to prefill first with `max_tokens=1`, then sends the original request to decode, relaying KV-transfer metadata between the two legs:

- **NIXL**: SMG tags the prefill request with `do_remote_decode=true`, harvests the `kv_transfer_params` the prefill engine returns (engine id, request id, block ids, side-channel address, TP size), and forwards them verbatim with the decode request so decode pulls the KV cache over NIXL.
- **Mooncake**: the connector is push-based and returns nothing, so SMG mints a shared `transfer_id`, tags the prefill request with it, and synthesizes the decode params (`remote_engine_id` discovered from the worker at registration, `remote_bootstrap_addr` from the worker's bootstrap host/port). With an older servicer that doesn't report `kv_engine_id`, SMG falls back to legacy host/port injection.

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

`VLLM_NIXL_SIDE_CHANNEL_PORT` must be unique per worker on the same host (with
data parallelism each rank uses `port + dp_rank`). When prefill and decode run
on different machines, also set `VLLM_NIXL_SIDE_CHANNEL_HOST` to an address
reachable from the decode worker — prefill embeds this host/port in the
handoff params that decode uses to fetch the KV cache.

To verify KV transfer is active, send a request and look for `Transfer plan:`
in the decode worker log (vLLM >= 0.20). If the router logs
`prefill returned no kv_transfer_params`, upgrade the servicer
(smg-grpc-servicer >= 0.5.4, smg-grpc-proto >= 0.4.9) or check the
`--kv-transfer-config` on the workers.

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
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","engine_id":"prefill-0"}'

# Decode worker
python -m vllm.entrypoints.grpc_server \
  --model /path/to/model \
  --port 50052 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

Set an explicit `engine_id` on each prefill worker in production. SMG discovers
the id once at worker registration; without a pinned id, vLLM generates a new
one per process, so a restarted prefill container would invalidate the
registered id until the worker is re-registered.

Limitation: Mooncake PD requires the prefill workers to run without vLLM data
parallelism (`data_parallel_size = 1`). With DP active the servicer reports no
engine id and SMG falls back to legacy host/port injection (decode recomputes
the prompt locally).

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

## vLLM vs SGLang PD at a Glance

| | vLLM PD | SGLang PD |
|---|---------|-----------|
| **Protocol** | gRPC | HTTP |
| **Dispatch** | Prefill first, then decode | Both workers receive request simultaneously |
| **KV Transfer** | NIXL (RDMA) or Mooncake (TCP/RDMA) | Bootstrap-based coordination |
| **SMG flags** | `--prefill grpc://...` + `--model-path` | `--prefill http://... <bootstrap_port>` |

---

## Next Steps

For sizing guidelines, per-phase routing policies, Kubernetes service discovery, and monitoring, see the full [PD Disaggregation Concepts](../concepts/routing/pd-disaggregation.md) page.
