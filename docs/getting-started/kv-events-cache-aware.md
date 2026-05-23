---
title: KV Events Cache-Aware Routing
---

# KV Events Cache-Aware Routing

This guide walks through wiring an **SGLang worker emitting KV cache events** to **SMG running the cache-aware policy in event-driven mode**, so the gateway routes each request to the worker whose KV cache already holds the longest prefix.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- Read [Cache-Aware Routing](../concepts/routing/cache-aware.md) for the routing concepts
- A machine that can run an SGLang worker (GPU + CUDA-capable Python environment)
- `smg-grpc-servicer[sglang]` installed alongside SGLang

</div>

---

## Why event-driven?

Cache-aware routing has three internal flavours. The one this guide configures is the most accurate of the three because it routes against the worker's **actual** KV cache state rather than an approximation.

| Flavour | Tree | Input | Worker connection | Triggered when |
|---|---|---|---|---|
| **Event-driven** | `PositionalIndexer` (event-built) | Token IDs | gRPC | Worker emits KV events |
| Approximate token tree | `TokenTree` (prefix observed at routing time) | Token IDs | gRPC | Worker is gRPC but emits no events |
| Approximate string tree | `Tree` (prefix observed at routing time) | Raw text | HTTP | Worker is HTTP |

Selection is automatic and per-worker: enabling events on one worker upgrades that worker's routing path; the others keep using the approximate tree.

---

## How the pieces fit together

```
┌────────────┐     ┌────────────────────────┐     ┌──────────────────┐
│   client   │ ──▶ │ smg gateway            │ ──▶ │ smg-grpc-servicer│
│            │     │ ─ cache_aware policy   │     │ + sglang scheduler│
│            │     │ ─ KvEventMonitor       │ ◀── │ ZMQ PUB ─ KV evt │
└────────────┘     └────────────────────────┘     └──────────────────┘
                         gRPC                            ZMQ (in-process)
                         SubscribeKvEvents
```

1. SGLang's scheduler publishes block-stored / block-removed events on a ZMQ `PUB` socket configured by `--kv-events-config`.
2. `smg-grpc-servicer` (running in the same process, launched via `--grpc-mode`) subscribes to that ZMQ socket and re-publishes the events as a gRPC server-streaming RPC (`SubscribeKvEvents`).
3. SMG's `KvEventMonitor` opens one gRPC subscription per worker, feeds the events into a per-model `PositionalIndexer`, and the `cache_aware` policy queries that indexer at routing time.

---

## Step 1 — Launch the SGLang worker

Install the SGLang extra of the servicer, then launch the SGLang server with both `--grpc-mode` and `--kv-events-config`:

```bash
pip install "smg-grpc-servicer[sglang]"

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 50051 \
  --grpc-mode \
  --page-size 16 \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv-events"}'
```

What each flag does:

| Flag | Why |
|---|---|
| `--grpc-mode` | Hands the request loop off to `smg-grpc-servicer`'s gRPC `SglangScheduler` service instead of SGLang's default HTTP server. Required for SMG to talk to this worker in gRPC mode. |
| `--page-size 16` | The KV cache block size, in tokens. Mirror this in SMG's worker config so the gateway can align its overlap scoring to the right page boundaries (see [Block size alignment](#block-size-alignment)). |
| `--kv-events-config` | A JSON object parsed by SGLang's `KVEventsConfig.from_cli`. Setting `publisher: "zmq"` is what actually turns on event publishing — the default `publisher: "null"` is a no-op. |

### `--kv-events-config` field reference

All fields and defaults match SGLang's `KVEventsConfig` (see `python/sglang/srt/disaggregation/kv_events.py` upstream):

| Field | Default | Notes |
|---|---|---|
| `publisher` | `"null"` | Set to `"zmq"` to enable. Any other value disables event bridging in the servicer. |
| `endpoint` | `"tcp://*:5557"` | ZMQ `PUB` socket address. The publisher **binds** when the endpoint contains `*`, `::`, or starts with `ipc://` / `inproc://`; otherwise it connects. |
| `topic` | `""` | ZMQ topic prefix. Match this on the subscriber side; SMG accepts any topic, so the value here matters only if you wire other subscribers in parallel. |
| `replay_endpoint` | `null` | Optional REQ/REP socket for replaying missed events. SMG does not currently use replay. |
| `buffer_steps` | `10000` | Size of the in-publisher replay buffer (events). |
| `hwm` | `100000` | ZMQ high-water mark. Once N events are queued and the consumer hasn't drained them, new events drop. |
| `max_queue_size` | `100000` | Internal queue between SGLang and the ZMQ thread. |

For data-parallel deployments, the actual TCP port becomes `endpoint_port + dp_rank` (rank 0 keeps the configured port).

---

## Step 2 — Launch SMG

Point SMG at the gRPC worker and select `cache_aware`:

```bash
smg \
  --worker-urls grpc://worker-1:50051 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --policy cache_aware \
  --block-size 16 \
  --host 0.0.0.0 \
  --port 30000
```

The flags that matter for event-driven routing:

| Flag | Why |
|---|---|
| `grpc://...` worker URL | Event subscription only runs over gRPC; HTTP workers are skipped silently. |
| `--policy cache_aware` | The only policy that consults the `PositionalIndexer`. |
| `--block-size 16` | Fallback block size used until the first event arrives. After events start flowing, SMG **learns** the worker's true block size from the event payload and uses the learned value automatically. |

`--model-path` is still required for tokenization at the gateway, the same as any gRPC-worker deployment ([gRPC Workers](grpc-workers.md)).

### Block size alignment

The cache-aware policy chunks an incoming request's token IDs into blocks of `block_size` tokens to look them up in the `PositionalIndexer`. If the block size does not match what SGLang actually wrote to its cache, **the lookup misses every block** and the policy silently falls back to load-only routing.

Order of precedence inside SMG:

1. **Event-learned block size** (highest priority — discovered per-model from the event stream).
2. **Per-worker `kv_block_size`** in the worker spec, if you load workers from a config file.
3. **`--block-size` CLI flag** (router-wide default).

In practice: keep `--page-size` (SGLang) and `--block-size` (SMG) numerically equal, and let SMG correct itself once events arrive.

### Worker config file

If you load workers from a config file rather than CLI, pin the block size per worker so event-driven routing works on the very first request:

```yaml
workers:
  - url: grpc://worker-1:50051
    connection_mode: grpc
    kv_block_size: 16
  - url: grpc://worker-2:50052
    connection_mode: grpc
    kv_block_size: 16
```

---

## Step 3 — Send a request

The API surface is unchanged:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ]
  }'
```

Send the same prompt twice. On the second call the request should land on the worker that already serves the first call's prefix.

---

## Verifying event delivery

The gateway logs three events that prove the path is live.

**1. Subscription started.** When SMG registers a gRPC worker, `KvEventMonitor::on_worker_added` logs:

```
INFO Starting KV event subscription worker_url=grpc://worker-1:50051 model_id=meta-llama/Llama-3.1-8B-Instruct
```

If you do not see this line for a worker, that worker is either HTTP or the subscription task crashed before the first connect — check the worker logs.

**2. Backend block size learned.** Once the first event arrives, SMG records the backend's actual block size:

```
DEBUG Learned block_size=16 model_id=meta-llama/Llama-3.1-8B-Instruct
```

**3. Routing decision uses the indexer.** With `RUST_LOG=model_gateway::policies::cache_aware=debug`, a routed request prints the overlap count and the chosen worker.

If events never arrive, the policy keeps working — it falls back to the approximate `TokenTree` for that worker — so cache hits will still happen, just less accurately.

---

## Tuning

| Knob | Where | Effect |
|---|---|---|
| `--cache-threshold` | SMG | Minimum prefix overlap ratio before cache affinity overrides load. Default 0.5. Lower for more aggressive cache stickiness. |
| `--balance-abs-threshold` / `--balance-rel-threshold` | SMG | Imbalance triggers. When workers diverge in load past both thresholds, the policy switches to shortest-queue regardless of cache. |
| `hwm` | SGLang `--kv-events-config` | Raise if you see SGLang logs reporting dropped events under bursty load. |
| `buffer_steps` | SGLang `--kv-events-config` | Raise if SMG ever reports gap-detected reconnects on its KV event stream. |

---

## Caveats

- **gRPC only.** Event-driven routing requires a gRPC worker — `smg-grpc-servicer` is the bridge that turns SGLang's in-process ZMQ feed into a gRPC server-streaming surface SMG can subscribe to. HTTP workers fall back to the approximate string tree automatically.
- **Per-worker block size assumed homogeneous within a model.** If you mix workers serving the same model with different `--page-size` values, the policy uses whichever block size the most recent event reported. Keep page sizes homogeneous within a model.
- **`mesh` mode synchronizes the approximate trees, not events.** When multiple SMG instances cluster via `--enable-mesh`, the event-driven indexer is local to each gateway. Each gateway independently subscribes to each worker.
- **No replay on reconnect today.** SMG reconnects with exponential backoff on stream drops, but does not currently consume SGLang's `replay_endpoint`. A drop window may briefly degrade routing to load-only until events resume.

---

## Reference

- Policy implementation: `model_gateway/src/policies/cache_aware.rs`
- Event subscription manager: `model_gateway/src/worker/kv_event_monitor.rs`
- KV event proto: `crates/grpc_client/proto/common.proto` (messages `KvEventBatch`, `KvCacheEvent`, `KvBlocksStored`, `KvBlocksRemoved`)
- Servicer bridge: `grpc_servicer/smg_grpc_servicer/sglang/servicer.py` (`SubscribeKvEvents`)
- SGLang upstream config: `python/sglang/srt/disaggregation/kv_events.py` (class `KVEventsConfig`)
