# mock-worker

Multi-port mock HTTP/gRPC inference workers for scale-testing the SMG gateway's
routing and async-runtime behavior. One process hosts many protocol-accurate
stand-ins for vLLM/SGLang engines; all responses are canned (no real model).

## What it implements

**HTTP** (vLLM/SGLang HTTP surface the gateway probes and routes to):
- `GET /health` → `200 OK` (gates registration + health promotion)
- `GET /v1/models` → one model, `owned_by: sglang` (backend/model detection)
- `POST /v1/chat/completions` · `/v1/completions` · `/generate` — non-stream JSON
  or SSE (`data: {chunk}\n\n … data: [DONE]\n\n`)
- `GET /v1/loads?include=core` → `WorkerLoadResponse` (load-aware policies)

**gRPC** (TokenSpeed scheduler — the gateway tokenizes, the worker speaks token
ids): `HealthCheck`, `GetModelInfo`, `GetServerInfo`, `Generate` (streamed
chunks + complete), `GetLoads`, `Abort`; admin RPCs return `unimplemented`.

## Run

```bash
cargo run --release -p mock-worker -- \
  --http-base-port 9000 --http-count 2000 \
  --grpc-base-port 19000 --grpc-count 0 \
  --model mock-model --gen-ms 5
```

Each worker is one port. Register them against an IGW gateway with
`POST /workers` (`{"url":"http://127.0.0.1:9000"}`, or `grpc://…` with
`connection_mode`/`runtime`/`models` for gRPC).

## Scale-test rig

`scripts/scale_test.sh` launches an IGW gateway, starts the mock fleet,
REST-registers it, and samples the gateway PID's CPU + `/health` latency at idle
and under load (`scripts/scale_load.py`):

```bash
scripts/scale_test.sh --http 2000 --policy cache_aware --rps 500 --duration 30
scripts/scale_test.sh --grpc 1000 --policy least_load
```

gRPC runs use `--disable-tokenizer-autoload` so registration/routing can be
measured without a real tokenizer (generation itself is not exercised).
