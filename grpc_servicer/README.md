# smg-grpc-servicer

gRPC servicer implementations for LLM inference engines. Supports vLLM and SGLang.

## Installation

For vLLM:

```bash
pip install smg-grpc-servicer[vllm]
```

For SGLang:

```bash
pip install smg-grpc-servicer[sglang]
```

## Usage

### vLLM

```bash
vllm serve meta-llama/Llama-2-7b-hf --grpc
```

### SGLang

```bash
sglang serve --model-path meta-llama/Llama-2-7b-hf --grpc-mode
```

### Upstream vLLM and KV Event Streaming

`smg-grpc-servicer` can stream KV cache events from upstream vLLM without any
changes to the vLLM repository. To do that, start the stock vLLM gRPC server
with `--kv-events-config` enabled:

```bash
python -m vllm.entrypoints.grpc_server \
  --model meta-llama/Llama-2-7b-hf \
  --host 0.0.0.0 \
  --port 50051 \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558","buffer_steps":10000,"topic":""}'
```

This is only required if you want SMG to consume `SubscribeKvEvents`, for
example with cache-aware routing or KV cache monitoring. Basic gRPC inference
works without it.

If you carry a local vLLM patch that auto-enables KV events for gRPC mode, that
patch is only a convenience feature. It is not required for compatibility with
SMG.

Current limitation: restart recovery is sequence-based. If the backend restarts
and SMG reconnects while the new producer is still behind the previously seen
sequence number, SMG detects the regression, clears that worker's KV state, and
rebuilds from sequence 0. If the backend restarts and catches back up before
SMG reconnects, the protocol has no producer epoch/generation marker, so SMG
cannot prove that a restart happened. A future protocol update should add a
producer instance identifier to `SubscribeKvEvents`, and vLLM null-block
`BlockStored` layouts would need per-block token ranges to support lossless
translation instead of the current fail-closed behavior.

## Architecture

```
smg-grpc-servicer[vllm]    ──optional dep──>  vllm     (lazy import)
smg-grpc-servicer[sglang]  ──optional dep──>  sglang   (lazy import)
smg-grpc-servicer           ──depends on──>  smg-grpc-proto  (hard dependency)
vllm                        ──optional──>    smg-grpc-servicer (via vllm serve --grpc)
sglang                      ──optional──>    smg-grpc-servicer (via --grpc-mode)
```

Backend dependencies are isolated via extras to avoid conflicts between vLLM and SGLang.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for local development setup, CI, and release workflows.
