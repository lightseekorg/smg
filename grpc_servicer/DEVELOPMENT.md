# smg-grpc-servicer Development Guide

## Installing with vLLM and KV Cache Event Streaming

There are three paths depending on your starting point. Pick the one that
matches your situation.

---

### Path 1 — Upstream vLLM (fastest, no source build)

Use this when you are happy with a released vLLM wheel and only need to add the
SMG gRPC servicer on top.

```bash
# 1. Install vLLM (or use the official vllm/vllm-openai Docker image)
pip install vllm

# 2. Install the gRPC packages
pip install smg-grpc-proto smg-grpc-servicer

# 3. Start the vLLM gRPC server with KV cache events enabled
python -m vllm.entrypoints.grpc_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 50051 \
  --kv-events-config '{
    "enable_kv_cache_events": true,
    "publisher": "zmq",
    "endpoint": "tcp://*:5557",
    "replay_endpoint": "tcp://*:5558",
    "buffer_steps": 10000,
    "topic": ""
  }'
```

SMG can then route cache-aware requests by calling `SubscribeKvEvents` on that
gRPC endpoint.

---

### Path 2 — Development from SMG source

Use this when you are modifying `smg-grpc-proto` or `smg-grpc-servicer` and
want changes picked up immediately without a publish/install cycle.

```bash
# 1. Install vLLM nightly (recommended — latest gRPC surface)
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cu129

# 2. Install both gRPC packages as editable from the SMG repo root
uv pip install -e crates/grpc_client/python/
uv pip install -e grpc_servicer/

# 3. Run the server as in Path 1
```

The editable installs mean any edit to proto files or servicer code is live
immediately. Proto stubs are regenerated automatically on each `pip install -e`
invocation (handled by the custom `setup.py` build hook).

---

### Path 3 — Custom vLLM built from source

Use this when you are working on a patched or unreleased vLLM and need to
install it from a local clone inside a pre-existing vLLM Docker image.

The key trick: `VLLM_USE_PRECOMPILED=1` tells vLLM's build system to reuse
the CUDA kernels already compiled into the base image instead of recompiling
them, which would take 30+ minutes and requires the full CUDA toolkit.

```bash
# 1. Clone vLLM (or check out your fork/branch)
git clone https://github.com/vllm-project/vllm.git /opt/vllm-src
cd /opt/vllm-src
git checkout <your-branch-or-commit>

# 2. Install vLLM over the base image's copy, reusing precompiled kernels
VLLM_USE_PRECOMPILED=1 pip install \
  --no-deps \
  --force-reinstall \
  --editable .

# 3. Reinstall the gRPC packages
#    --force-reinstall above can displace editable installs or .pth entries,
#    so always reinstall smg-grpc-proto and smg-grpc-servicer afterwards.
pip install smg-grpc-proto smg-grpc-servicer
# or from source if you are also modifying those:
pip install -e /path/to/smg/crates/grpc_client/python/
pip install -e /path/to/smg/grpc_servicer/

# 4. Run the server as in Path 1
```

**Why reinstall the gRPC packages after step 2?**
`pip install --force-reinstall --editable` rewrites the `easy-install.pth` /
direct-url metadata for the vLLM site-packages entry. This can invalidate
other editable installs that were registered before the force-reinstall ran.
Reinstalling `smg-grpc-proto` and `smg-grpc-servicer` last ensures their
`.pth` entries are current and their proto stubs are freshly generated.

---

### KV cache events — what `--kv-events-config` does

| Field | Purpose |
|---|---|
| `enable_kv_cache_events` | Must be `true` to activate the ZMQ publisher inside vLLM |
| `publisher` | Must be `"zmq"` — the only publisher the bridge understands |
| `endpoint` | ZMQ PUB bind address; one port per data-parallel rank (rank N gets port + N) |
| `replay_endpoint` | ZMQ REQ/REP address for history replay on reconnect |
| `buffer_steps` | How many batches vLLM keeps in its replay ring buffer |
| `topic` | ZMQ topic prefix for filtering; `""` means receive all events |

The `VllmKvEventBridge` inside `smg-grpc-servicer` subscribes to these ZMQ
sockets, translates vLLM's internal event types
(`BlockStored` / `BlockRemoved` / `AllBlocksCleared`) into the
`smg.grpc.common.KvEventBatch` proto, and streams them to SMG via the
`SubscribeKvEvents` gRPC RPC.

---

## Local Development

Install both proto and servicer as editable packages:

```bash
pip install -e crates/grpc_client/python/
pip install -e grpc_servicer/
```

No version concerns locally — editable installs always use the latest source.

## Containerized DX Smoke Test

To simulate a new developer starting from a clean Linux environment, use the
containerized smoke harness from the repository root:

```bash
scripts/run_vllm_grpc_servicer_smoke.sh
```

That flow builds a fresh image, installs `vllm`, `smg-grpc-proto`, and
`smg-grpc-servicer` from the current checkout, verifies imports, and checks that
`python -m vllm.entrypoints.grpc_server --help` works.

If you have GPU-backed hardware and a model available, you can also smoke-test
real gRPC server startup:

```bash
scripts/run_vllm_grpc_servicer_smoke.sh \
  --mode serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --run-arg=--gpus=all
```

Notes:

- The wrapper defaults to `--platform linux/amd64` because upstream vLLM wheels
  are typically published for that target.
- To start from an upstream vLLM image instead of a plain Python base, pass
  `--base-image <vllm-image> --use-base-vllm`. That path is useful for testing
  the developer experience of layering `smg-grpc-servicer` onto an existing
  vLLM container.
- If your environment cannot reach Docker Hub, pass `--base-image <image>` to
  reuse a local/preloaded Python image. Standard proxy variables are forwarded
  automatically during both `docker build` and `docker run`.
- The serve mode checks that the gRPC port opens; it is meant as a startup
  smoke test, not a full inference benchmark.

## CI — vLLM

PR tests install both `smg-grpc-proto` and `smg-grpc-servicer` from source (not PyPI),
so changes to either package are always tested against the PR's code.

This is handled in `scripts/ci_install_vllm.sh`:

```bash
uv pip install vllm
uv pip install -e crates/grpc_client/python/
uv pip install -e grpc_servicer/
```

## CI — SGLang

PR tests install `smg-grpc-servicer` from source alongside SGLang:

```bash
uv pip install "sglang[all]"
uv pip install -e crates/grpc_client/python/
uv pip install -e grpc_servicer/
```

This is handled in `scripts/ci_install_sglang.sh`.

## Release Process

Both `smg-grpc-proto` and `smg-grpc-servicer` are published to PyPI via GitHub Actions.
Releases are triggered by version bumps in the respective `pyproject.toml` files on `main`.

### Scenario 1: Changing only the servicer

1. Make changes to servicer code
2. Bump version in `grpc_servicer/pyproject.toml`
3. Open PR, CI tests pass (both packages installed from source)
4. Merge to `main` — `release-grpc-servicer.yml` fires, publishes to PyPI

### Scenario 2: Changing only the proto

1. Make changes to proto files and regenerate stubs
2. Bump version in `crates/grpc_client/python/pyproject.toml`
3. Open PR, CI tests pass
4. Merge to `main` — `release-grpc-proto.yml` fires, publishes to PyPI

### Scenario 3: Changing both proto and servicer (one PR)

1. Make all changes (proto + servicer) in a single PR
2. Bump versions in both `crates/grpc_client/python/pyproject.toml` and `grpc_servicer/pyproject.toml`
3. Open PR, CI tests pass (both installed from source)
4. Merge to `main` — both release workflows trigger
5. `release-grpc-servicer.yml` waits for `release-grpc-proto.yml` to complete via `workflow_run`
6. Proto publishes first, then servicer publishes (can resolve the new proto from PyPI)

### Scenario 4: Changing both proto and servicer (two PRs)

1. Open PR 1 with proto changes, bump proto version
2. Merge PR 1 — proto publishes to PyPI
3. Open PR 2 with servicer changes, bump servicer version
4. Merge PR 2 — servicer publishes to PyPI (new proto already available)

## Version Pinning

- `smg-grpc-servicer` pins `smg-grpc-proto >= X.Y.Z` loosely
- This means in Scenario 3, the previous proto version satisfies the requirement during
  the brief window between proto and servicer publishing
- If the servicer uses new proto fields that don't exist in the old version, use Scenario 4
  (two separate PRs) instead
