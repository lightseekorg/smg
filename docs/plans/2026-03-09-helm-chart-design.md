# SMG Helm Chart Design

## Overview

A single Helm chart at `deploy/helm/smg/` that deploys the Shepherd Model Gateway (SMG) in one of three user-selectable modes:

- **`router`** — Router only, connects to pre-existing inference workers
- **`router-worker`** — Router + inference worker Deployments
- **`router-pd`** — Router + prefill/decode disaggregated worker Deployments

## Goals

- Extremely user-friendly: one `helm install` command for any mode
- Easy to maintain: single chart, no sub-chart complexity
- Self-documenting: heavily commented values.yaml, JSON Schema validation, example files
- Escape hatches: `extraArgs` and `extraEnv` on every component for unsupported flags

## Architecture

```
              ┌─────────────────────┐
              │   SMG Router        │  ← Always deployed
              │   Port: 30000       │
              │   Metrics: 29000    │
              └─────────┬───────────┘
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Worker1 │    │ Worker2 │    │ Worker3 │  ← mode: router-worker
    │ (GPU)   │    │ (GPU)   │    │ (GPU)   │     OR external workers
    └─────────┘    └─────────┘    └─────────┘

    ┌─────────────────┐  ┌─────────────────┐
    │ Prefill Workers  │  │ Decode Workers   │  ← mode: router-pd
    │ (GPU)            │  │ (GPU)            │
    └─────────────────┘  └─────────────────┘
```

## Deployment Modes

### Mode 1: `router` (default)

Deploys only the SMG router. Workers are external — specified via `router.workerUrls` or discovered via K8s service discovery (`router.serviceDiscovery`).

```bash
helm install smg deploy/helm/smg \
  --set router.workerUrls[0]=http://worker-1:8000 \
  --set router.workerUrls[1]=http://worker-2:8000
```

### Mode 2: `router-worker`

Deploys the router plus a worker Deployment running a chosen inference backend (sglang, vllm, or trtllm). The router is auto-wired to workers via internal Service DNS.

```bash
helm install smg deploy/helm/smg \
  --set mode=router-worker \
  --set worker.model=meta-llama/Llama-3-70b \
  --set worker.backend=sglang \
  --set worker.replicas=2 \
  --set worker.resources.limits.nvidia\.com/gpu=4
```

### Mode 3: `router-pd`

Deploys the router with separate prefill and decode worker Deployments for PD disaggregation. Each pool has independent scaling, policies, and resource allocation.

```bash
helm install smg deploy/helm/smg \
  --set mode=router-pd \
  --set pd.prefill.model=meta-llama/Llama-3-70b \
  --set pd.prefill.replicas=2 \
  --set pd.decode.model=meta-llama/Llama-3-70b \
  --set pd.decode.replicas=4
```

## File Layout

```
deploy/helm/smg/
├── Chart.yaml
├── values.yaml
├── values.schema.json
├── README.md
├── templates/
│   ├── NOTES.txt
│   ├── _helpers.tpl
│   ├── configmap.yaml
│   ├── deployment-router.yaml
│   ├── service-router.yaml
│   ├── serviceaccount.yaml
│   ├── role.yaml
│   ├── rolebinding.yaml
│   ├── hpa-router.yaml
│   ├── ingress.yaml
│   ├── servicemonitor.yaml
│   ├── grafana-dashboard.yaml
│   ├── secret.yaml
│   ├── pdb.yaml
│   ├── worker/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── pd/
│       ├── deployment-prefill.yaml
│       ├── deployment-decode.yaml
│       ├── service-prefill.yaml
│       └── service-decode.yaml
├── examples/
│   ├── router-only.yaml
│   ├── router-worker-sglang.yaml
│   ├── router-worker-vllm.yaml
│   ├── router-worker-trtllm.yaml
│   ├── router-pd.yaml
│   ├── with-postgres.yaml
│   ├── with-service-discovery.yaml
│   ├── with-ingress.yaml
│   ├── with-monitoring.yaml
│   └── production.yaml
└── tests/
    └── test-connection.yaml
```

## Values Design

### Top-Level Mode

```yaml
mode: router  # "router" | "router-worker" | "router-pd"
```

### Key Design Decisions

1. **Flat structure** — `router.policy` not `router.routing.policy.name`
2. **Sensible defaults** — works with just `mode` + `workerUrls` or `worker.model`
3. **existingSecret pattern** — users reference pre-created Secrets instead of inline creds
4. **extraArgs/extraEnv** on every component — escape hatch for any CLI flag
5. **Worker image override** — workers use backend-specific images (sglang/vllm), separate from router image

### Configuration Sections

| Section | Purpose |
|---------|---------|
| `global` | Image registry, pull secrets |
| `router` | Router deployment, routing policy, networking, observability, resilience |
| `auth` | API key, OIDC, rate limiting |
| `history` | Storage backend (none/memory/postgres/redis/oracle) |
| `worker` | Worker deployment for `router-worker` mode |
| `pd.prefill` | Prefill workers for `router-pd` mode |
| `pd.decode` | Decode workers for `router-pd` mode |
| `serviceAccount` | SA creation and annotations |
| `rbac` | ClusterRole for K8s service discovery |
| `grafana` | Grafana dashboard ConfigMap |
| `postgresql` | Optional sub-chart dependency |
| `redis` | Optional sub-chart dependency |

### Router Configuration Highlights

- **Worker discovery**: Both explicit `workerUrls` list and K8s `serviceDiscovery` with label selectors
- **All routing policies**: cache_aware, round_robin, power_of_two, consistent_hashing, prefix_hash, manual, random, bucket
- **Cache tuning**: threshold, balance thresholds, eviction interval, tree size, block size
- **Request handling**: payload size, timeout, concurrency, queue size
- **Resilience**: retry (with backoff), circuit breaker, health checks — all independently toggleable
- **Networking**: Service type, Ingress, custom annotations
- **Observability**: Prometheus metrics port, ServiceMonitor, OTLP tracing, structured logging
- **Autoscaling**: HPA with CPU target
- **PDB**: Pod disruption budget
- **Features**: WASM plugins, MCP config, reasoning/tool-call parsers

### Worker Configuration Highlights

- **Backend selection**: `worker.backend` = sglang | vllm | trtllm
- **Image override**: Workers use backend-specific images, separate from the router image
- **GPU resources**: `nvidia.com/gpu` limits
- **Model storage**: emptyDir (download at start), hostPath, or existing PVC
- **Shared memory**: configurable `/dev/shm` size (default 16Gi)
- **Parallelism**: tensor-parallel and data-parallel size

### PD Configuration Highlights

- **Independent scaling**: prefill and decode pools have separate replica counts
- **Independent policies**: e.g. cache_aware for prefill, power_of_two for decode
- **Bootstrap port**: optional gRPC bootstrap port for prefill nodes
- **Separate resources**: different GPU allocations per pool

## Template Logic

### Conditional Rendering

```yaml
# Worker templates only render when mode=router-worker
{{- if eq .Values.mode "router-worker" }}

# PD templates only render when mode=router-pd
{{- if eq .Values.mode "router-pd" }}
```

### Router CLI Args Assembly

A `_helpers.tpl` helper builds the full CLI args from values, keeping the Deployment template clean:

```
{{ include "smg.routerArgs" . }}
```

The helper constructs args conditionally:
- Always: `--host`, `--port`, `--policy`
- If `workerUrls`: `--worker-urls ...`
- If `serviceDiscovery.enabled`: `--service-discovery`, `--selector`, etc.
- If `mode=router-pd`: `--pd-disaggregation`, `--prefill-policy`, `--decode-policy`
- If `history.backend != memory/none`: backend-specific flags
- Always appends: `extraArgs`

### RBAC Auto-Enable

When `serviceDiscovery.enabled=true`, RBAC resources are created automatically:

```yaml
{{- if or .Values.rbac.create .Values.router.serviceDiscovery.enabled }}
```

ClusterRole requires `pods: [list, watch]` in the target namespace.

### Worker → Router Wiring

In `router-worker` and `router-pd` modes, the router auto-discovers co-deployed workers via internal Service DNS (e.g. `http://{{ .Release.Name }}-worker:8000`). No manual `workerUrls` or service discovery RBAC needed for self-managed workers.

Users can still add additional external `workerUrls` alongside chart-managed workers.

### Secret Management

Three patterns:
1. **Inline value** — e.g. `auth.apiKey: "sk-..."` → chart creates a Secret
2. **Existing secret** — e.g. `auth.apiKeySecret: "my-secret"` → chart references it via `envFrom`
3. **Neither** — feature disabled, no secret mounted

### JSON Schema Validation

`values.schema.json` validates:
- `mode` enum: `router | router-worker | router-pd`
- `worker.backend` enum: `sglang | vllm | trtllm`
- `history.backend` enum: `none | memory | postgres | redis | oracle`
- Required fields per mode (e.g. `worker.model` when `mode=router-worker`)

## Observability

### Prometheus

- Router exposes metrics on port 29000 (`/metrics`)
- Optional `ServiceMonitor` CRD for prometheus-operator auto-discovery
- Configurable scrape interval and labels

### Grafana

- Optional ConfigMap with Grafana dashboard JSON
- Uses sidecar label (`grafana_dashboard: "1"`) for auto-import

### Tracing

- Optional OTLP traces endpoint for OpenTelemetry

### Logging

- Configurable log level and format (text/JSON)
- Optional log directory for file output

## Documentation

### README.md

1. Overview — what SMG is and what this chart deploys
2. Prerequisites — K8s version, Helm version, GPU operator
3. Quick Start — one command per mode
4. Configuration Reference — auto-generated from values.schema.json
5. Examples — links to examples/ with descriptions
6. Upgrading — migration notes
7. Troubleshooting — common issues

### Example Files

| File | Scenario |
|------|----------|
| `router-only.yaml` | Minimal router pointing at worker URLs |
| `router-worker-sglang.yaml` | Router + sglang workers with GPUs |
| `router-worker-vllm.yaml` | Router + vllm workers |
| `router-worker-trtllm.yaml` | Router + TensorRT-LLM workers |
| `router-pd.yaml` | PD disaggregation |
| `with-postgres.yaml` | PostgreSQL history backend |
| `with-service-discovery.yaml` | K8s auto-discovery |
| `with-ingress.yaml` | Ingress with TLS |
| `with-monitoring.yaml` | ServiceMonitor + Grafana dashboard |
| `production.yaml` | Production-hardened with HPA, PDB, resources, monitoring |

### NOTES.txt

Post-install notes showing:
- Deployment mode and key endpoints
- Worker count and backend (if applicable)
- How to run `helm test`

## Dependencies

Optional sub-chart dependencies in Chart.yaml:

```yaml
dependencies:
  - name: postgresql
    version: "~15"
    repository: https://charts.bitnami.com/bitnami
    condition: history.postgres.deploy
  - name: redis
    version: "~19"
    repository: https://charts.bitnami.com/bitnami
    condition: history.redis.deploy
```

## Supported Backends

| Backend | Image | Notes |
|---------|-------|-------|
| sglang | `lmsysorg/sglang:latest` | Default, best cache-aware support |
| vllm | `vllm/vllm-openai:latest` | Widely adopted |
| trtllm | User-provided | NVIDIA TensorRT-LLM |

## Port Summary

| Service | Port | Purpose |
|---------|------|---------|
| Router | 30000 (container), 80 (service) | Main inference endpoint |
| Metrics | 29000 | Prometheus metrics |
| Worker | 8000 | Inference backend |
| Prefill Bootstrap | 9001 | PD disagg bootstrap (optional) |
