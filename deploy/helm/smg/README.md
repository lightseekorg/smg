# SMG Helm Chart

Helm chart for deploying the Shepherd Model Gateway (SMG) — a high-performance inference router for LLM deployments.

## Prerequisites

- Kubernetes >= 1.26
- Helm >= 3.12
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/) (for worker modes)

## Quick Start

### Router Only (connect to existing workers)

```bash
helm install smg deploy/helm/smg \
  --set router.workerUrls[0]=http://worker-1:8000 \
  --set router.workerUrls[1]=http://worker-2:8000
```

### Router + Workers

```bash
helm install smg deploy/helm/smg \
  --set mode=router-worker \
  --set worker.model=meta-llama/Llama-3-70b \
  --set worker.backend=sglang \
  --set worker.replicas=2 \
  --set worker.resources.limits.nvidia\\.com/gpu=4
```

### Router + PD Disaggregation

```bash
helm install smg deploy/helm/smg \
  --set mode=router-pd \
  --set pd.prefill.model=meta-llama/Llama-3-70b \
  --set pd.prefill.replicas=2 \
  --set pd.decode.model=meta-llama/Llama-3-70b \
  --set pd.decode.replicas=4
```

## Deployment Modes

| Mode | Description |
|------|-------------|
| `router` (default) | Router only — connects to pre-existing inference workers |
| `router-worker` | Router + inference worker Deployments |
| `router-pd` | Router + prefill/decode disaggregated worker Deployments |

## Configuration

See [`values.yaml`](values.yaml) for the full list of configurable parameters.

### Key Sections

| Section | Description |
|---------|-------------|
| `global` | Image registry, pull secrets |
| `router` | Router deployment, routing policy, networking, observability |
| `auth` | API key authentication, rate limiting |
| `history` | Storage backend (none/memory/postgres/redis/oracle) |
| `worker` | Worker config for `router-worker` mode |
| `pd.prefill` / `pd.decode` | PD disaggregation config for `router-pd` mode |
| `serviceAccount` | Service account creation and annotations |
| `rbac` | RBAC for Kubernetes service discovery |

### Routing Policies

`cache_aware` (default), `round_robin`, `power_of_two`, `consistent_hashing`, `prefix_hash`, `manual`, `random`, `bucket`

### Inference Backends

| Backend | Default Image | Notes |
|---------|--------------|-------|
| `sglang` | `lmsysorg/sglang:latest` | Default, best cache-aware support |
| `vllm` | `vllm/vllm-openai:latest` | Widely adopted |
| `trtllm` | User-provided | NVIDIA TensorRT-LLM |

## Examples

| File | Scenario |
|------|----------|
| [`router-only.yaml`](examples/router-only.yaml) | Minimal router with external workers |
| [`router-worker-sglang.yaml`](examples/router-worker-sglang.yaml) | Router + sglang workers with GPUs |
| [`router-worker-vllm.yaml`](examples/router-worker-vllm.yaml) | Router + vLLM workers |
| [`router-worker-trtllm.yaml`](examples/router-worker-trtllm.yaml) | Router + TensorRT-LLM workers |
| [`router-pd.yaml`](examples/router-pd.yaml) | PD disaggregation |
| [`with-postgres.yaml`](examples/with-postgres.yaml) | PostgreSQL history backend |
| [`with-service-discovery.yaml`](examples/with-service-discovery.yaml) | K8s auto-discovery |
| [`with-ingress.yaml`](examples/with-ingress.yaml) | Ingress with TLS |
| [`with-monitoring.yaml`](examples/with-monitoring.yaml) | ServiceMonitor + Grafana dashboard |
| [`production.yaml`](examples/production.yaml) | Production-hardened with HPA, PDB, monitoring |

## Testing

```bash
helm test smg
```

## Upgrading

### 0.1.0

Initial release.

## Troubleshooting

### Workers not scheduling

Ensure the NVIDIA GPU Operator is installed and GPU nodes have the expected labels:

```bash
kubectl get nodes -l nvidia.com/gpu.product
```

### Router can't reach workers

Check that worker pods are running and the Service endpoints are populated:

```bash
kubectl get endpoints -l app.kubernetes.io/component=worker
```

### Service discovery not finding pods

Verify RBAC is enabled and the selector matches your worker pod labels:

```bash
kubectl get role,rolebinding -l app.kubernetes.io/instance=smg
kubectl get pods -l <your-selector>
```
