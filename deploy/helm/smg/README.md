# SMG Helm Chart

Helm chart for deploying the Shepherd Model Gateway (SMG) — a high-performance inference router for LLM deployments.

## Prerequisites

- Kubernetes >= 1.26
- Helm >= 3.12

## Quick Start

```bash
helm install smg deploy/helm/smg \
  --set router.workerUrls[0]=http://worker-1:8000 \
  --set router.workerUrls[1]=http://worker-2:8000
```

## Configuration

See [`values.yaml`](values.yaml) for the full list of configurable parameters.

### Key Sections

| Section | Description |
|---------|-------------|
| `global` | Image registry, pull secrets |
| `router` | Router deployment, routing policy, networking, observability |
| `auth` | API key authentication, rate limiting |
| `history` | Storage backend (none/memory/postgres/redis/oracle) |
| `serviceAccount` | Service account creation and annotations |
| `rbac` | RBAC for Kubernetes service discovery |

### Routing Policies

`cache_aware` (default), `round_robin`, `power_of_two`, `consistent_hashing`, `prefix_hash`, `manual`, `random`, `bucket`

## Examples

| File | Scenario |
|------|----------|
| [`router-only.yaml`](examples/router-only.yaml) | Minimal router with external workers |
| [`with-postgres.yaml`](examples/with-postgres.yaml) | PostgreSQL history backend |
| [`with-service-discovery.yaml`](examples/with-service-discovery.yaml) | K8s auto-discovery |
| [`with-ingress.yaml`](examples/with-ingress.yaml) | Ingress with TLS |
| [`with-monitoring.yaml`](examples/with-monitoring.yaml) | ServiceMonitor + Grafana dashboard |

## Testing

```bash
helm test smg
```

## Upgrading

### 0.1.0

Initial release.

## Troubleshooting

### Service discovery not finding pods

Verify RBAC is enabled and the selector matches your worker pod labels:

```bash
kubectl get role,rolebinding -l app.kubernetes.io/instance=smg
kubectl get pods -l <your-selector>
```
