---
title: Service Discovery
---

# Service Discovery

SMG can automatically discover workers in Kubernetes by watching pods with label selectors. Workers are registered and removed as pods scale up or down — no manual URL management needed.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- A Kubernetes cluster with worker pods deployed
- `kubectl` configured for your cluster

</div>

---

## Basic Setup

Enable service discovery with a label selector that matches your worker pods:

```bash
smg \
  --service-discovery \
  --selector app=sglang-worker \
  --service-discovery-namespace inference \
  --service-discovery-port 8000
```

SMG watches for pods matching the selector and automatically adds or removes workers.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--service-discovery` | `false` | Enable Kubernetes service discovery |
| `--selector` | — | Label selector for worker pods (required) |
| `--service-discovery-namespace` | `default` | Kubernetes namespace to watch |
| `--service-discovery-port` | `8000` | Port to use for worker connections |
| `--service-discovery-protocol` | `http` | Protocol: `http` or `grpc` |

---

## Label Selectors

### Single Label

```bash
smg --service-discovery --selector app=vllm
```

### Multiple Labels

```bash
smg --service-discovery --selector "app=sglang,environment=production"
```

Matches pods with both labels.

### Set-Based Selectors

```bash
smg --service-discovery --selector "app in (sglang, vllm),tier=inference"
```

---

## PD Disaggregation Discovery

For prefill-decode deployments, use separate selectors:

```bash
smg \
  --service-discovery \
  --pd-disaggregation \
  --prefill-selector "app=sglang,role=prefill" \
  --decode-selector "app=sglang,role=decode" \
  --service-discovery-namespace inference
```

Label your pods accordingly:

```yaml
# Prefill worker pod
metadata:
  labels:
    app: sglang
    role: prefill

# Decode worker pod
metadata:
  labels:
    app: sglang
    role: decode
```

---

## RBAC

SMG needs permissions to watch pods. Apply these resources to your cluster:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: smg
  namespace: inference
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: smg-discovery
  namespace: inference
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: smg-discovery
  namespace: inference
subjects:
  - kind: ServiceAccount
    name: smg
    namespace: inference
roleRef:
  kind: Role
  name: smg-discovery
  apiGroup: rbac.authorization.k8s.io
```

For cross-namespace discovery, use a `ClusterRole` and `ClusterRoleBinding` instead.

---

## Deployment Example

### SMG Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smg
  namespace: inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: smg
  template:
    metadata:
      labels:
        app: smg
    spec:
      serviceAccountName: smg
      containers:
        - name: smg
          image: ghcr.io/lightseekorg/smg:latest
          args:
            - --service-discovery
            - --selector=app=sglang-worker
            - --service-discovery-namespace=inference
            - --service-discovery-port=8000
            - --policy=cache_aware
          ports:
            - containerPort: 8000
              name: http
```

### Worker StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: sglang-worker
  namespace: inference
spec:
  serviceName: sglang-worker
  replicas: 3
  selector:
    matchLabels:
      app: sglang-worker
  template:
    metadata:
      labels:
        app: sglang-worker
    spec:
      containers:
        - name: sglang
          image: lmsysorg/sglang:latest
          args:
            - --model-path=meta-llama/Llama-3.1-8B-Instruct
            - --port=8000
          ports:
            - containerPort: 8000
```

---

## Verify

```bash
# Check discovered workers
curl http://localhost:30000/workers | jq

# Check pod labels match selector
kubectl get pods -n inference -l app=sglang-worker

# Verify RBAC permissions
kubectl auth can-i watch pods -n inference --as=system:serviceaccount:inference:smg
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| No workers discovered | Wrong selector | Verify labels match: `kubectl get pods -l <selector>` |
| RBAC error | Missing permissions | Apply Role and RoleBinding above |
| Workers not ready | Health check failing | Check worker health endpoint |
| Stale workers | Watch disconnected | Check Kubernetes API connectivity |

---

## Next Steps

- [Service Discovery Concepts](../concepts/architecture/service-discovery.md) — Worker lifecycle, monitoring metrics, cross-namespace discovery
- [Load Balancing](load-balancing.md) — Choose a routing policy for discovered workers
- [PD Disaggregation](pd-disaggregation.md) — Full PD setup with SGLang and vLLM
