---
title: Service Discovery
---

# Service Discovery

SMG automatically discovers and registers workers in Kubernetes environments, eliminating manual worker URL management and enabling dynamic scaling.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-kubernetes: Native Kubernetes

Watch pods matching label selectors with automatic registration and removal.

</div>

<div class="card" markdown>

### :material-sync: Dynamic Scaling

Workers are automatically added and removed as pods scale up or down.

</div>

<div class="card" markdown>

### :material-filter: Label Selectors

Target specific workers using Kubernetes label selectors.

</div>

<div class="card" markdown>

### :material-swap-horizontal: PD Support

Separate discovery for prefill and decode workers in disaggregated deployments.

</div>

</div>

---

## How It Works

<div class="architecture-diagram">
  <img src="../../../assets/images/service-discovery.svg" alt="Service Discovery Architecture">
</div>

### Discovery Flow

1. **Watch Pods**: SMG creates a Kubernetes watcher for pods matching the configured label selector
2. **Filter Events**: Only pods matching the selector (regular or PD mode) are processed
3. **Handle Events**: Pod creation triggers `AddWorker` job, deletion triggers `RemoveWorker` job
4. **Register Workers**: Workers are added to the registry with health checks starting immediately
5. **Track State**: A HashSet tracks discovered pods to prevent duplicate registrations

---

## Configuration

### Basic Setup

```bash
smg \
  --service-discovery \
  --selector app=sglang-worker \
  --service-discovery-namespace inference \
  --service-discovery-port 8000
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--service-discovery` | `false` | Enable Kubernetes service discovery |
| `--selector` | - | Label selector for worker pods (required) |
| `--service-discovery-namespace` | `default` | Kubernetes namespace to watch |
| `--service-discovery-port` | `8000` | Port to use for worker connections |
| `--service-discovery-protocol` | `http` | Protocol for worker connections (`http` or `grpc`) |

### Environment Variables

```bash
export SMG_SERVICE_DISCOVERY=true
export SMG_SELECTOR="app=sglang-worker"
export SMG_SERVICE_DISCOVERY_NAMESPACE=inference
export SMG_SERVICE_DISCOVERY_PORT=8000
```

---

## Label Selectors

SMG uses Kubernetes label selectors to identify worker pods.

### Simple Selector

Match pods with a single label:

```bash
smg --service-discovery --selector app=vllm
```

Matches pods with label `app=vllm`.

### Multiple Labels

Match pods with multiple labels:

```bash
smg --service-discovery --selector "app=sglang,environment=production"
```

Matches pods with both `app=sglang` AND `environment=production`.

### Complex Selectors

Use set-based selectors for more complex matching:

```bash
smg --service-discovery --selector "app in (sglang, vllm),tier=inference"
```

---

## PD Disaggregation Discovery

For prefill-decode disaggregated deployments, use separate selectors for each worker type.

### Configuration

```bash
smg \
  --service-discovery \
  --pd-disaggregation \
  --prefill-selector "app=sglang,role=prefill" \
  --decode-selector "app=sglang,role=decode" \
  --service-discovery-namespace inference
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--prefill-selector` | Label selector for prefill workers |
| `--decode-selector` | Label selector for decode workers |

### Worker Labels

Label your pods appropriately:

```yaml
# Prefill worker
apiVersion: v1
kind: Pod
metadata:
  name: sglang-prefill-0
  labels:
    app: sglang
    role: prefill
spec:
  containers:
    - name: sglang
      image: lmsysorg/sglang:latest
      args: ["--dp-size", "1", "--prefill-only"]

---
# Decode worker
apiVersion: v1
kind: Pod
metadata:
  name: sglang-decode-0
  labels:
    app: sglang
    role: decode
spec:
  containers:
    - name: sglang
      image: lmsysorg/sglang:latest
      args: ["--dp-size", "1", "--decode-only"]
```

---

## Required RBAC

SMG needs permissions to watch pods in the target namespace.

### Role

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: smg-discovery
  namespace: inference
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
```

### RoleBinding

```yaml
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

### ServiceAccount

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: smg
  namespace: inference
```

### Cross-Namespace Discovery

To discover workers across multiple namespaces, use a ClusterRole:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: smg-discovery
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: smg-discovery
subjects:
  - kind: ServiceAccount
    name: smg
    namespace: inference
roleRef:
  kind: ClusterRole
  name: smg-discovery
  apiGroup: rbac.authorization.k8s.io
```

---

## Complete Deployment Example

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
            - containerPort: 3001
              name: admin
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

## Worker Lifecycle

### Registration Flow

1. **Pod Created**: Kubernetes creates a new worker pod
2. **Watch Event**: SMG receives the pod creation event
3. **Capability Query**: SMG queries the worker's `/get_model_info` endpoint
4. **Registration**: Worker is added to the registry
5. **Health Check**: Background health checks begin

### Removal Flow

1. **Pod Terminating**: Kubernetes begins pod termination
2. **Watch Event**: SMG receives the pod deletion event
3. **Drain**: SMG stops sending new requests to the worker
4. **Removal**: Worker is removed from the registry

### Worker States

| State | Description | Receives Traffic |
|-------|-------------|------------------|
| **Registering** | Querying capabilities | No |
| **Ready** | Healthy and registered | Yes |
| **Unhealthy** | Failing health checks | No |
| **Draining** | Pending removal | No |

---

## Monitoring

### Metrics

| Metric | Description |
|--------|-------------|
| `smg_discovered_workers_total` | Total workers discovered |
| `smg_worker_registrations_total` | Worker registration events |
| `smg_worker_removals_total` | Worker removal events |

### Logs

```bash
# Enable discovery debug logging
RUST_LOG=smg::discovery=debug smg --service-discovery ...
```

Example log output:

```
[INFO] Watching pods in namespace 'inference' with selector 'app=sglang-worker'
[INFO] Discovered new pod: sglang-worker-0 (10.0.0.5:8000)
[INFO] Registered worker: http://10.0.0.5:8000
[INFO] Discovered new pod: sglang-worker-1 (10.0.0.6:8000)
[INFO] Registered worker: http://10.0.0.6:8000
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| No workers discovered | Wrong selector | Verify labels match selector |
| RBAC error | Missing permissions | Apply Role and RoleBinding |
| Workers not ready | Health check failing | Check worker health endpoint |
| Stale workers | Watch disconnected | Check Kubernetes API connectivity |

### Verify Discovery

```bash
# Check discovered workers via admin API
curl http://smg:3001/workers | jq

# Check pod labels match selector
kubectl get pods -n inference -l app=sglang-worker

# Verify RBAC
kubectl auth can-i watch pods -n inference --as=system:serviceaccount:inference:smg
```

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-swap-horizontal: PD Disaggregation

Learn about prefill-decode separation.

[PD Disaggregation →](../routing/pd-disaggregation.md)

</div>

<div class="card" markdown>

### :material-scale-balance: Load Balancing

Configure routing policies for discovered workers.

[Load Balancing →](../routing/load-balancing.md)

</div>

<div class="card" markdown>

### :material-heart-pulse: Health Checks

Configure health monitoring for workers.

[Health Checks →](../reliability/health-checks.md)

</div>

</div>
