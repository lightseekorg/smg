---
title: High Availability
---

# High Availability

SMG supports high-availability cluster deployments using mesh networking for fault tolerance, scalability, and zero-downtime updates.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-shield-check: Fault Tolerance

Continue serving requests when individual router nodes fail. Automatic failover with zero manual intervention.

</div>

<div class="card" markdown>

### :material-arrow-expand-all: Scalability

Distribute load across multiple router instances. Add nodes without downtime.

</div>

<div class="card" markdown>

### :material-sync: State Synchronization

Share worker states, policy configurations, and rate limits across the cluster in real-time.

</div>

<div class="card" markdown>

### :material-rocket-launch: Zero Downtime Updates

Perform rolling updates without service interruption. Graceful shutdown with request draining.

</div>

</div>

---

## Mesh Architecture

<div class="architecture-diagram">
  <img src="../../../assets/images/mesh-architecture.svg" alt="SMG Mesh Architecture">
</div>

<div class="grid" markdown>

<div class="card" markdown>

### :material-connection: Gossip Protocol

SWIM-based protocol for membership and failure detection.

- 1-second heartbeat interval
- Automatic peer discovery
- Failure detection in seconds

</div>

<div class="card" markdown>

### :material-crown: Cluster Coordination

Node coordination for cluster operations.

- Membership tracking
- Node status management
- Graceful shutdown coordination

</div>

<div class="card" markdown>

### :material-database-sync: CRDT Stores

Conflict-free Replicated Data Types for eventual consistency.

- No coordination locks
- Partition tolerant
- Automatic conflict resolution

</div>

<div class="card" markdown>

### :material-share-variant: State Replication

Real-time synchronization of all cluster state.

- Worker registry
- Rate limit counters
- Cache-aware routing trees

</div>

</div>

---

## Configuration

### Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-mesh` | `false` | Enable mesh networking for HA deployments |
| `--mesh-server-name` | (auto) | Unique identifier for this node in the cluster |
| `--mesh-host` | `0.0.0.0` | Host address for mesh communication |
| `--mesh-port` | `39527` | Port for mesh gRPC communication |
| `--mesh-peer-urls` | (none) | Initial peer URLs for cluster bootstrap |

### Basic Configuration

<div class="grid" markdown>

<div class="card" markdown>

**Node 1** (Bootstrap)

```bash
smg --enable-mesh \
    --mesh-server-name node1 \
    --mesh-host 0.0.0.0 \
    --mesh-port 39527 \
    --host 0.0.0.0 \
    --port 8000
```

</div>

<div class="card" markdown>

**Node 2** (Join)

```bash
smg --enable-mesh \
    --mesh-server-name node2 \
    --mesh-port 39527 \
    --mesh-peer-urls "node1:39527" \
    --host 0.0.0.0 \
    --port 8000
```

</div>

<div class="card" markdown>

**Node 3** (Join)

```bash
smg --enable-mesh \
    --mesh-server-name node3 \
    --mesh-port 39527 \
    --mesh-peer-urls "node1:39527,node2:39527" \
    --host 0.0.0.0 \
    --port 8000
```

</div>

</div>

### Environment Variables

```bash
export SMG_ENABLE_MESH=true
export SMG_MESH_SERVER_NAME=node1
export SMG_MESH_HOST=0.0.0.0
export SMG_MESH_PORT=39527
export SMG_MESH_PEER_URLS="node1:39527,node2:39527"
```

---

## Gossip Protocol

### State Synchronization

SMG uses a SWIM-based gossip protocol for cluster membership and state propagation:

1. **Ping/Ping-Req**: Each node periodically pings random peers to check health
2. **State Sync**: Healthy nodes exchange state information during pings
3. **Failure Detection**: Unreachable nodes are marked as suspected, then down
4. **Broadcast**: Status changes are broadcast to all cluster members

### Node Status States

| Status | Description |
|--------|-------------|
| `INIT` | Node is starting up |
| `ALIVE` | Node is healthy and reachable |
| `SUSPECTED` | Node may be unreachable (failed ping) |
| `DOWN` | Node confirmed unreachable (failed ping-req) |
| `LEAVING` | Node is gracefully shutting down |

### Failure Detection Timing

| Phase | Duration | Action |
|-------|----------|--------|
| Ping | 1s interval | Direct probe to peer |
| Down | After missed pings | Remove from active cluster |

---

## State Synchronization

### Synchronized State Types

<div class="grid" markdown>

<div class="card" markdown>

### :material-server: Worker Registry

All nodes share worker discovery and health status.

- Worker URLs and metadata
- Health check results
- Circuit breaker states

</div>

<div class="card" markdown>

### :material-speedometer: Rate Limits

Cluster-wide rate limiting coordination.

- Token bucket state
- Request counters
- Quota synchronization

</div>

<div class="card" markdown>

### :material-tree: Routing Trees

Cache-aware routing state shared across nodes.

- Radix tree operations
- Prefix match data
- LRU eviction coordination

</div>

<div class="card" markdown>

### :material-cog: Policy State

Routing policy configuration and state.

- Policy parameters
- Load balancing weights
- Session affinity mappings

</div>

</div>

### CRDT Implementation

SMG uses several CRDT types for conflict-free synchronization:

| CRDT Type | Used For | Merge Strategy |
|-----------|----------|----------------|
| G-Counter | Request counts | Sum of all increments |
| PN-Counter | Token buckets | Sum of positive and negative |
| LWW-Register | Worker state | Last-writer-wins by timestamp |
| OR-Set | Worker sets | Union with tombstones |

---

## Deployment Patterns

### Three-Node Cluster (Minimum HA)

<div class="grid" markdown>

<div class="card" markdown>

**Characteristics**

- Tolerates 1 node failure
- Quorum of 2 for leader election
- Recommended for most deployments

</div>

<div class="card" markdown>

**Configuration**

```bash
# All nodes
smg --enable-mesh \
    --mesh-peer-urls "node1:39527,node2:39527,node3:39527" \
    --worker-urls http://worker1:8000,http://worker2:8000
```

</div>

</div>

### Five-Node Cluster (Higher Availability)

<div class="grid" markdown>

<div class="card" markdown>

**Characteristics**

- Tolerates 2 node failures
- Quorum of 3 for leader election
- Suitable for critical workloads

</div>

<div class="card" markdown>

**Configuration**

```bash
# All nodes
smg --enable-mesh \
    --mesh-peer-urls "node1:39527,node2:39527,node3:39527,node4:39527,node5:39527" \
    --worker-urls http://worker1:8000,http://worker2:8000
```

</div>

</div>

---

## Kubernetes Deployment

### StatefulSet Configuration

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: smg
spec:
  serviceName: smg-mesh
  replicas: 3
  selector:
    matchLabels:
      app: smg
  template:
    metadata:
      labels:
        app: smg
    spec:
      containers:
      - name: smg
        image: ghcr.io/lightseekorg/smg:latest
        args:
        - --enable-mesh
        - --mesh-server-name=$(POD_NAME)
        - --mesh-host=0.0.0.0
        - --mesh-port=39527
        - --mesh-peer-urls=smg-0.smg-mesh:39527,smg-1.smg-mesh:39527,smg-2.smg-mesh:39527
        - --worker-urls=$(WORKER_URLS)
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 39527
          name: mesh
```

### Headless Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: smg-mesh
spec:
  clusterIP: None
  selector:
    app: smg
  ports:
  - port: 39527
    name: mesh
```

---

## HA Management API

### Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ha/health` | GET | Node health status |
| `/ha/status` | GET | Cluster status information |
| `/ha/workers` | GET | Worker states across cluster |
| `/ha/policies` | GET | Policy states across cluster |
| `/ha/shutdown` | POST | Graceful shutdown trigger |

### Cluster Status Response

```json
{
  "node_name": "node1",
  "node_count": 3,
  "nodes": [
    {"name": "node1", "status": "ALIVE", "address": "node1:39527"},
    {"name": "node2", "status": "ALIVE", "address": "node2:39527"},
    {"name": "node3", "status": "ALIVE", "address": "node3:39527"}
  ],
  "stores": {
    "workers": {"entry_count": 5, "last_sync": "2024-01-15T10:30:00Z"},
    "policies": {"entry_count": 2, "last_sync": "2024-01-15T10:30:00Z"}
  }
}
```

---

## Monitoring

### Mesh Metrics

| Metric | Description |
|--------|-------------|
| `smg_mesh_peers_total` | Number of connected peers |
| `smg_mesh_peer_status` | Status of each peer (1=alive, 0=down) |
| `smg_mesh_sync_operations_total` | State sync operations by type |
| `smg_mesh_sync_latency_seconds` | State sync latency histogram |
| `smg_mesh_leader_elections_total` | Leader election events |
| `smg_mesh_gossip_messages_total` | Gossip messages sent/received |

### Alerting Rules

```yaml
groups:
- name: smg-mesh
  rules:
  - alert: SMGClusterDegraded
    expr: smg_mesh_peers_total < 2
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "SMG cluster has fewer than 3 nodes"

  - alert: SMGNodeDown
    expr: smg_mesh_peer_status == 0
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "SMG mesh node {{ $labels.peer }} is down"
```

---

## Best Practices

<div class="grid" markdown>

<div class="card" markdown>

### :material-numeric-3-circle: Odd Node Counts

Use 3, 5, or 7 nodes to avoid split-brain scenarios during network partitions.

</div>

<div class="card" markdown>

### :material-earth: Availability Zones

Distribute nodes across availability zones for resilience against zone failures.

</div>

<div class="card" markdown>

### :material-network: Network Latency

Keep mesh nodes in the same region (< 10ms RTT) for optimal state sync performance.

</div>

<div class="card" markdown>

### :material-monitor: Monitoring

Monitor `smg_mesh_peers_total` and alert when cluster size drops below threshold.

</div>

</div>

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Node stuck in INIT | Cannot reach peers | Check firewall rules for mesh port |
| Frequent leader elections | Network instability | Increase gossip timeouts |
| State inconsistency | Clock skew | Synchronize NTP across nodes |
| High sync latency | Large state | Increase sync interval |

### Debug Logging

```bash
RUST_LOG=smg::mesh=debug smg --enable-mesh ...
```

### Verify Cluster Health

```bash
# Check cluster status
curl http://node1:8000/ha/status | jq

# Check individual node health
curl http://node1:8000/ha/health | jq

# Check worker states
curl http://node1:8000/ha/workers | jq

# Check policy states
curl http://node1:8000/ha/policies | jq
```

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-power: Graceful Shutdown

Allow in-flight requests to complete during shutdown.

[Graceful Shutdown →](../reliability/graceful-shutdown.md)

</div>

<div class="card" markdown>

### :material-electric-switch: Circuit Breakers

Isolate failing workers to prevent cascade failures.

[Circuit Breakers →](../reliability/circuit-breakers.md)

</div>

<div class="card" markdown>

### :material-chart-box: Metrics Reference

Complete list of mesh networking metrics.

[Metrics Reference →](../../reference/metrics.md)

</div>

</div>
