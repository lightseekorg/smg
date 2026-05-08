# Mesh Crate Structure

This directory is the cluster membership and v2 MeshKV transport crate.

Current production data sync is:

```text
gateway adapters
  -> MeshKV namespaces
     -> CRDT operation-log batches for durable keys
     -> Stream batches for tenant deltas and tree repair traffic
  -> SyncStream between mesh peers
  -> remote MeshKV subscribers
  -> gateway registries / policies
```

The legacy v1 data-sync modules have been removed:

- `sync.rs`: removed `MeshSyncManager`, subscribers, tree checkpointing, and v1 apply helpers.
- `stores.rs`: removed typed v1 CRDT stores, app store, per-actor rate-limit shards, and tree buffers.
- `collector.rs`: removed v1 store collector and peer watermark filtering.
- `tree_ops.rs`: removed legacy `TreeState`, `TreeOperation`, `TenantDelta`, and compression helpers.
- `rate_limit_window.rs`: removed v1 periodic counter reset task.
- `consistent_hash.rs`: removed v1 rate-limit ownership ring.
- `tests/comprehensive.rs`, `model_gateway/tests/mesh_integration_test.rs`, and `benches/mesh_serialization.rs`: removed tests/benchmarks that exercised the deleted v1 path.

## Wire Shape

`proto/gossip.proto` now has only the v2 stream payloads:

- `HEARTBEAT`: keep the bidirectional stream alive.
- `ACK` / `NACK`: lightweight delivery acknowledgement.
- `STREAM_BATCH`: ephemeral `StreamNamespace` entries such as `td:`, `tree:req:`, and `tree:page:`.
- `CRDT_BATCH`: explicit `CrdtBatchEntry` records derived from the shared `CrdtOrMap`.

`CRDT_BATCH` is intentionally a transport detail. Application code should not build CRDT payloads directly; it should write through `CrdtNamespace`. The internal operation log remains a `CrdtOrMap` implementation detail, while the wire format carries explicit key/value/tombstone/timestamp/replica entries so deletes and conflict ordering stay visible at the protocol boundary.

## Current Files

| File | Status | Responsibility | Keep/Rename Notes |
| --- | --- | --- | --- |
| `lib.rs` | Active | Public crate surface. Re-exports `MeshKV`, namespace handles, CRDT helpers, mesh server types, shared gateway-facing data types, and tree hash helpers. | Keep. Keep exports sparse; do not re-export transport internals. |
| `service.rs` | Active | Owns `MeshServerBuilder`, `MeshServer`, `MeshServerHandler`, cluster state, ping helper, startup/shutdown, mTLS wiring, controller/service construction, and `mesh_run!`. | Keep. Could later split builder/handler/server for readability, but no v1 code remains. |
| `controller.rs` | Active | Outbound gossip controller. Chooses peers, sends SWIM-style pings, owns outbound `SyncStream` connections, drains MeshKV once per round, and sends CRDT/stream batches to peers. | Rename to `gossip_controller.rs` or `outbound.rs`; it is both a client and stream sender, so `controller` is vague. |
| `ping_server.rs` | Active | Inbound gRPC `Gossip` service. Handles `PingServer`, accepts inbound `SyncStream`, applies incoming CRDT/stream batches, and sends server-side batches back on inbound streams. | Rename to `gossip_service.rs` or `inbound.rs`; it is both a server and a stream sender, so `ping_server` understates its role. |
| `stream_sync.rs` | Active | Shared helpers for SyncStream payload construction/application: heartbeats, ACKs, CRDT batch encode/decode, stream batch chunking, and stream dispatch. | Keep. This was split out so controller and inbound service do not duplicate v2 send/apply logic. |
| `kv.rs` | Active | Generic MeshKV API. Defines `CrdtNamespace`, `StreamNamespace`, drain callbacks, subscriber registry, stream round collection, CRDT entry export, and CRDT merge notification. | Keep. This is the application-agnostic v2 API surface. |
| `crdt_kv/mod.rs` | Active | Internal CRDT module facade. | Keep. Public re-export only what `lib.rs` needs. |
| `crdt_kv/crdt.rs` | Active | Observed-remove map, per-key merge strategies, tombstones, compaction, operation-log merge, and EpochMaxWins application. | Keep. This is durable state convergence. |
| `crdt_kv/operation.rs` | Active | `Operation` and `OperationLog` internals. The stream transport converts logs to explicit `CrdtBatchEntry` records at the wire boundary. | Keep, but keep `Operation` non-public unless a real caller needs it. |
| `crdt_kv/epoch_max_wins.rs` | Active | Binary encoding and merge logic for epoch-scoped counters. Used by rate-limit CRDT keys. | Keep. |
| `crdt_kv/kv_store.rs` | Active | Local key/value store backing `CrdtOrMap`, including generation counters. | Keep. |
| `crdt_kv/merge_strategy.rs` | Active | Merge strategy enum (`LastWriterWins`, `MaxValueWins`, `EpochMaxWins`). | Keep. |
| `crdt_kv/replica.rs` | Active | Replica id and Lamport clock helpers. | Keep. |
| `crdt_kv/tests.rs` | Active tests | Unit tests for CRDT behavior and merge strategies. | Keep. |
| `chunking.rs` | Active | Sender-side stream chunking, stream batch construction, chunk dispatch entry point, and generation ids. | Keep. Might be folded under a future `stream/` module. |
| `chunk_assembler.rs` | Active | Receiver-side chunk reassembly and timeout GC. | Keep. |
| `hash.rs` | Active helper | Stable 64-bit tree prefix hash helpers used by TreeSync tenant deltas and repair-page hash indexes. Also owns `GLOBAL_EVICTION_HASH`. | Keep. Exists because old `tree_ops.rs` was removed but these helpers are still live v2 API. |
| `types.rs` | Active types | Shared gateway-facing types: `WorkerState`, `MembershipState`, `RateLimitConfig`, and rate-limit key constants. | Keep. These are data contracts, not sync implementation. |
| `flow_control.rs` | Active support | Message-size validation and retry/backoff helpers for mesh streams. | Keep. |
| `metrics.rs` | Active support | Metrics declarations plus active stream/convergence/peer metrics. Some old snapshot/store metric functions remain as dormant helpers. | Keep for now; later trim unused snapshot/store functions if dashboards confirm they are obsolete. |
| `mtls.rs` | Active support | mTLS config/cert loading for mesh gRPC. | Keep. |
| `partition.rs` | Active support | Partition/quorum decision helper used by handler readiness/serving checks. | Keep. |
| `node_state_machine.rs` | Dormant active support | Readiness state machine. No longer depends on v1 stores; currently mostly lifecycle scaffolding. | Keep until readiness/snapshot design is revisited. |
| `topology.rs` | Dormant support | Full/sparse topology planner based on cluster state and region/AZ metadata. Not currently the main peer selection path. | Keep for membership hardening; not part of v1 data sync. |
| `proto/gossip.proto` | Active wire contract | gRPC service and v2 stream messages. | Keep. No v1 `StoreType`, snapshots, or incremental store updates remain. |
| `tests/chunking_integration.rs` | Active tests | Chunking and stream dispatch integration tests. | Keep. |
| `tests/test_utils.rs` | Active tests | Ephemeral bind helper, polling helper, and cluster-state helper. | Keep. V1 store/sync helpers were removed. |
| `tests/mod.rs` | Active tests | Test module wiring. | Keep. |

## Proposed Directory Structure

The current flat file layout is workable, but the names still mix roles. A cleaner structure would group by subsystem and make directionality explicit:

```text
src/
  lib.rs
  types.rs
  hash.rs

  server/
    mod.rs              # current service.rs public builder/handler/server
    inbound.rs          # current ping_server.rs
    outbound.rs         # current controller.rs

  transport/
    mod.rs
    stream_sync.rs      # current stream_sync.rs
    chunking.rs         # current chunking.rs
    chunk_assembler.rs  # current chunk_assembler.rs
    flow_control.rs     # current flow_control.rs
    proto/gossip.proto

  kv/
    mod.rs              # current kv.rs
    crdt/
      mod.rs
      crdt.rs
      operation.rs
      epoch_max_wins.rs
      kv_store.rs
      merge_strategy.rs
      replica.rs

  membership/
    partition.rs
    topology.rs
    node_state_machine.rs

  observability/
    metrics.rs
```

Suggested rename order:

1. Rename `ping_server.rs` to `inbound.rs` or `gossip_service.rs`.
2. Rename `controller.rs` to `outbound.rs` or `gossip_controller.rs`.
3. Move `chunking.rs`, `chunk_assembler.rs`, `flow_control.rs`, and `stream_sync.rs` under `transport/`.
4. Move `kv.rs` plus `crdt_kv/` under `kv/`.
5. Move `partition.rs`, `topology.rs`, and `node_state_machine.rs` under `membership/`.

The first two renames are the highest value because the current names obscure that both files participate in bidirectional SyncStream traffic.

## V2 Namespace Map

| Namespace | Mode | Producer | Consumer | Wire payload |
| --- | --- | --- | --- | --- |
| `worker:` | CRDT LWW | `WorkerSyncAdapter` | remote `WorkerSyncAdapter` subscriber | `CRDT_BATCH` entries |
| `rl:` | CRDT EpochMaxWins | `RateLimitSyncAdapter` | remote `RateLimitSyncAdapter` subscriber | `CRDT_BATCH` entries |
| `config:` | CRDT LWW | mesh admin/config callers | config readers | `CRDT_BATCH` entries |
| `td:` | Stream broadcast drain | `TreeSyncAdapter` drain callback | remote TreeSync subscriber | `STREAM_BATCH` |
| `tree:req:` | Stream targeted | repair requester | target peer TreeSync subscriber | `STREAM_BATCH` |
| `tree:page:` | Stream targeted | repair responder | requesting peer TreeSync subscriber | `STREAM_BATCH` |

## What Is Not In This Crate

- Gateway registry/policy integration lives in `model_gateway/src/mesh/adapters/`.
- Tree application logic lives in the gateway `CacheAwarePolicy` and `kv-index`.
- SWIM hardening beyond current ping/ping-req remains future membership work.
- Mixed v1/v2 cluster compatibility is not supported after this cleanup.
