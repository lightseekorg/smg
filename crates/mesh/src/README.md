# Mesh Crate Maintainer Notes

This crate is in the middle of the mesh v2 cutover. The production gateway now talks to the generic `MeshKV` namespaces through gateway-side adapters, while several v1 state-sync modules still exist inside this crate for protocol compatibility, tests, and follow-up cleanup.

Cluster-wide mixed v1/v2 deployments are not supported by the d-3 cutover plan. Every live gateway in the cluster is expected to run the same v2 data path.

## Current Data Path

The gateway-facing v2 path is:

1. `service.rs` builds one node-wide `MeshKV`.
2. `model_gateway/src/mesh/mod.rs` configures namespaces:
   - `worker:` CRDT, `LastWriterWins`
   - `rl:` CRDT, `EpochMaxWins`
   - `config:` CRDT, auto-registered as `LastWriterWins`
   - `td:` stream, broadcast
   - `tree:req:` stream, targeted
   - `tree:page:` stream, targeted
3. Gateway adapters publish and consume those namespaces:
   - `WorkerSyncAdapter`
   - `RateLimitSyncAdapter`
   - `TreeSyncAdapter`
4. Gossip transports CRDT entries and stream batches between peers.

The older `MeshSyncManager`/`StateStores` path is no longer used by the gateway registry or cache-aware policy in production, but it still remains in this crate and in some in-crate tests.

## File Inventory

### Public Entry Points

| File | Current role | State |
| --- | --- | --- |
| `lib.rs` | Re-exports public mesh types. Still exports both v2 (`MeshKV`, namespaces, merge helpers) and v1 (`MeshSyncManager`, tree ops, legacy subscribers). | Transitional. Remove v1 exports after v1 modules are deleted. |
| `service.rs` | Builds the mesh server, handler, stores, controller, ping service, partition detector, state machine, and `MeshKV`. Exposes `MeshServerHandler::mesh_kv()`. | Active, but still carries v1 handler APIs and `MeshSyncManager`. |
| `kv.rs` | Generic v2 API: `MeshKV`, `CrdtNamespace`, `StreamNamespace`, drain/subscription registries, config namespace. | Active v2 core. |

### CRDT Core

| File | Current role | State |
| --- | --- | --- |
| `crdt_kv/crdt.rs` | OR-map store, tombstones, prefix merge strategy lookup, live operation application, compaction. | Active v2 core. |
| `crdt_kv/operation.rs` | Operation log and strategy-aware latest-operation selection/compaction. | Active v2 core. |
| `crdt_kv/merge_strategy.rs` | `LastWriterWins`, reserved `MaxValueWins`, and `EpochMaxWins`. | Active v2 core. Consider removing or implementing `MaxValueWins`. |
| `crdt_kv/epoch_max_wins.rs` | Fixed 16-byte epoch/count codec and merge helper for rate-limit shards. | Active v2 core. |
| `crdt_kv/kv_store.rs` | DashMap-backed storage plus generation counter. | Active shared infrastructure. |
| `crdt_kv/replica.rs` | Replica id and Lamport clock helpers. | Active shared infrastructure. |
| `crdt_kv/tests.rs` | In-crate CRDT tests. | Active tests. |
| `crdt_kv/mod.rs` | Internal module exports. | Active. |

### Gossip Transport

| File | Current role | State |
| --- | --- | --- |
| `controller.rs` | Outbound gossip loop plus inbound response handling for peers this node dials. Despite the name, it acts as a client, sender, receiver, and round coordinator. | Active transport, poorly named. |
| `ping_server.rs` | Inbound tonic service for ping, sync stream, snapshot, and stream-batch requests. Also sends responses and applies inbound state. Despite the name, it is not only a ping server. | Active transport, poorly named. |
| `collector.rs` | Collects per-round v1 store deltas and v2 stream drains, then filters by peer watermarks. | Mixed. Stream drain portion is active v2; store-delta collection is legacy. |
| `chunking.rs` | Sender-side stream chunking and receiver-side dispatch into stream subscribers. | Active v2 stream infrastructure. |
| `chunk_assembler.rs` | Receiver-side reassembly for chunked stream entries. Tree repair pages avoid this path by staying under the page cap. | Active generic infrastructure, lightly used today. |
| `flow_control.rs` | Message size limits, backpressure, and reconnect helpers. | Active transport support. |
| `proto/gossip.proto` | Gossip wire protocol. It still contains v1 store types and v2 stream batch messages. | Mixed wire surface. |

### Legacy State-Sync Layer

| File | Current role | State |
| --- | --- | --- |
| `sync.rs` | `MeshSyncManager`, worker/policy/app/rate-limit helpers, tree-state subscribers, checkpointing, and v1 tree delta code. | Legacy. Gateway no longer calls this in production after d-3, but tests and server internals still compile it. |
| `stores.rs` | Typed v1 stores for membership, app config, workers, policies, per-actor rate limits, and legacy tree state buffers. | Mixed. Membership still supports transport; app/worker/policy/tree/rate-limit store APIs are mostly legacy after v2 cutover. |
| `tree_ops.rs` | Legacy `TreeState`, `TreeOperation`, `TenantDelta`, compression helpers, and hash helpers. | Legacy except hash helpers are still exported. Move hash helpers or delete the rest with `MeshSyncManager`. |
| `rate_limit_window.rs` | Old periodic reset task for v1 rate-limit counters. | Legacy. v2 rate limits derive epoch from wall-clock during `RateLimitSyncAdapter::check_counter`. |
| `consistent_hash.rs` | Old owner-selection ring for v1 rate-limit shards. | Legacy for rate limiting. May still be useful only if a future feature needs ownership assignment. |

### Membership And Operations Support

| File | Current role | State |
| --- | --- | --- |
| `partition.rs` | Partition detection over node membership state. | Active/orthogonal to data-sync v2. |
| `node_state_machine.rs` | Startup readiness state machine. | Active/orthogonal. |
| `topology.rs` | Full/sparse topology planner. Currently test-only in non-test builds. | Dormant. Keep until membership hardening is revisited. |
| `metrics.rs` | Mesh metric descriptions. | Active but broad. Needs pruning after v1 deletion. |
| `mtls.rs` | Optional mTLS support for mesh gRPC. | Active. |

### Tests, Benches, Build

| File | Current role | State |
| --- | --- | --- |
| `tests/` | In-crate integration-style tests for chunking, comprehensive legacy sync behavior, and test utilities. | Mixed. Some tests should disappear with v1 modules. |
| `benches/mesh_serialization.rs` | Serialization benchmark coverage for mesh payloads. | Mixed; references legacy payloads. |
| `build.rs` | Prost/tonic code generation for `proto/gossip.proto`. | Active. |

## Why Some Files Were Not Touched By v2

The v2 slices intentionally targeted data-sync first, not every mesh subsystem.

- Touched by v2: generic KV, CRDT merge strategy, stream chunking/dispatch hooks, and gateway adapters.
- Mostly untouched by v2: membership hardening, topology, partition detection, mTLS, and older sync-manager internals.
- Still present by design for now: v1 store/sync code that is entangled with tests, generated proto shape, and `MeshServerHandler` APIs.

Git history shows Chang Su has touched all current `crates/mesh/src` files at least once. The more useful distinction is not authorship; it is whether a file is on the v2 production data path after d-3.

## Naming Problems

`ping_server.rs` and `controller.rs` are confusing names:

- `ping_server.rs` is an inbound gRPC service, but it handles much more than ping: snapshots, sync streams, stream batches, and applying received state.
- `controller.rs` is an outbound peer loop, but it also receives responses and applies inbound updates from those responses.

Both files contain sender and receiver behavior. The real split is inbound service vs outbound peer sessions, not server vs client in the abstract.

## Proposed Structure

Recommended target structure after the d-3 follow-up cleanup:

```text
crates/mesh/src/
  lib.rs
  service.rs                 # builder + handler only
  metrics.rs
  mtls.rs

  transport/
    mod.rs
    inbound_service.rs       # rename ping_server.rs
    peer_session.rs          # rename controller.rs
    collector.rs             # split stream drain vs legacy store collector first
    flow_control.rs
    chunking.rs
    chunk_assembler.rs
    proto.rs                 # generated gossip exports / proto adapters

  kv/
    mod.rs                   # current kv.rs namespace API
    crdt/
      mod.rs
      or_map.rs              # current crdt_kv/crdt.rs
      operation_log.rs       # current crdt_kv/operation.rs
      merge_strategy.rs
      epoch_max_wins.rs
      store.rs               # current crdt_kv/kv_store.rs
      replica.rs

  membership/
    mod.rs
    partition.rs
    readiness.rs             # current node_state_machine.rs
    topology.rs

  legacy/
    mod.rs
    sync_manager.rs          # current sync.rs, temporary
    stores.rs                # current stores.rs, temporary
    tree_ops.rs              # current tree_ops.rs, temporary
    rate_limit_window.rs     # current rate_limit_window.rs, temporary
    consistent_hash.rs       # current consistent_hash.rs, if no owner remains
```

Short-term rename-only target, if we want a smaller PR:

```text
ping_server.rs -> inbound_service.rs
controller.rs  -> peer_session.rs
```

Do the rename after the v1 deletion pass if possible. Otherwise the diff will be noisy because those files still carry both v1 state-store handling and v2 stream handling.

## D-3 Cleanup Checklist

Already done on the d-3 branch:

- `rl:` namespace configured with `EpochMaxWins`.
- CRDT live merge and operation-log compaction honor prefix-specific merge strategy.
- Gateway starts v2 worker/tree/rate-limit adapters from `server.rs`.
- Cache-aware local inserts publish through the TreeSync adapter.
- Worker registry local changes publish through the WorkerSync adapter.
- Rate-limit middleware checks the v2 adapter and reads config from `config:`.
- Cold-start TreeSync repair fan-out runs from configured model ids.
- Tree repair page cap is read from runtime mesh config and validated by `TreeSyncConfig`.
- Gateway registry/policy/cache-aware v1 `mesh_sync` hooks are removed.

Still to do in a follow-up cleanup PR:

- Remove `TreeStateSubscriber` and `WorkerStateSubscriber` exports.
- Delete or quarantine `MeshSyncManager` after remaining server/tests callers are gone.
- Delete legacy `tree_ops.rs` data types, or move only `hash_node_path` / `hash_token_path` to a small hash helper module first.
- Remove `tree_ops_pending`, `tree_configs`, and `tree_versions` from `StateStores`.
- Remove legacy rate-limit store/window/consistent-hash paths.
- Split `collector.rs` so v2 stream draining is not tangled with v1 store update collection.
- Rename `ping_server.rs` and `controller.rs` once the v1/v2 split is clearer.
