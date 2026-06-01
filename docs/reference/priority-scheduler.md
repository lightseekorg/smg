---
title: Priority Scheduler Reference
---

# Priority Scheduler Reference

Precise contract for the priority-aware admission scheduler: the request header clients send, the response codes the gateway returns, and every configuration knob with its exact name and default. For how it works conceptually, see [Priority Scheduling](../concepts/reliability/priority-scheduling.md).

The scheduler is **disabled by default**. When off, the gateway uses its [legacy concurrency-limit admission path](../concepts/reliability/rate-limiting.md).

---

## Request header: `x-smg-priority`

Clients request a priority class with the `x-smg-priority` request header.

| Property | Behavior |
|----------|----------|
| **Header name** | `x-smg-priority` |
| **Values** | `system`, `interactive`, `default`, `bulk` |
| **Case** | Case-insensitive (`Bulk`, `INTERACTIVE`, `SyStEm` all parse). Surrounding whitespace is trimmed. |
| **Missing header** | Treated as `default`. |
| **Unknown value** | Any unrecognized value (including the empty string) silently degrades to `default` — admission never fails because of a typo in this header. Counted under `smg_scheduler_unknown_priority_value_total`. |

### Tenant clamp

The header chooses a class; the **tenant's configured maximum class caps it**. The effective class is:

```text
effective = min(requested_class, tenant_max_class)
```

- The clamp only ever moves a request **down**. The header can never promote a request above the tenant's ceiling.
- A tenant whose `max_class` is `default` that sends `x-smg-priority: system` is admitted as `default`.
- A clamp (effective class below requested) is counted under `smg_scheduler_clamp_total`.

A tenant's `max_class` comes from the per-tenant policy in the YAML config, or from the gateway-wide default (`--priority-scheduler-default-max-class`) for tenants not listed. See [Tenant policy](#tenant-policy).

---

## Response codes

The scheduler surfaces admission and preemption outcomes as HTTP status codes. Each rejection also carries the gateway's standard JSON error body and `X-SMG-Error-Code` header.

| Status | Condition | `X-SMG-Error-Code` | Extra headers |
|--------|-----------|--------------------|---------------|
| **503** Service Unavailable | **Preempted** — admitted, then cancelled before its first byte to make room for a higher-priority request | `scheduler_preempted` | `X-SMG-Preempted: true`, `Retry-After: 1` |
| **429** Too Many Requests | **Queue full** — the request's per-class queue is at its configured depth | `scheduler_queue_full` | — |
| **408** Request Timeout | **Queue timeout** — the request waited longer than its class's `queue_timeout` | `scheduler_queue_timeout` | — |
| **499** Client Closed Request | **Client gone** — the client disconnected before admission completed (nginx convention; never actually read) | `scheduler_client_cancelled` | — |

!!! tip "Telling a preemption apart from an overload"
    Both preemption and a genuinely overloaded backend can return `503`. The `X-SMG-Preempted: true` header is what distinguishes a preemption. A preempted request is safe to retry immediately, which is why it carries `Retry-After: 1`.

---

## Enabling the scheduler

The scheduler is controlled by CLI flags (also settable in the config file). Per-class tuning and per-tenant policy live in a separate optional YAML file.

```bash
smg \
  --worker-urls http://w1:8000 http://w2:8000 \
  --priority-scheduler-enabled \
  --priority-scheduler-default-max-class interactive \
  --priority-scheduler-config /etc/smg/priority.yaml
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--priority-scheduler-enabled` | `false` | Master switch. When unset, the legacy concurrency-limit middleware stays wired and no scheduler is constructed. |
| `--priority-scheduler-default-max-class` | `default` | Maximum class for tenants not listed in the YAML (`system` \| `interactive` \| `default` \| `bulk`). Parsed with the same rules as the header — an unknown value falls back to `default`. |
| `--priority-scheduler-config` | unset | Path to the optional priority-scheduler YAML (per-class overrides + per-tenant policy). Absent → built-in defaults and an empty tenant policy map. |
| `--priority-scheduler-tenant-metric-top-n` | `32` | Cap on per-tenant scheduler metric label cardinality. Tenants past the top N are bucketed under `tenant="other"`. |

!!! warning "Fail-safe startup"
    If the scheduler is enabled but cannot start — unparseable YAML, or class reservations that sum to more than the live backend capacity — the gateway logs at `ERROR` and **falls back to legacy admission** instead of aborting. It does not take the data plane down.

---

## YAML configuration

The file referenced by `--priority-scheduler-config` has two top-level maps, both optional. An empty or absent file means "use built-in defaults for every class, no per-tenant overrides."

```yaml
# Per-class tuning. Any class you omit keeps its built-in default.
classes:
  interactive:
    reserved: 128
    queue_size: 256
    queue_timeout_secs: 30
    starvation_threshold_secs: 5
    can_preempt: true
  bulk:
    reserved: 0
    queue_size: 1024
    queue_timeout_secs: 300
    starvation_threshold_secs: 120
    can_preempt: false

# Per-tenant priority ceiling. Tenants not listed use
# --priority-scheduler-default-max-class.
tenant_policies:
  "auth:acme":
    max_class: interactive
  "auth:internal-cron":
    max_class: system
```

Class keys and `max_class` values are lowercase: `system`, `interactive`, `default`, `bulk`. An unknown class name in the YAML is a parse error (which triggers the fail-safe fallback above), unlike the lenient request header.

### Per-class knobs

Each entry under `classes` accepts the following fields. All are per-class.

| Field | Type | Meaning |
|-------|------|---------|
| `reserved` | integer (slots) | Slots reserved for this class. A higher class's *unused* reservation is held back from lower classes; a class's own reservation never reduces its own headroom. The sum across all classes must fit under live capacity. |
| `queue_size` | integer | Per-class queue depth limit. A request that arrives when the queue is full is rejected with **429**. |
| `queue_timeout_secs` | integer (seconds) | How long a queued request waits before it is rejected with **408**. Must be `> 0`. |
| `starvation_threshold_secs` | integer (seconds) | Head-of-queue age past which the dispatcher promotes a waiter out of normal priority order (and lets it use a reserved-but-unused slot) to avoid starvation. Must be `> 0`. |
| `can_preempt` | boolean | Whether admissions in this class may preempt a lower-class in-flight request that has not yet emitted its first byte. |

### Built-in defaults

These apply to any class with no YAML override.

| Class | `reserved` | `queue_size` | `queue_timeout_secs` | `starvation_threshold_secs` | `can_preempt` |
|-------|-----------:|-------------:|---------------------:|----------------------------:|:-------------:|
| `system` | 32 | 64 | 30 | 5 | `true` |
| `interactive` | 128 | 256 | 30 | 5 | `true` |
| `default` | 0 | 512 | 60 | 30 | `false` |
| `bulk` | 0 | 1024 | 300 | 120 | `false` |

Higher classes fail fast (short queues, short timeouts) and reserve capacity; lower classes wait patiently (deep queues, long timeouts) and reserve nothing.

### Validation

At startup the scheduler validates:

- `queue_timeout_secs > 0` for every class (else startup fails for that class).
- `starvation_threshold_secs > 0` for every class.
- The sum of all `reserved` values must not exceed the live backend capacity. On a capacity *shrink* that would otherwise break this invariant, the scheduler scales reservations down proportionally rather than locking itself out.

Any validation failure triggers the [fail-safe fallback to legacy admission](#enabling-the-scheduler).

---

## Tenant policy

A tenant's priority ceiling is resolved per request:

1. If the tenant key appears in `tenant_policies`, its `max_class` is used.
2. Otherwise the gateway-wide `--priority-scheduler-default-max-class` applies.

The resolved `max_class` is the upper bound for the [tenant clamp](#tenant-clamp). Tenant keys are the same keys the gateway uses elsewhere for tenancy (for example `auth:acme`).

| Field | Type | Meaning |
|-------|------|---------|
| `max_class` | `system` \| `interactive` \| `default` \| `bulk` | Highest class this tenant may be admitted under. A request's effective class is `min(header_class, max_class)`. |

---

## Metrics

The scheduler exposes these Prometheus metrics (see the [Metrics Reference](metrics.md) for the full catalog):

| Metric | Type | Key labels | Use |
|--------|------|------------|-----|
| `smg_scheduler_admit_total` | Counter | `class`, `outcome` | Admission outcomes (`admitted`, `rejected_queue_full`, `rejected_queue_timeout`, `preempted`, `client_cancelled`). |
| `smg_scheduler_queue_wait_seconds` | Histogram | `class` | Time spent queued before admission, timeout, or cancel. |
| `smg_scheduler_preemption_total` | Counter | `victim_class`, `by_class` | Successful preemptions. Authoritative preemption count. |
| `smg_scheduler_clamp_total` | Counter | `tenant`, `requested_class`, `effective_class` | Requests clamped below the class they asked for. |
| `smg_scheduler_unknown_priority_value_total` | Counter | `tenant` | Requests with an unrecognized `x-smg-priority` value. |
| `smg_scheduler_starvation_promotion_total` | Counter | `class` | Waiters admitted via the starvation override. |
| `smg_scheduler_inflight` | Gauge | `class` | Current in-flight requests per class. |
| `smg_scheduler_queue_depth` | Gauge | `class` | Current queued waiters per class. |
| `smg_scheduler_queue_size_limit` | Gauge | `class` | Configured queue limit per class. |
| `smg_scheduler_utilization` | Gauge | — | Total in-flight divided by backend capacity. |
| `smg_scheduler_class_capacity_pressure` | Gauge | `class` | Normalized 0.0–1.0 pressure (worse of queue and slot pressure). |

---

## See also

<div class="grid" markdown>

<div class="card" markdown>

### :material-priority-high: Priority Scheduling Concept

How slots, reservations, preemption, and starvation promotion fit together.

[Priority Scheduling →](../concepts/reliability/priority-scheduling.md)

</div>

<div class="card" markdown>

### :material-cog: Configuration Reference

All gateway CLI flags and configuration options.

[Configuration →](configuration.md)

</div>

<div class="card" markdown>

### :material-chart-box: Metrics Reference

Full catalog of Prometheus metrics.

[Metrics →](metrics.md)

</div>

</div>
