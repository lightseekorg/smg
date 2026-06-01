---
title: Priority Scheduling
---

# Priority Scheduling

When the gateway is at capacity, a single global queue makes every request compete on first-come-first-served terms. A latency-sensitive chat completion ends up waiting behind whatever batch job happened to arrive first. The **priority scheduler** replaces that flat queue with a priority-aware admission layer: clients declare a class with the `x-smg-priority` header, the gateway admits higher classes first, reserves capacity for them, and — when it has to — preempts a lower-class request that has not yet started streaming.

The priority scheduler is **opt-in**. When it is disabled (the default), the gateway keeps its [legacy concurrency-limit admission path](rate-limiting.md), so existing deployments see zero behavior change.

---

## A busy restaurant

The mechanics are easier to hold in your head with an analogy:

- **Tables are slots.** The dining room seats a fixed number of guests — the gateway's live backend capacity. A guest occupies a table for the length of their meal, the way a request holds a slot until its response finishes streaming.
- **Guest tiers are priority classes.** Walk-ins, regulars, and VIPs get different treatment. SMG has four tiers: `system`, `interactive`, `default`, and `bulk`.
- **The host is the scheduler.** The host decides who gets seated now, who waits by the door, and in what order the waiting line is called.
- **Reserved tables are reserved slots.** A few tables are kept open for VIPs even when the room looks full, so a VIP rarely waits.
- **Bumping a guest whose food has not arrived is preemption.** If a VIP walks in and every table is taken, the host may clear a table occupied by a walk-in *who has not been served yet* — nobody's dinner gets thrown away. A guest already eating is never bumped.
- **Calling a long-waiting walk-in out of turn is starvation promotion.** If someone has waited by the door far too long, the host seats them next even though higher tiers are still in line, so they are not stuck forever.

The rest of this page is the real mechanics behind that picture.

---

## Priority classes

Every admitted request is assigned one of four classes. They are strictly ordered — higher classes win contention, get reservations, and may preempt lower ones.

| Class | Rank | Intended traffic | Preempts lower classes? |
|-------|------|------------------|-------------------------|
| `system` | highest | Internal control-plane callers | Yes |
| `interactive` | high | Latency-sensitive (chat completions, autocomplete) | Yes |
| `default` | middle | Unlabeled traffic — what a request gets with no header | No |
| `bulk` | lowest | Background / batch jobs | No |

A request's class comes from the `x-smg-priority` header, then is **clamped down** to the maximum class the tenant is allowed to use. The header can only ever *lower* a request's class relative to the tenant cap — it can never promote a tenant above its ceiling. A free-tier tenant cannot escalate itself to `system` by setting a header. See the [reference page](../../reference/priority-scheduler.md) for the exact header contract.

!!! note "`system` is for the control plane"
    `system` is reserved for internal control-plane traffic. In practice the tenant clamp keeps external tenants out of it: their policy caps them below `system`, so no external request lands there regardless of the header they send.

---

## Slots and capacity

The scheduler admits against a pool of **slots**. Total slot capacity is the gateway's live backend capacity — it is read from the worker fleet, not a static number, and the scheduler reacts when it changes (workers scaling up or down). A request holds exactly one slot from the moment it is admitted until its response body finishes draining (or it is preempted or the client disconnects), at which point the slot returns to the pool.

When a slot frees, the scheduler's dispatcher wakes and tries to admit waiting requests, highest class first.

### Reserved slots

Each class can reserve a share of capacity. A reservation is a floor, not a fixed partition:

- A higher class's **unused** reservation is held back from lower classes. If `interactive` has reserved slots it is not currently using, `bulk` and `default` cannot consume them — they stay available so an arriving `interactive` request is admitted immediately instead of queueing.
- Once a higher class **actually uses** a reserved slot, the hold collapses one-for-one; the slot is simply in flight.
- A class's own reservation never counts against its own headroom.

This is why `system` and `interactive` ship with reservations by default while `default` and `bulk` reserve nothing — interactive traffic should almost never wait behind a batch burst. The sum of all reservations must fit under capacity; if it does not at startup, the scheduler refuses to build and the gateway falls back to legacy admission.

---

## Per-class queues

When no slot is immediately available, a request does not fail right away — it joins a **per-class FIFO queue**. Each class has its own queue with its own depth limit and its own wait timeout:

- If the queue is already at its configured depth, the request is rejected immediately (**429**).
- If the request waits longer than the class's timeout, it is rejected (**408**).

A client that disconnects *while queued* is not currently detected — its place is held until that timeout fires, because the cancel signal isn't yet wired to client disconnect at this stage. (The **499** code exists for this case but isn't emitted today.)

Higher classes have shorter queues and shorter timeouts (fail fast, the latency matters); lower classes have deeper queues and longer timeouts (wait patiently, throughput matters). The dispatcher drains queues in **strict priority order** — `system` first, then `interactive`, `default`, and `bulk`, fully draining a higher tier before serving a lower one. A sustained higher-priority flood *will* hold off a lower tier; the [starvation guard](#starvation-promotion) below is what keeps that from lasting forever.

---

## Preemption

Reservations alone are not enough. If `bulk` fills its share *and* spills into unreserved capacity at the same moment an `interactive` request arrives, reservations leave a gap. Preemption closes it.

When a preempt-capable request (`system` or `interactive` by default) finds no free slot and no reservation to draw on, the scheduler looks for a **victim**: a strictly-lower-class request that is still in flight **but has not yet emitted its first response byte**. If it finds one, it cancels that request and claims the freed slot.

The TTFT boundary is the rule that makes this safe:

- **Only pre-first-byte requests are eligible.** A request that has already streamed a byte to its client is *never* preempted — the user is already seeing output, and truncating it would be worse than making the higher-priority request wait. The handoff is decided by a single atomic compare-and-swap on the victim: whichever happens first wins, "first byte emitted" or "selected for preemption," and the loser backs off.
- **Victim selection minimizes wasted work.** Among eligible victims the scheduler prefers the **lowest class** (cheapest to cancel) and, within that class, the **most-recently-admitted** request (the least upstream work thrown away).
- **At most one victim per admission.** No cascading preemptions.

A preempted request receives **503** with `Retry-After: 1` and an `X-SMG-Preempted: true` header, so clients and proxies can tell a preemption apart from an ordinary overload and retry promptly. If the victim's slot does not free within a short budget, the preemptor simply falls through to the queue — the cancel has already fired, so the slot frees shortly regardless.

!!! warning "Preemption requires cancel-aware handlers"
    A victim only actually unwinds if its request handler honors the scheduler's cancel signal. Until every long-running handler is wired to that signal, `can_preempt` is expected to stay off in practice even though the machinery is in place — a marked-but-not-unwound victim has its first data frame truncated rather than its upstream work stopped.

---

## TTFT protection

The flip side of preemption is a guarantee for the request being served: **once you have produced a first byte, you are safe.** The scheduler tracks time-to-first-byte per in-flight request. The response-body wrapper marks the first *data* frame, and from that instant the request is no longer a preemption candidate. This bounds the blast radius of a priority surge — a higher-priority spike can reclaim slots from work that has not visibly started, but it can never tear down responses already streaming to users.

---

## Starvation promotion

Strict priority ordering risks starving the lowest classes: if higher classes stay busy, a `bulk` request could wait at the head of its queue indefinitely. To prevent that, each class has a **starvation threshold**. When the request at the head of a queue has waited longer than its class's threshold, the dispatcher promotes it out of normal priority order and admits it next — even letting it consume a slot that a higher class had reserved but is not using.

The dispatcher checks for starved waiters **lowest priority first** (the classes most at risk), before it does its normal high-to-low drain. The tradeoff is deliberate: avoiding indefinite starvation is worth occasionally lending out a reserved-but-idle slot.

---

## Fallback to legacy admission

The priority scheduler is designed to **fail safe**. It only takes over when it is both enabled and able to start cleanly:

- If the scheduler is **disabled** (the default), the gateway wires the legacy concurrency-limit middleware — no scheduler is even constructed.
- If the scheduler is **enabled but misconfigured** — unparseable YAML, or reservations that sum to more than the live capacity — the gateway logs the failure at `ERROR` and **falls back to legacy admission** rather than aborting startup. A broken scheduler config must never take the data plane down.

In both cases the request path is the familiar token-bucket concurrency limiter described in [Rate Limiting](rate-limiting.md). The admission path is chosen once at startup.

---

## Observability

The scheduler emits Prometheus metrics for admission outcomes, queue waits, preemptions, priority clamps, and per-class capacity pressure. These are the signals you watch to size reservations and queue depths, and to detect clients mis-setting the priority header. The full list is in the [Metrics Reference](../../reference/metrics.md).

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-file-document-outline: Priority Scheduler Reference

The exact header values, response codes, and configuration knobs.

[Priority Scheduler Reference →](../../reference/priority-scheduler.md)

</div>

<div class="card" markdown>

### :material-tray-full: Rate Limiting

The legacy concurrency-limit path the scheduler falls back to.

[Rate Limiting →](rate-limiting.md)

</div>

<div class="card" markdown>

### :material-electric-switch: Circuit Breakers

Isolate failing workers to prevent cascade failures.

[Circuit Breakers →](circuit-breakers.md)

</div>

<div class="card" markdown>

### :material-chart-box: Metrics Reference

Scheduler admission, queue, and preemption metrics.

[Metrics Reference →](../../reference/metrics.md)

</div>

</div>
