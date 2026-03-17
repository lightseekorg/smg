---
title: Home
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Shepherd Model Gateway

**High-performance inference gateway for LLM deployments**

One gateway for routing, load balancing, and orchestrating traffic across your LLM fleet.

[Get Started](getting-started/index.md){ .button .button--primary }
[GitHub](https://github.com/lightseekorg/smg){ .button .button--secondary }

</div>

<div class="stats-bar">
<div class="stat">
<span class="stat-value">70%</span>
<span class="stat-label">TTFT Reduction</span>
</div>
<div class="stat">
<span class="stat-value"><1ms</span>
<span class="stat-label">Routing Latency</span>
</div>
<div class="stat">
<span class="stat-value">40+</span>
<span class="stat-label">Prometheus Metrics</span>
</div>
<div class="stat">
<span class="stat-value">100%</span>
<span class="stat-label">OpenAI Compatible</span>
</div>
</div>

<div class="backends">
<div class="backends-label">Works with</div>
<div class="backends-grid">
<span class="backend">vLLM</span>
<span class="backend">SGLang</span>
<span class="backend">TensorRT-LLM</span>
<span class="backend">OpenAI</span>
<span class="backend">Claude</span>
<span class="backend">Gemini</span>
</div>
</div>

---

## What SMG does

SMG sits between your applications and LLM workers. It manages routing, failover, tokenization, and observability so you can scale inference without building that infrastructure yourself.

<div class="grid" markdown>

<div class="card" markdown>

### :material-server-network: Full OpenAI server mode

In gRPC mode, SMG handles tokenization, chat templates, tool calling, MCP, reasoning loops, and detokenization at the gateway. Workers just run inference.

</div>

<div class="card" markdown>

### :material-speedometer: Sub-millisecond routing

Written in Rust. Gateway-side tokenizer caching, token-level streaming, cache-aware routing. Designed for throughput.

</div>

<div class="card" markdown>

### :material-shield-check: Production reliability

Circuit breakers, retries with exponential backoff, rate limiting, health monitoring. Keeps your inference stack up.

</div>

<div class="card" markdown>

### :material-chart-line: Built-in observability

40+ Prometheus metrics, OpenTelemetry tracing, structured logging. See what's happening without extra tooling.

</div>

</div>

---

## Three operating modes

<div class="architecture-diagram">
  <img src="assets/images/architecture-animated.svg" alt="SMG Architecture">
</div>

<div class="grid" markdown>

<div class="card" markdown>

### :material-lightning-bolt: gRPC mode

SMG handles everything — tokenization, chat templates, tool parsing, MCP loops, detokenization, PD routing. Workers run raw inference.

</div>

<div class="card" markdown>

### :material-swap-horizontal: HTTP mode

SMG handles routing, load balancing, and failover. Workers run full OpenAI-compatible servers. Supports prefill-decode disaggregation.

</div>

<div class="card" markdown>

### :material-cloud-outline: External mode

Route to OpenAI, Claude, Gemini through a single endpoint. Mix self-hosted and cloud models behind one API.

</div>

</div>

---

<div class="grid" markdown>

<div class="card" markdown>

### Getting started

Install, connect workers, send your first request.

[Quickstart →](getting-started/index.md)

</div>

<div class="card" markdown>

### Architecture & concepts

Routing strategies, reliability features, extensibility.

[Concepts →](concepts/index.md)

</div>

<div class="card" markdown>

### API reference

OpenAI-compatible API, admin endpoints, gateway extensions.

[Reference →](reference/index.md)

</div>

<div class="card" markdown>

### Contributing

Development setup, code style, how to contribute.

[Contribute →](contributing/index.md)

</div>

</div>

<div class="community-links" markdown>
:fontawesome-brands-github: [GitHub](https://github.com/lightseekorg/smg) · :fontawesome-brands-slack: [Slack](https://join.slack.com/t/lightseekorg/shared_invite/zt-3py6mpreo-XUGd064dSsWeQizh3YKQrQ) · :fontawesome-brands-discord: [Discord](https://discord.gg/wkQ73CVTvR)
</div>
