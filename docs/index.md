---
title: Home
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Shepherd Model Gateway

**The high-performance inference gateway for production LLM deployments**

Route, balance, and orchestrate traffic across your LLM fleet with enterprise-grade reliability.

[Get Started](getting-started/index.md){ .button .button--primary }
[View on GitHub](https://github.com/lightseekorg/smg){ .button .button--secondary }

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
<span class="stat-label">Metrics</span>
</div>
<div class="stat">
<span class="stat-value">100%</span>
<span class="stat-label">OpenAI Compatible</span>
</div>
</div>

<div class="backends">
<div class="backends-label">Works With</div>
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

## Why Shepherd Model Gateway?

SMG sits between your applications and LLM workers, providing a unified control and data plane for managing inference at scale. Whether you're running a single model or orchestrating hundreds of workers across multiple clusters, SMG gives you the tools to do it reliably.

<div class="grid" markdown>

<div class="card" markdown>

### :material-server-network: Full OpenAI Server Mode

With gRPC workers, SMG becomes a complete OpenAI-compatible server — handling tokenization, chat templates, tool calling, MCP, reasoning loops, and detokenization at the gateway level.

</div>

<div class="card" markdown>

### :material-speedometer: High Performance

Native Rust implementation with gateway-side tokenization caching, token-level streaming, and sub-millisecond routing. Built for throughput at scale.

</div>

<div class="card" markdown>

### :material-shield-check: Enterprise Reliability

Circuit breakers, automatic retries with exponential backoff, rate limiting, and health monitoring. Your inference stack stays up.

</div>

<div class="card" markdown>

### :material-chart-line: Full Observability

40+ Prometheus metrics, OpenTelemetry distributed tracing, and structured logging. Know exactly what's happening.

</div>

</div>

---

## How It Works

<div class="architecture-diagram">
  <img src="assets/images/architecture-animated.svg" alt="SMG Architecture">
</div>

<div class="grid" markdown>

<div class="card" markdown>

### :material-lightning-bolt: gRPC Mode

**Gateway = Full Server**

SMG handles everything: tokenization, chat templates, tool parsing, MCP loops, detokenization, and PD routing. Workers run raw inference on vLLM, SGLang, or TensorRT-LLM.

</div>

<div class="card" markdown>

### :material-swap-horizontal: HTTP Mode

**Gateway = Smart Proxy**

SMG handles routing, load balancing, and failover. Workers run full OpenAI-compatible servers (vLLM, SGLang, TRT-LLM). Supports PD disaggregation.

</div>

<div class="card" markdown>

### :material-cloud-outline: External Mode

**Gateway = Unified Router**

Route to OpenAI, Claude, Gemini through one endpoint. Mix self-hosted and cloud models seamlessly.

</div>

</div>

---

## Choose Your Path

<div class="grid" markdown>

<div class="card" markdown>

### :material-rocket-launch: New to SMG?

Start here to understand what SMG does and get it running in minutes.

[Getting Started →](getting-started/index.md)

</div>

<div class="card" markdown>

### :material-book-open-variant: Learn the Concepts

Understand SMG's architecture, routing strategies, and reliability features.

[Read Concepts →](concepts/index.md)

</div>

<div class="card" markdown>

### :material-wrench: Ops Setup

Continue onboarding with monitoring, logging, and TLS guides.

[View Getting Started Guides →](getting-started/index.md)

</div>

<div class="card" markdown>

### :material-api: API Reference

Complete reference for the OpenAI-compatible API and SMG extensions.

[View Reference →](reference/index.md)

</div>

</div>

<div class="community-links" markdown>
:fontawesome-brands-github: [GitHub](https://github.com/lightseekorg/smg) · :fontawesome-brands-slack: [Slack](https://join.slack.com/t/lightseekorg/shared_invite/zt-3py6mpreo-XUGd064dSsWeQizh3YKQrQ) · :fontawesome-brands-discord: [Discord](https://discord.gg/wkQ73CVTvR)
</div>
