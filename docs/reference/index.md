---
title: Reference
---

# Reference

Reference documentation provides detailed specifications for SMG's APIs, CLI, configuration options, and metrics. Use these pages when you need precise information about specific features.

---

## API Reference

<div class="grid cards" markdown>

-   :material-api:{ .lg .middle } **OpenAI-Compatible API**

    ---

    Complete reference for the OpenAI-compatible endpoints including chat completions, completions, and models.

    [:octicons-arrow-right-24: API Reference](api/openai.md)

-   :material-puzzle:{ .lg .middle } **Extension API**

    ---

    SMG-specific extensions for worker management, health checks, and gateway configuration.

    [:octicons-arrow-right-24: Extensions](api/extensions.md)

</div>

---

## Configuration Reference

<div class="grid cards" markdown>

-   :material-console:{ .lg .middle } **CLI Reference**

    ---

    Complete command-line interface reference with all options and environment variables.

    [:octicons-arrow-right-24: CLI Reference](cli.md)

-   :material-cog:{ .lg .middle } **Configuration**

    ---

    Configuration options for tuning SMG behavior including routing, rate limiting, and reliability.

    [:octicons-arrow-right-24: Configuration](configuration.md)

</div>

---

## Observability Reference

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **Metrics Reference**

    ---

    Complete list of Prometheus metrics exposed by SMG for monitoring and alerting.

    [:octicons-arrow-right-24: Metrics](metrics.md)

</div>

---

## Quick Links

| Reference | Description |
|-----------|-------------|
| [CLI Options](cli.md#options) | All command-line flags |
| [Environment Variables](cli.md#environment-variables) | Configurable environment variables |
| [Chat Completions API](api/openai.md#chat-completions) | `/v1/chat/completions` endpoint |
| [HTTP Metrics](metrics.md#layer-1-http-metrics) | HTTP request metrics |
| [Worker Metrics](metrics.md#layer-3-worker-metrics) | Worker health and performance metrics |
