---
title: Configuration Guides
---

# Configuration Guides

In-depth guides for configuring specific SMG features and subsystems.

---

## Available Guides

<div class="grid cards" markdown>

-   :material-format-text:{ .lg .middle } **Tokenization**

    ---

    Configure tokenizers, chat templates, and caching for optimal performance with your models.

    [:octicons-arrow-right-24: Tokenization Guide](tokenization.md)

-   :material-lock:{ .lg .middle } **Authentication**

    ---

    Configure API key authentication, JWT validation, and access control.

    [:octicons-arrow-right-24: Authentication Guide](authentication.md)

-   :material-server-network:{ .lg .middle } **gRPC Pipeline**

    ---

    Configure gRPC communication, request processing, and streaming.

    [:octicons-arrow-right-24: gRPC Pipeline Guide](grpc-pipeline.md)

-   :material-robot:{ .lg .middle } **MCP (Model Context Protocol)**

    ---

    Configure MCP integration for tool calling and external services.

    [:octicons-arrow-right-24: MCP Guide](mcp.md)

-   :material-database:{ .lg .middle } **Storage**

    ---

    Configure conversation history storage backends (Memory, Redis, PostgreSQL, Oracle).

    [:octicons-arrow-right-24: Storage Guide](storage.md)

-   :material-shield-check:{ .lg .middle } **Resilience**

    ---

    Configure retries, circuit breakers, health checks, timeouts, and graceful shutdown.

    [:octicons-arrow-right-24: Resilience Guide](resilience.md)

-   :material-routes:{ .lg .middle } **Routing Policies**

    ---

    Configure request routing policies, cache-aware scheduling, PD disaggregation, and session affinity.

    [:octicons-arrow-right-24: Routing Guide](routing.md)

</div>

---

## Quick Reference

For a complete list of all configuration options, see:

- [CLI Reference](../reference/cli.md) - All command-line flags
- [Configuration Reference](../reference/configuration.md) - Configuration options by category

---

## Configuration Methods

SMG can be configured through:

1. **Command-line arguments** (highest priority)
2. **Environment variables** (with `SMG_` prefix)
3. **Default values** (lowest priority)

See the [Configuration Reference](../reference/configuration.md) for details on each method.
