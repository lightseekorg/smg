# Reliability & Resilience

SMG provides several mechanisms to ensure high availability and stability.

## Core Mechanisms

*   **[Circuit Breakers](./circuit-breakers.md)**: Automatically detect failing workers and temporarily stop sending traffic to them until they recover.
*   **[Retries & Backoff](./retries.md)**: Configurable retry logic with exponential backoff to handle transient network issues or worker busy states.
*   **[Rate Limiting](./rate-limiting.md)**: Protect your workers from being overwhelmed by controlling the concurrency and request rate.
*   **[Health Checks](./health-checks.md)**: Active and passive monitoring of worker health to remove unhealthy nodes from the rotation.
*   **[Graceful Shutdown](./graceful-shutdown.md)**: Ensure in-flight requests complete before the server stops.
