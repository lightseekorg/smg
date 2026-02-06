---
title: Tasks
---

# Tasks

Tasks are step-by-step guides for accomplishing specific goals with SMG. Each task is self-contained and can be completed independently.

---

## Deployment

Get SMG running in various environments.

| Task | Description |
|------|-------------|
| [Run with Docker](deployment/docker.md) | Deploy SMG using Docker containers |
| [Deploy to Kubernetes](deployment/kubernetes.md) | Production deployment with service discovery |
| [Configure TLS](deployment/tls.md) | Secure communications with TLS/mTLS |

---

## Operations

Day-to-day operational tasks.

| Task | Description |
|------|-------------|
| [Monitor with Prometheus](operations/monitoring.md) | Set up metrics and dashboards |
| [Configure Logging](operations/logging.md) | Structured logging and log aggregation |
| [Manage Workers](operations/workers.md) | Add, remove, and update workers |

---

## Before You Begin

Most tasks assume you have:

- [ ] SMG [installed](../getting-started/index.md)
- [ ] Basic familiarity with [SMG concepts](../concepts/index.md)
- [ ] Access to one or more inference workers

---

## Task Format

Each task follows a consistent format:

1. **Prerequisites** — What you need before starting
2. **Steps** — Numbered steps to complete the task
3. **Verification** — How to confirm success
4. **Troubleshooting** — Common issues and solutions
