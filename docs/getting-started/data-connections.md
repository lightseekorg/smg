---
title: Data Connections
---

# Data Connections

This guide helps you quickly enable conversation history storage with a backend that matches your environment.

<div class="prerequisites" markdown>

#### Before you begin

- Completed the [Getting Started](index.md) guide
- SMG running with at least one worker

</div>

---

## Choose a Backend

SMG supports these history backends via `--history-backend`:

- `memory` (default): in-process, non-persistent
- `none`: disable history storage
- `postgres`: durable relational storage
- `redis`: fast key-value storage with optional retention
- `oracle`: enterprise Oracle backend

---

## Quick Start Commands

### Memory (default)

```bash
smg \
  --worker-urls http://worker:8000 \
  --history-backend memory
```

### No history

```bash
smg \
  --worker-urls http://worker:8000 \
  --history-backend none
```

### PostgreSQL

```bash
smg \
  --worker-urls http://worker:8000 \
  --history-backend postgres \
  --postgres-db-url "postgres://user:password@localhost:5432/smg" \
  --postgres-pool-max-size 16
```

### Redis

```bash
smg \
  --worker-urls http://worker:8000 \
  --history-backend redis \
  --redis-url "redis://localhost:6379" \
  --redis-pool-max-size 16 \
  --redis-retention-days 30
```

Set `--redis-retention-days -1` for persistent retention.

### Oracle

```bash
smg \
  --worker-urls http://worker:8000 \
  --history-backend oracle \
  --oracle-wallet-path /path/to/wallet \
  --oracle-tns-alias mydb_high \
  --oracle-user admin \
  --oracle-password "$ORACLE_PASSWORD"
```

---

## Required Flags by Backend

| Backend | Required flags |
|---------|----------------|
| `memory` | none |
| `none` | none |
| `postgres` | `--postgres-db-url` |
| `redis` | `--redis-url` |
| `oracle` | `--oracle-user`, `--oracle-password`, and one of (`--oracle-dsn`) or (`--oracle-wallet-path` + `--oracle-tns-alias`) |

---

## Environment Variables

You can provide Oracle credentials via environment variables:

- `ATP_WALLET_PATH`
- `ATP_TNS_ALIAS`
- `ATP_DSN`
- `ATP_USER`
- `ATP_PASSWORD`
- `ATP_POOL_MIN`
- `ATP_POOL_MAX`
- `ATP_POOL_TIMEOUT_SECS`

---

## Verify

```bash
curl http://localhost:30000/health
```

If startup fails, SMG returns a config validation error (for example missing DB URL or Oracle credentials).

---

## Next Steps

- [Chat History Concepts](../concepts/data/chat-history.md) — backend architecture and tradeoffs
- [Configuration Reference](../reference/configuration.md#storage-configuration) — full storage flag reference
