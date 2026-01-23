---
title: Chat History
---

# Chat History

SMG supports multiple storage backends for persisting conversation history, responses, and feedback data for analytics, debugging, and compliance.

---

## Overview

<div class="grid" markdown>

<div class="card" markdown>

### :material-database: Multiple Backends

Choose from in-memory, PostgreSQL, Redis, or Oracle based on your requirements.

</div>

<div class="card" markdown>

### :material-message-text: Conversation Tracking

Store complete conversation history including messages, tool calls, and reasoning.

</div>

<div class="card" markdown>

### :material-thumb-up: Feedback Collection

Collect user feedback on responses for quality monitoring and fine-tuning.

</div>

<div class="card" markdown>

### :material-tune: Configurable Retention

Control data retention with TTL settings for compliance and storage management.

</div>

</div>

---

## Backend Comparison

| Backend | Use Case | Persistence | Scalability |
|---------|----------|-------------|-------------|
| `memory` | Development, testing | Process lifetime | Single instance |
| `none` | Stateless deployments | None | N/A |
| `postgres` | Production, self-hosted | Durable | High |
| `redis` | Caching, ephemeral storage | Configurable TTL | High |
| `oracle` | Enterprise, OCI deployments | Durable | High |

---

## Configuration

### Backend Selection

```bash
smg --history-backend <backend> [backend-specific options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--history-backend` | `memory` | Storage backend: `memory`, `none`, `oracle`, `postgres`, `redis` |

---

## Memory Backend

The default in-process storage. Suitable for development and testing.

```bash
smg --history-backend memory
```

<div class="grid" markdown>

<div class="card" markdown>

#### :material-check-circle: Advantages

- Zero configuration
- Fast access
- No external dependencies

</div>

<div class="card" markdown>

#### :material-close-circle: Limitations

- Data lost on restart
- Not shared across instances
- Memory grows with conversations

</div>

</div>

---

## None Backend

Disables history storage entirely. Use for stateless deployments where persistence isn't needed.

```bash
smg --history-backend none
```

**Use when**: Privacy requirements prohibit storing conversations, or external systems handle logging.

---

## PostgreSQL Backend

Production-ready storage with PostgreSQL.

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--postgres-db-url` | - | PostgreSQL connection URL |
| `--postgres-pool-max-size` | `16` | Maximum connection pool size |

### Connection URL Format

```
postgres://[user[:password]@][host][:port][/database][?param=value]
```

### Examples

<div class="grid" markdown>

<div class="card" markdown>

#### Basic Connection

```bash
smg --history-backend postgres \
  --postgres-db-url "postgres://user:password@localhost:5432/smg"
```

</div>

<div class="card" markdown>

#### With SSL

```bash
smg --history-backend postgres \
  --postgres-db-url "postgres://user:password@db.example.com:5432/smg?sslmode=require"
```

</div>

</div>

### SSL Modes

| Mode | Description |
|------|-------------|
| `disable` | No SSL |
| `allow` | Try non-SSL first, then SSL |
| `prefer` | Try SSL first, then non-SSL (default) |
| `require` | Require SSL, skip verification |
| `verify-ca` | Require SSL, verify CA |
| `verify-full` | Require SSL, verify CA and hostname |

---

## Redis Backend

High-performance caching with optional persistence and TTL-based retention.

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--redis-url` | - | Redis connection URL |
| `--redis-pool-max-size` | `16` | Maximum connection pool size |
| `--redis-retention-days` | `30` | Data retention in days (-1 for persistent) |

### Connection URL Format

```
redis://[:password@]host[:port][/db]
rediss://[:password@]host[:port][/db]  # TLS
```

### Examples

<div class="grid" markdown>

<div class="card" markdown>

#### Basic Connection

```bash
smg --history-backend redis \
  --redis-url "redis://localhost:6379"
```

</div>

<div class="card" markdown>

#### With TLS and Auth

```bash
smg --history-backend redis \
  --redis-url "rediss://:password@redis.example.com:6379"
```

</div>

<div class="card" markdown>

#### Persistent Storage

```bash
smg --history-backend redis \
  --redis-url "redis://localhost:6379" \
  --redis-retention-days -1
```

</div>

</div>

---

## Oracle Backend

Enterprise-grade storage using Oracle Autonomous Database.

### Configuration Options

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `--oracle-wallet-path` | `ATP_WALLET_PATH` | - | Path to ATP wallet directory |
| `--oracle-tns-alias` | `ATP_TNS_ALIAS` | - | TNS alias from tnsnames.ora |
| `--oracle-dsn` | `ATP_DSN` | - | Direct connection descriptor |
| `--oracle-user` | `ATP_USER` | - | Database username |
| `--oracle-password` | `ATP_PASSWORD` | - | Database password |
| `--oracle-pool-min` | `ATP_POOL_MIN` | `1` | Minimum connection pool size |
| `--oracle-pool-max` | `ATP_POOL_MAX` | `16` | Maximum connection pool size |
| `--oracle-pool-timeout-secs` | `ATP_POOL_TIMEOUT_SECS` | `30` | Connection timeout in seconds |

### Examples

<div class="grid" markdown>

<div class="card" markdown>

#### Using ATP Wallet

```bash
smg --history-backend oracle \
  --oracle-wallet-path /path/to/wallet \
  --oracle-tns-alias mydb_high \
  --oracle-user admin \
  --oracle-password "$ORACLE_PASSWORD"
```

</div>

<div class="card" markdown>

#### Using Direct DSN

```bash
smg --history-backend oracle \
  --oracle-dsn "(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=db.example.com)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=myservice)))" \
  --oracle-user admin \
  --oracle-password "$ORACLE_PASSWORD"
```

</div>

</div>

---

## What Gets Stored

### Conversations

Container for a sequence of interactions:

- Conversation ID
- Creation timestamp
- Metadata (model, user, session info)

### Conversation Items

Individual items within a conversation:

| Type | Description |
|------|-------------|
| **Messages** | User and assistant messages with content |
| **Reasoning** | Model reasoning/thinking steps |
| **Tool Calls** | Tool invocations and results |
| **MCP Calls** | MCP server interactions |
| **Function Calls** | Function calling results |

### Responses

Complete response records including:

- Input (original request)
- Output (model response)
- Tool calls executed
- Model information
- Timestamps and metadata
- Token usage

### Feedback

User feedback on responses for quality tracking:

- Rating (positive/negative)
- Comments
- Timestamp
- Response reference

---

## Recommended Configurations

<div class="grid" markdown>

<div class="card" markdown>

### :material-laptop: Development

In-memory for fast iteration.

```bash
smg --history-backend memory
```

</div>

<div class="card" markdown>

### :material-server-network: Production (Self-Hosted)

PostgreSQL for durable storage.

```bash
smg --history-backend postgres \
  --postgres-db-url "postgres://smg:$DB_PASSWORD@postgres:5432/smg?sslmode=require" \
  --postgres-pool-max-size 32
```

</div>

<div class="card" markdown>

### :material-office-building: Enterprise (OCI)

Oracle for enterprise deployments.

```bash
smg --history-backend oracle \
  --oracle-wallet-path /etc/smg/wallet \
  --oracle-tns-alias smg_high \
  --oracle-user smg_app \
  --oracle-password "$ATP_PASSWORD" \
  --oracle-pool-max 32
```

</div>

<div class="card" markdown>

### :material-lightning-bolt: Caching Layer

Redis for high-performance ephemeral storage.

```bash
smg --history-backend redis \
  --redis-url "rediss://:$REDIS_PASSWORD@redis.example.com:6379" \
  --redis-retention-days 7 \
  --redis-pool-max-size 64
```

</div>

</div>

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| Connection timeouts | Slow network | Increase pool timeout |
| Pool exhaustion | High concurrency | Increase pool size |
| Data not persisting | Wrong backend | Verify `--history-backend` setting |
| Redis data expiring | TTL too short | Increase `--redis-retention-days` |

### Pool Configuration

```bash
# PostgreSQL
--postgres-db-url "postgres://...?connect_timeout=30"
--postgres-pool-max-size 64

# Oracle
--oracle-pool-timeout-secs 60
--oracle-pool-max 64

# Redis
--redis-pool-max-size 64
```

---

## What's Next?

<div class="grid" markdown>

<div class="card" markdown>

### :material-shield-lock: Authentication

Secure access to your SMG deployment.

[Authentication →](../security/authentication.md)

</div>

<div class="card" markdown>

### :material-chart-box: Metrics Reference

Monitor storage backend performance.

[Metrics Reference →](../../reference/metrics.md)

</div>

<div class="card" markdown>

### :material-shield-check: High Availability

Deploy SMG in a highly available configuration.

[High Availability →](../architecture/high-availability.md)

</div>

</div>
