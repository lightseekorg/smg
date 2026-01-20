# Chat History Backend

SMG supports multiple storage backends for persisting conversation history, responses, and feedback data.

## Overview

| Backend | Use Case | Persistence | Scalability |
|---------|----------|-------------|-------------|
| `memory` | Development, testing | Process lifetime | Single instance |
| `none` | Stateless deployments | None | N/A |
| `oracle` | Enterprise, OCI deployments | Durable | High |
| `postgres` | Production, self-hosted | Durable | High |
| `redis` | Caching, ephemeral storage | Configurable TTL | High |

## Backend Selection

```bash
smg --history-backend <backend> [backend-specific options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--history-backend` | `memory` | Storage backend: `memory`, `none`, `oracle`, `postgres`, `redis` |

## Memory Backend

The default in-process storage. Suitable for development and testing.

```bash
smg --history-backend memory
```

**Limitations:**
- Data lost on restart
- Not shared across instances
- Memory usage grows with conversation count

## None Backend

Disables history storage entirely. Use for stateless deployments where persistence isn't needed.

```bash
smg --history-backend none
```

## Oracle Database

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

### Using ATP Wallet

```bash
smg --history-backend oracle \
  --oracle-wallet-path /path/to/wallet \
  --oracle-tns-alias mydb_high \
  --oracle-user admin \
  --oracle-password "$ORACLE_PASSWORD"
```

### Using Direct DSN

```bash
smg --history-backend oracle \
  --oracle-dsn "(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=db.example.com)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=myservice)))" \
  --oracle-user admin \
  --oracle-password "$ORACLE_PASSWORD"
```

## PostgreSQL

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

```bash
# Basic connection
smg --history-backend postgres \
  --postgres-db-url "postgres://user:password@localhost:5432/smg"

# With SSL
smg --history-backend postgres \
  --postgres-db-url "postgres://user:password@db.example.com:5432/smg?sslmode=require"

# With connection pool tuning
smg --history-backend postgres \
  --postgres-db-url "postgres://user:password@localhost:5432/smg" \
  --postgres-pool-max-size 32
```

### SSL Modes

| Mode | Description |
|------|-------------|
| `disable` | No SSL |
| `allow` | Try non-SSL first, then SSL |
| `prefer` | Try SSL first, then non-SSL (default) |
| `require` | Require SSL, skip verification |
| `verify-ca` | Require SSL, verify CA |
| `verify-full` | Require SSL, verify CA and hostname |

## Redis

High-performance caching with optional persistence.

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

```bash
# Basic connection
smg --history-backend redis \
  --redis-url "redis://localhost:6379"

# With authentication
smg --history-backend redis \
  --redis-url "redis://:password@localhost:6379/0"

# With TLS
smg --history-backend redis \
  --redis-url "rediss://:password@redis.example.com:6379"

# Persistent storage (no TTL)
smg --history-backend redis \
  --redis-url "redis://localhost:6379" \
  --redis-retention-days -1
```

## What Gets Stored

### Conversations

Container for a sequence of interactions:
- Conversation ID
- Creation timestamp
- Metadata

### Conversation Items

Individual items within a conversation:
- **Messages**: User and assistant messages
- **Reasoning**: Model reasoning/thinking steps
- **Tool Calls**: Tool invocations and results
- **MCP Calls**: MCP server interactions
- **Function Calls**: Function calling results

### Responses

Complete response records including:
- Input (original request)
- Output (model response)
- Tool calls executed
- Model information
- Timestamps and metadata

### Feedback

User feedback on responses for quality tracking.

## Configuration Examples

### Development

```bash
# In-memory for fast iteration
smg --history-backend memory
```

### Production (Self-Hosted)

```bash
smg --history-backend postgres \
  --postgres-db-url "postgres://smg:$DB_PASSWORD@postgres:5432/smg?sslmode=require" \
  --postgres-pool-max-size 32
```

### Enterprise (OCI)

```bash
smg --history-backend oracle \
  --oracle-wallet-path /etc/smg/wallet \
  --oracle-tns-alias smg_high \
  --oracle-user smg_app \
  --oracle-password "$ATP_PASSWORD" \
  --oracle-pool-max 32
```

### Caching Layer

```bash
smg --history-backend redis \
  --redis-url "rediss://:$REDIS_PASSWORD@redis.example.com:6379" \
  --redis-retention-days 7 \
  --redis-pool-max-size 64
```

## Troubleshooting

### Connection Timeouts

Increase pool timeout for slow networks:
```bash
--oracle-pool-timeout-secs 60
--postgres-db-url "postgres://...?connect_timeout=30"
```

### Pool Exhaustion

Increase pool size for high concurrency:
```bash
--oracle-pool-max 64
--postgres-pool-max-size 64
--redis-pool-max-size 64
```

### Data Not Persisting

- Verify `--history-backend` is set correctly
- Check database connectivity and credentials
- For Redis, ensure `--redis-retention-days` isn't set to 0
