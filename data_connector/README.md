# data-connector

**Version:** 2.0.0 | **License:** Apache-2.0

Pluggable storage abstraction for the Shepherd Model Gateway (SMG). Provides
trait-based backends for persisting conversations, conversation items, and
responses. Backend selection is a runtime decision driven by configuration,
allowing the same application binary to target in-memory storage during
development and a production database in deployment.

## Architecture

### Core Traits

The crate defines three async traits in `core.rs`. Every backend implements all
three:

| Trait | Responsibility |
|-------|---------------|
| `ConversationStorage` | CRUD for conversation records (create, get, update, delete) |
| `ConversationItemStorage` | Item creation, linking items to conversations, cursor-based listing, deletion of links |
| `ResponseStorage` | Store/retrieve responses, walk response chains, index and query by safety identifier |

All traits are `Send + Sync + 'static` and use `async_trait` so they can be
held behind `Arc<dyn Trait>`.

### Factory Pattern

`create_storage()` is the single entry point. It accepts a
`StorageFactoryConfig` (backend selector plus optional per-backend configs) and
returns a `StorageTuple`:

```rust
pub type StorageTuple = (
    Arc<dyn ResponseStorage>,
    Arc<dyn ConversationStorage>,
    Arc<dyn ConversationItemStorage>,
);
```

### Supported Backends

| Backend | Variant | Description |
|---------|---------|-------------|
| **Memory** | `HistoryBackend::Memory` | In-process `HashMap`/`BTreeMap` storage. Default. No persistence across restarts. Suitable for development and testing. |
| **None / NoOp** | `HistoryBackend::None` | Accepts all writes silently, returns empty on reads. Use when persistence is intentionally disabled. |
| **PostgreSQL** | `HistoryBackend::Postgres` | Production backend using `tokio-postgres` with `deadpool` connection pooling. Fully async. Tables are auto-created on first connection. |
| **Redis** | `HistoryBackend::Redis` | Production backend using `deadpool-redis`. Supports optional TTL-based data retention (`retention_days`). |
| **Oracle ATP** | `HistoryBackend::Oracle` | Enterprise backend using the synchronous `oracle` crate. Async bridging is handled via `tokio::task::spawn_blocking`. Tables are auto-created on initialization. |

## Usage

```rust
use data_connector::{
    create_storage, StorageFactoryConfig, HistoryBackend,
    NewConversation, NewConversationItem, StoredResponse,
};
use serde_json::json;
use std::sync::Arc;

// Build factory config -- memory backend needs no extra configuration.
let config = StorageFactoryConfig {
    backend: &HistoryBackend::Memory,
    oracle: None,
    postgres: None,
    redis: None,
};

let (responses, conversations, items) = create_storage(config).await.unwrap();

// Create a conversation
let conv = conversations
    .create_conversation(NewConversation { id: None, metadata: None })
    .await
    .unwrap();

// Store a response
let mut resp = StoredResponse::new(None);
resp.input = json!([{"role": "user", "content": "Hello"}]);
let resp_id = responses.store_response(resp).await.unwrap();

// Create and link a conversation item
let item = items
    .create_item(NewConversationItem {
        id: None,
        response_id: Some(resp_id.0.clone()),
        item_type: "message".to_string(),
        role: Some("user".to_string()),
        content: json!([]),
        status: Some("completed".to_string()),
    })
    .await
    .unwrap();

items.link_item(&conv.id, &item.id, chrono::Utc::now()).await.unwrap();
```

## Configuration

Backend selection is controlled by the `HistoryBackend` enum
(`"memory"`, `"none"`, `"oracle"`, `"postgres"`, `"redis"` when deserialized
from JSON/YAML). Each database backend has a dedicated config struct.

### `PostgresConfig`

| Field | Type | Description |
|-------|------|-------------|
| `db_url` | `String` | Connection URL (`postgres://user:pass@host:port/dbname`). Validated for scheme, host, and database name. |
| `pool_max` | `usize` | Maximum connections in the deadpool pool (default helper: 16). Must be > 0. |
| `schema` | `Option<SchemaConfig>` | Optional schema customization. See [Schema Configuration](#schema-configuration). |

Call `validate()` to check the URL before use.

### `RedisConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | `String` | -- | Connection URL (`redis://` or `rediss://`). |
| `pool_max` | `usize` | 16 | Maximum pool connections. |
| `retention_days` | `Option<u64>` | `Some(30)` | TTL in days for stored data. `None` disables expiration. |
| `schema` | `Option<SchemaConfig>` | `None` | Optional schema customization. See [Schema Configuration](#schema-configuration). |

Call `validate()` to check the URL before use.

### `OracleConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wallet_path` | `Option<String>` | `None` | Path to ATP wallet / TLS config directory. |
| `connect_descriptor` | `String` | -- | DSN, e.g. `tcps://host:port/service`. |
| `external_auth` | `bool` | `false` | Use OS/external authentication. |
| `username` | `String` | -- | Database username. |
| `password` | `String` | -- | Database password (redacted in `Debug` output). |
| `pool_min` | `usize` | 1 | Minimum pool connections. |
| `pool_max` | `usize` | 16 | Maximum pool connections. |
| `pool_timeout_secs` | `u64` | 30 | Connection acquisition timeout in seconds. |
| `schema` | `Option<SchemaConfig>` | `None` | Optional schema customization. See [Schema Configuration](#schema-configuration). |

### Schema Configuration

All three database backends (Oracle, Postgres, Redis) accept an optional
`SchemaConfig` that lets you customize table names and column names without
modifying source code. When `schema` is omitted, all backends use their
default table and column names — zero behavioral change.

```yaml
postgres:
  db_url: "postgres://user:pass@localhost:5432/mydb"
  pool_max: 16

  schema:
    owner: "myschema"           # Oracle: schema prefix (MYSCHEMA."TABLE")
                                # Redis: key prefix ("myschema:conversation:{id}")
                                # Postgres: ignored (use search_path for schema control)

    conversations:
      table: "my_conversations" # Overrides default "conversations"
      columns:
        id: "conv_id"           # Overrides column name "id" -> "conv_id"
        metadata: "conv_meta"   # Overrides column name "metadata" -> "conv_meta"

    responses:
      table: "my_responses"
      columns:
        safety_identifier: "user_identifier"

    conversation_items:
      table: "my_items"

    conversation_item_links:
      table: "my_links"
```

`SchemaConfig` has two types:

| Type | Fields | Purpose |
|------|--------|---------|
| `SchemaConfig` | `owner`, `conversations`, `responses`, `conversation_items`, `conversation_item_links` | Top-level config with an optional owner/prefix and per-table settings |
| `TableConfig` | `table`, `columns` | Per-table config: physical table name and a map of logical-to-physical column name overrides |

Key behaviors:

- **`col(field)`** returns the physical column name for a logical field name.
  If no override is configured, the logical name is returned unchanged.
- **`qualified_table(owner)`** returns `OWNER."TABLE"` when an owner is set
  (used by Oracle), or just the table name otherwise.
- **Validation** runs at startup. All identifiers must match `[a-zA-Z0-9_]+`.
  Invalid identifiers are rejected before any queries execute.
- **Redis**: Only `owner` (key prefix) and `columns` (hash field names) affect
  Redis behavior. The `table` field is ignored for Redis key patterns — keys
  always use hardcoded entity names (`conversation`, `item`, `response`).

## Data Model

### Conversations

A `Conversation` has an `id` (`ConversationId`), a `created_at` timestamp, and
optional JSON `metadata`. Conversations act as containers for ordered sets of
conversation items.

### Conversation Items

A `ConversationItem` represents a single turn or event within a conversation.
Items are created independently and then linked to a conversation via
`link_item()`. Listing uses cursor-based pagination (`ListParams` with `limit`,
`order`, and an optional `after` cursor).

Fields: `id`, `response_id` (optional back-reference), `item_type`, `role`,
`content` (JSON), `status`, `created_at`.

### Responses

A `StoredResponse` captures a complete model interaction: input, output,
instructions, tool calls, metadata, model identifier, and raw response payload.
Responses support chaining via `previous_response_id` and can be queried by
`safety_identifier` for content moderation workflows. The `ResponseChain`
struct reconstructs the chronological sequence of related responses.

## ID Generation

| ID Type | Format | Example |
|---------|--------|---------|
| `ConversationId` | `conv_` + 50 random hex chars (25 bytes) | `conv_a1b2c3...` |
| `ConversationItemId` | `{prefix}_` + 50 random hex chars | `msg_d4e5f6...` |
| `ResponseId` | ULID (26 chars, lexicographically sortable, millisecond precision) | `01ARZ3NDEKTSV4RRFFQ69G5FAV` |

**Item type prefixes:**

| `item_type` | Prefix |
|-------------|--------|
| `message` | `msg` |
| `reasoning` | `rs` |
| `mcp_call` | `mcp` |
| `mcp_list_tools` | `mcpl` |
| `function_call` | `fc` |
| (other) | first 3 chars of the type, or `itm` if empty |

## Database Schema

All database backends auto-create their schemas on first connection. The
following default table names are used (configurable via `SchemaConfig`):

| Default Table | Purpose |
|---------------|---------|
| `conversations` | Conversation records with metadata |
| `conversation_items` | Individual items (messages, tool calls, etc.) |
| `conversation_item_links` | Join table linking items to conversations with ordering (`added_at`) |
| `responses` | Stored model responses with chaining and safety identifier indexing |

PostgreSQL additionally creates an index on
`conversation_item_links(conversation_id, added_at)` for efficient
cursor-based listing.

Column names within each table can also be overridden via `SchemaConfig`. The
config describes the existing database schema — it does not perform migrations.
If you rename a column in config, the corresponding database column must
already exist with that name.

## Testing

Run the unit tests (Memory and NoOp backends, config validation, ID generation):

```bash
cargo test -p data-connector
```

Integration tests against live PostgreSQL, Redis, or Oracle instances require
the corresponding backend to be running and configured.
