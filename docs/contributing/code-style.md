---
title: Code Style Guide
---

# Code Style Guide

This guide describes the coding standards and conventions used in SMG.

---

## Rust Style

### Formatting

All code must be formatted with `rustfmt`:

```bash
cargo fmt
```

We use the default rustfmt configuration. Key points:

- 4 spaces for indentation
- 100 character line limit
- Trailing commas in multi-line constructs

### Linting

All code must pass clippy without warnings:

```bash
cargo clippy -- -D warnings
```

---

## Naming Conventions

### General Rules

| Item | Convention | Example |
|------|------------|---------|
| Crates | `snake_case` | `smg` |
| Modules | `snake_case` | `load_balancer` |
| Types | `PascalCase` | `CircuitBreaker` |
| Functions | `snake_case` | `get_healthy_workers` |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_RETRY_COUNT` |
| Variables | `snake_case` | `worker_count` |

### Specific Patterns

**Constructors**: Use `new()` or `with_*()`:

```rust
impl Config {
    pub fn new() -> Self { ... }
    pub fn with_timeout(timeout: Duration) -> Self { ... }
}
```

**Builders**: Use the builder pattern for complex configuration:

```rust
let gateway = Gateway::builder()
    .workers(workers)
    .policy(Policy::CacheAware)
    .build()?;
```

**Async functions**: Don't suffix with `_async`:

```rust
// Good
async fn fetch_models() -> Result<Vec<Model>>

// Bad
async fn fetch_models_async() -> Result<Vec<Model>>
```

---

## Code Organization

### Module Structure

```rust
// 1. Imports (grouped and sorted)
use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::config::Config;
use crate::error::Error;

// 2. Constants
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

// 3. Type definitions
type Result<T> = std::result::Result<T, Error>;

// 4. Main type(s)
pub struct Gateway {
    workers: Vec<Worker>,
    policy: Box<dyn Policy>,
}

// 5. Implementations
impl Gateway {
    pub fn new(config: Config) -> Self { ... }
}

// 6. Trait implementations
impl Default for Gateway {
    fn default() -> Self { ... }
}

// 7. Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gateway_creation() { ... }
}
```

### Import Organization

Group imports in this order, separated by blank lines:

1. Standard library (`std`)
2. External crates
3. Internal crates (`crate::`)

```rust
use std::collections::HashMap;
use std::sync::Arc;

use axum::{Router, routing::get};
use tokio::sync::mpsc;
use tracing::info;

use crate::config::Config;
use crate::routing::Policy;
```

---

## Error Handling

### Error Types

Define domain-specific errors using `thiserror`:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GatewayError {
    #[error("no healthy workers available")]
    NoHealthyWorkers,

    #[error("worker {url} is unhealthy: {reason}")]
    WorkerUnhealthy { url: String, reason: String },

    #[error("request timed out after {0:?}")]
    Timeout(Duration),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

### Error Propagation

Use `?` operator for error propagation:

```rust
// Good
fn process() -> Result<Response> {
    let data = fetch_data()?;
    let result = transform(data)?;
    Ok(result)
}

// Avoid
fn process() -> Result<Response> {
    let data = match fetch_data() {
        Ok(d) => d,
        Err(e) => return Err(e),
    };
    // ...
}
```

### Error Context

Add context using `anyhow` or custom errors:

```rust
use anyhow::Context;

fn load_config(path: &Path) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .context(format!("failed to read config from {}", path.display()))?;

    serde_json::from_str(&content)
        .context("failed to parse config JSON")
}
```

---

## Documentation

### Module Documentation

Every public module should have a doc comment:

```rust
//! Rate limiting implementation using token bucket algorithm.
//!
//! This module provides rate limiting for incoming requests to prevent
//! overloading workers.
//!
//! # Example
//!
//! ```rust
//! use smg::rate_limit::TokenBucket;
//!
//! let limiter = TokenBucket::new(100, 10);
//! if limiter.try_acquire() {
//!     // Process request
//! }
//! ```
```

### Function Documentation

Document all public functions:

```rust
/// Routes a request to an appropriate worker.
///
/// # Arguments
///
/// * `request` - The incoming HTTP request
///
/// # Returns
///
/// Returns the selected worker, or `None` if no healthy workers are available.
///
/// # Errors
///
/// Returns an error if the routing policy fails to make a selection.
///
/// # Example
///
/// ```rust
/// let worker = gateway.route(&request)?;
/// ```
pub fn route(&self, request: &Request) -> Result<Option<&Worker>> {
    // ...
}
```

### Inline Comments

Use inline comments sparingly, only when the code isn't self-explanatory:

```rust
// Good: explains why, not what
// Use a longer timeout for large requests to avoid false positives
let timeout = if request.body_size() > LARGE_REQUEST_THRESHOLD {
    Duration::from_secs(60)
} else {
    Duration::from_secs(30)
};

// Bad: explains what (obvious from code)
// Set timeout to 30 seconds
let timeout = Duration::from_secs(30);
```

---

## Testing

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Group related tests with descriptive names
    mod round_robin {
        use super::*;

        #[test]
        fn cycles_through_workers() { ... }

        #[test]
        fn skips_unhealthy_workers() { ... }
    }

    mod cache_aware {
        use super::*;

        #[test]
        fn prefers_cached_worker() { ... }
    }
}
```

### Test Naming

Use descriptive names that explain what's being tested:

```rust
// Good
#[test]
fn circuit_breaker_opens_after_threshold_failures()

#[test]
fn rate_limiter_rejects_when_bucket_empty()

// Bad
#[test]
fn test1()

#[test]
fn circuit_breaker_test()
```

### Test Assertions

Use specific assertions with clear messages:

```rust
// Good
assert_eq!(worker.health_status(), HealthStatus::Healthy,
    "worker should be healthy after successful health check");

// Bad
assert!(worker.health_status() == HealthStatus::Healthy);
```

---

## Performance

### Avoid Unnecessary Allocations

```rust
// Good: reuse buffer
let mut buffer = Vec::with_capacity(1024);
for item in items {
    buffer.clear();
    serialize_into(&mut buffer, item)?;
    send(&buffer).await?;
}

// Bad: allocate each iteration
for item in items {
    let buffer = serialize(item)?;
    send(&buffer).await?;
}
```

### Use Appropriate Data Structures

```rust
// Good: HashMap for frequent lookups
let workers: HashMap<String, Worker> = ...;

// Bad: Vec for frequent lookups
let workers: Vec<Worker> = ...;
workers.iter().find(|w| w.url == url) // O(n) each time
```

### Async Best Practices

```rust
// Good: concurrent operations
let (health1, health2) = tokio::join!(
    check_health(&worker1),
    check_health(&worker2),
);

// Bad: sequential when not needed
let health1 = check_health(&worker1).await;
let health2 = check_health(&worker2).await;
```

---

## Security

### Input Validation

Always validate external input:

```rust
pub fn parse_worker_url(input: &str) -> Result<Url> {
    let url = Url::parse(input)?;

    // Validate scheme
    if !matches!(url.scheme(), "http" | "https") {
        return Err(Error::InvalidScheme(url.scheme().to_string()));
    }

    // Validate host
    if url.host().is_none() {
        return Err(Error::MissingHost);
    }

    Ok(url)
}
```

### Sensitive Data

Never log sensitive data:

```rust
// Good
info!("authenticating request from {}", request.client_ip());

// Bad
info!("authenticating with key {}", api_key);
```

---

## Git Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code change that neither fixes nor adds
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:

```
feat(routing): add weighted round-robin policy

Implements a new routing policy that distributes requests
based on configurable weights per worker.

Closes #123
```

```
fix(health): handle connection timeout gracefully

Previously, connection timeouts would crash the health check
loop. Now they are logged and the worker is marked unhealthy.
```
