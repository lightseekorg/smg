---
title: Development Guide
---

# Development Guide

This guide covers setting up a development environment and contributing code to SMG.

---

## Prerequisites

- **Rust**: 1.75 or later
- **Docker**: For running integration tests
- **Git**: For version control

---

## Setting Up

### 1. Clone the Repository

```bash
git clone https://github.com/lightseekorg/smg.git
cd smg
```

### 2. Install Rust

```bash
# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Ensure you have the latest stable
rustup update stable
rustup default stable

# Install nightly toolchain (required for rustfmt unstable options)
rustup toolchain install nightly
rustup component add rustfmt --toolchain nightly
```

### 3. Install Development Tools

```bash
# Install clippy and rustfmt
rustup component add clippy rustfmt

# Install cargo-watch for auto-rebuild (optional)
cargo install cargo-watch
```

### 4. Build the Project

```bash
# Debug build
cargo build

# Release build
cargo build --release
```

---

## Project Structure

```
smg/
├── model_gateway/            # Main gateway binary
│   ├── src/
│   │   ├── main.rs           # Entry point
│   │   ├── lib.rs            # Library root
│   │   ├── config/           # Configuration handling
│   │   ├── core/             # Core routing logic
│   │   ├── policies/         # Load balancing policies
│   │   ├── routers/          # HTTP/gRPC routers
│   │   └── observability/    # Metrics and tracing
│   ├── benches/              # Benchmarks
│   ├── tests/                # Integration tests
│   └── Cargo.toml            # Gateway package manifest
├── protocols/                # OpenAI protocol definitions
├── tokenizer/                # LLM tokenization
├── tool_parser/              # Tool call parsing
├── reasoning_parser/         # Reasoning extraction
├── mcp/                      # MCP integration
├── auth/                     # Authentication
├── mesh/                     # HA mesh networking
├── wasm/                     # WebAssembly plugins
├── grpc_client/              # gRPC client
├── data_connector/           # Storage backends
├── kv_index/                 # KV cache indexing
├── multimodal/               # Multimodal support
├── workflow/                 # Workflow automation
├── bindings/
│   ├── python/               # Python bindings
│   └── golang/               # Go SDK
├── docs/                     # Documentation (MkDocs)
├── e2e_test/                 # End-to-end tests
├── examples/                 # Example configurations
└── Cargo.toml                # Workspace manifest
```

---

## Development Workflow

### Running Locally

```bash
# Start with a mock worker
cargo run -- --worker-urls http://localhost:8000 --log-level debug

# With hot reload
cargo watch -x 'run -- --worker-urls http://localhost:8000'
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_round_robin

# Run with output
cargo test -- --nocapture

# Run integration tests (requires Docker)
cargo test --test integration
```

### Linting and Formatting (Rust)

```bash
# Format code (requires nightly for unstable rustfmt options)
cargo +nightly fmt --all

# Check formatting
cargo +nightly fmt --all -- --check

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings

# Fix clippy warnings automatically
cargo clippy --all-targets --all-features --fix --allow-dirty -- -D warnings
```

### Linting and Formatting (Python)

Python code in `e2e_test/`, `bindings/python/`, and `scripts/` is checked with
[ruff](https://docs.astral.sh/ruff/) (linting + formatting) and
[mypy](https://mypy-lang.org/) (type checking).

**Pre-commit hooks** run these checks automatically on every commit. To set up:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg  # enables DCO sign-off check
```

### DCO Sign-Off

All commits must include a `Signed-off-by` line ([Developer Certificate of Origin](https://developercertificate.org/)).
Use the `-s` flag when committing:

```bash
git commit -s -m "feat: add new feature"
```

This adds a line like `Signed-off-by: Your Name <your@email.com>` using your
`user.name` and `user.email` from git config. The pre-commit `commit-msg` hook
and CI will both reject commits without it.

To run checks manually:

```bash
# Lint (with auto-fix)
ruff check --fix e2e_test/ bindings/python/ scripts/

# Format
ruff format e2e_test/ bindings/python/ scripts/

# Type check
mypy e2e_test/ --config-file mypy.ini
mypy bindings/python/ --config-file mypy.ini
```

Configuration lives in `ruff.toml` and `mypy.ini` at the repo root.

---

## Testing

### Unit Tests

Unit tests are co-located with source code:

```rust
// model_gateway/src/policies/round_robin.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_cycles() {
        let workers = vec!["w1", "w2", "w3"];
        let mut rr = RoundRobin::new(workers.clone());

        assert_eq!(rr.next(), "w1");
        assert_eq!(rr.next(), "w2");
        assert_eq!(rr.next(), "w3");
        assert_eq!(rr.next(), "w1");
    }
}
```

### Integration Tests

Integration tests are in `model_gateway/tests/`:

```rust
// model_gateway/tests/routing/routing_test.rs

#[tokio::test]
async fn test_routing_with_failing_worker() {
    let gateway = TestGateway::start().await;
    gateway.add_worker("http://healthy:8000", true);
    gateway.add_worker("http://unhealthy:8000", false);

    // All requests should go to healthy worker
    for _ in 0..10 {
        let response = gateway.request("/v1/chat/completions").await;
        assert_eq!(response.status(), 200);
    }
}
```

### End-to-End Tests

E2E tests use Docker Compose:

```bash
# Run E2E tests
./scripts/e2e-test.sh
```

---

## Adding a New Feature

### Example: Adding a New Routing Policy

1. **Create the policy module**:

```rust
// model_gateway/src/policies/weighted.rs

use super::{Policy, Worker};

pub struct WeightedRoundRobin {
    workers: Vec<(Worker, u32)>,
    current: usize,
    count: u32,
}

impl Policy for WeightedRoundRobin {
    fn select(&mut self, _request: &Request) -> Option<&Worker> {
        // Implementation
    }
}
```

2. **Add to policies module**:

```rust
// model_gateway/src/policies/mod.rs

mod weighted;
pub use weighted::WeightedRoundRobin;
```

3. **Add CLI option**:

```rust
// model_gateway/src/config/mod.rs

#[derive(Clone, Debug, ValueEnum)]
pub enum RoutingPolicy {
    Random,
    RoundRobin,
    PowerOfTwo,
    CacheAware,
    Weighted,  // New policy
}
```

4. **Write tests**:

```rust
// model_gateway/src/policies/weighted.rs

#[cfg(test)]
mod tests {
    #[test]
    fn test_weighted_distribution() {
        // Test that requests are distributed according to weights
    }
}
```

5. **Update documentation**:

- Add to CLI reference
- Add to configuration reference
- Add to load balancing concepts

---

## Debugging

### Logging

```bash
# Enable debug logging
RUST_LOG=debug cargo run -- ...

# Enable trace logging for specific module
RUST_LOG=smg::routing=trace cargo run -- ...
```

### Using lldb/gdb

```bash
# Build with debug symbols
cargo build

# Run with lldb
lldb target/debug/smg -- --worker-urls http://localhost:8000
```

### Profiling

```bash
# CPU profiling with flamegraph
cargo install flamegraph
cargo flamegraph -- --worker-urls http://localhost:8000

# Memory profiling with heaptrack
heaptrack target/release/smg --worker-urls http://localhost:8000
```

---

## Documentation

### Building Docs

```bash
# Install MkDocs
pip install mkdocs-material

# Serve locally
mkdocs serve

# Build for production
mkdocs build
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Follow the existing structure (Concepts, Tasks, Tutorials, Reference)
- Test all code examples

---

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Creating a Release

1. Update version in `Cargo.toml`
2. Update CHANGELOG.md
3. Create a release PR
4. After merge, tag the release:

```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

---

## Common Issues

### Build Fails on macOS

```bash
# Install OpenSSL
brew install openssl

# Set environment variables
export OPENSSL_DIR=$(brew --prefix openssl)
```

### Tests Fail with "Address already in use"

Tests may conflict if run in parallel:

```bash
# Run tests serially
cargo test -- --test-threads=1
```

### Clippy Errors in CI

Run clippy locally before pushing:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

---

## Getting Help

- **Stuck?** Open a [Discussion](https://github.com/lightseekorg/smg/discussions)
- **Found a bug?** Open an [Issue](https://github.com/lightseekorg/smg/issues)
- **Have questions about a PR?** Tag a maintainer
