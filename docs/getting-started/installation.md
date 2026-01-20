---
title: Installation
---

# Installation

This page describes how to install Shepherd Model Gateway.

<div class="prerequisites" markdown>

#### Before you begin

- Docker 20.10+ (for container installation)
- Rust 1.75+ and Cargo (for building from source)
- 2 GB RAM minimum, 4 GB recommended

</div>

## Installation Methods

Choose the method that best fits your environment:

| Method | Best For | Complexity |
|--------|----------|------------|
| [Docker](#docker) | Quick start, production | Low |
| [Pre-built Binary](#pre-built-binary) | Bare metal, custom deployments | Low |
| [From Source](#from-source) | Development, customization | Medium |
| [Python Package](#python-package) | Integration with Python workflows | Low |

---

## Docker

The recommended way to run SMG in production. Multi-architecture images are available for both x86_64 and ARM64.

```bash
docker pull lightseekorg/smg:latest
```

### Verify the installation

```bash
docker run --rm lightseekorg/smg:latest --version
```

### Available tags

| Tag | Description |
|-----|-------------|
| `latest` | Latest stable release |
| `v0.3.x` | Specific version |
| `main` | Latest development build |

---

## Pre-built Binary

Download a pre-compiled binary for your platform.

=== "Linux (x86_64)"

    ```bash
    curl -LO https://github.com/lightseekorg/smg/releases/latest/download/smg-linux-amd64
    chmod +x smg-linux-amd64
    sudo mv smg-linux-amd64 /usr/local/bin/smg
    ```

=== "Linux (ARM64)"

    ```bash
    curl -LO https://github.com/lightseekorg/smg/releases/latest/download/smg-linux-arm64
    chmod +x smg-linux-arm64
    sudo mv smg-linux-arm64 /usr/local/bin/smg
    ```

=== "macOS (Apple Silicon)"

    ```bash
    curl -LO https://github.com/lightseekorg/smg/releases/latest/download/smg-darwin-arm64
    chmod +x smg-darwin-arm64
    sudo mv smg-darwin-arm64 /usr/local/bin/smg
    ```

### Verify the installation

```bash
smg --version
```

---

## From Source

Build SMG from source for development or customization.

### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Clone and build

```bash
git clone https://github.com/lightseekorg/smg.git
cd smg
cargo build --release
```

The binary is available at `./target/release/smg`.

### Build options

| Profile | Command | Use Case |
|---------|---------|----------|
| Debug | `cargo build` | Development |
| Release | `cargo build --release` | Production |
| CI | `cargo build --profile ci` | Faster CI builds |

---

## Python Package

For integration with Python-based workflows and the SGLang ecosystem.

### Install with pip

```bash
pip install maturin
cd smg/bindings/python
maturin develop  # Development mode
```

### Production build

```bash
maturin build --release --out dist --features vendored-openssl
pip install dist/*.whl
```

### Verify the installation

```python
import smg
print(smg.__version__)
```

---

## System Requirements

### Minimum

| Resource | Requirement |
|----------|-------------|
| CPU | 2 cores |
| Memory | 2 GB |
| Disk | 100 MB |
| Network | 1 Gbps |

### Recommended (Production)

| Resource | Requirement |
|----------|-------------|
| CPU | 4+ cores |
| Memory | 8 GB |
| Disk | 1 GB (for logs) |
| Network | 10 Gbps |

---

## What's Next?

After installing SMG, proceed to the [Quickstart](quickstart.md) to deploy your first gateway.
