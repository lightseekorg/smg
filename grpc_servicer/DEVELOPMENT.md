# smg-grpc-servicer Development Guide

## Local Development

Install both proto and servicer as editable packages:

```bash
pip install -e crates/grpc_client/python/
pip install -e grpc_servicer/
```

No version concerns locally — editable installs always use the latest source.

## CI

PR tests install both `smg-grpc-proto` and `smg-grpc-servicer` from source (not PyPI),
so changes to either package are always tested against the PR's code.

This is handled in `scripts/ci_install_vllm.sh`:

```bash
uv pip install vllm
pip install -e crates/grpc_client/python/
pip install -e grpc_servicer/
```

## Release Process

Both `smg-grpc-proto` and `smg-grpc-servicer` are published to PyPI via GitHub Actions.
Releases are triggered by version bumps in the respective `pyproject.toml` files on `main`.

### Scenario 1: Changing only the servicer

1. Make changes to servicer code
2. Bump version in `grpc_servicer/pyproject.toml`
3. Open PR, CI tests pass (both packages installed from source)
4. Merge to `main` — `release-grpc-servicer.yml` fires, publishes to PyPI

### Scenario 2: Changing only the proto

1. Make changes to proto files and regenerate stubs
2. Bump version in `crates/grpc_client/python/pyproject.toml`
3. Open PR, CI tests pass
4. Merge to `main` — `release-grpc-proto.yml` fires, publishes to PyPI

### Scenario 3: Changing both proto and servicer (one PR)

1. Make all changes (proto + servicer) in a single PR
2. Bump versions in both `crates/grpc_client/python/pyproject.toml` and `grpc_servicer/pyproject.toml`
3. Open PR, CI tests pass (both installed from source)
4. Merge to `main` — both release workflows trigger
5. `release-grpc-servicer.yml` waits for `release-grpc-proto.yml` to complete via `workflow_run`
6. Proto publishes first, then servicer publishes (can resolve the new proto from PyPI)

### Scenario 4: Changing both proto and servicer (two PRs)

1. Open PR 1 with proto changes, bump proto version
2. Merge PR 1 — proto publishes to PyPI
3. Open PR 2 with servicer changes, bump servicer version
4. Merge PR 2 — servicer publishes to PyPI (new proto already available)

## Version Pinning

- `smg-grpc-servicer` pins `smg-grpc-proto >= X.Y.Z` loosely
- This means in Scenario 3, the previous proto version satisfies the requirement during
  the brief window between proto and servicer publishing
- If the servicer uses new proto fields that don't exist in the old version, use Scenario 4
  (two separate PRs) instead
