# SMG Python Bindings

This directory contains the Python bindings for SMG (Shepherd Model Gateway), built using [maturin](https://github.com/PyO3/maturin) and [PyO3](https://github.com/PyO3/pyo3).

## Directory Structure

```
bindings/python/
├── src/                    # Source code (src layout)
│   ├── lib.rs              # Rust/PyO3 bindings implementation
│   └── smg/                # Python source code
│       ├── __init__.py
│       ├── version.py
│       ├── launch_server.py
│       ├── launch_router.py
│       ├── router.py
│       ├── router_args.py
│       └── mini_lb.py
├── tests/                  # Python unit tests
│   ├── conftest.py
│   ├── test_validation.py
│   ├── test_arg_parser.py
│   ├── test_router_config.py
│   └── test_startup_sequence.py
├── Cargo.toml              # Rust package configuration for bindings
├── pyproject.toml          # Python package configuration
├── setup.py                # Setup configuration
├── MANIFEST.in             # Package manifest
├── .coveragerc             # Test coverage configuration
└── README.md               # This file
```

## Building

### Development Build

```bash
# Install maturin
pip install maturin

# Build and install in development mode
cd smg/bindings/python
maturin develop --features vendored-openssl
```

### Production Build

```bash
# Build wheel
cd smg/bindings/python
maturin build --release --out dist --features vendored-openssl

# Install the built wheel
pip install dist/smg-*.whl
```

## Testing

```bash
# Run Python unit tests (after maturin develop)
cd smg/bindings/python
pytest tests/
```

## Configuration

- **pyproject.toml**: Defines package metadata, dependencies, and build configuration
- **python-source**: Set to `"src"` indicating Python source uses the src layout
- **module-name**: `smg.smg_rs` - the Rust extension module name

## Notes

- The Rust bindings source code is located in `src/lib.rs`
- The bindings have their own `Cargo.toml` in this directory
- The main SMG library is located in `../../model_gateway/` and is used as a dependency
- The package includes both Python code and Rust extensions built with PyO3
- PyO3 types are prefixed with `Py` in Rust but exposed to Python without the prefix using the `name` attribute
