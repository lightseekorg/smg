"""
Pytest configuration for smg Python binding tests.

These are unit tests that run without GPU resources or external dependencies.
"""

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test (no GPU required)")


@pytest.fixture(scope="session")
def hf_tokenizer_path(tmp_path_factory) -> str:
    """Download a small public HF tokenizer for tests; cache for the whole session.

    Uses Qwen/Qwen2.5-0.5B-Instruct (~1MB tokenizer files; supports chat template + tools).
    Returns a path that `smg.Tokenizer.from_file` can accept.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        pytest.skip("huggingface_hub not installed; required for tokenizer tests")

    target = tmp_path_factory.mktemp("hf_tokenizer")
    return snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        local_dir=str(target),
        allow_patterns=["tokenizer*", "special_tokens_map.json", "vocab.json", "merges.txt"],
    )
