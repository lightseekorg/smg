"""Diff-test infrastructure: validates smg parsers/tokenizer match tokenspeed.

Skipped automatically if tokenspeed is not importable.
"""

import pytest

try:
    import tokenspeed  # noqa: F401

    HAS_TOKENSPEED = True
except ImportError:
    HAS_TOKENSPEED = False


def pytest_collection_modifyitems(config, items):
    if HAS_TOKENSPEED:
        return
    skip_marker = pytest.mark.skip(reason="tokenspeed not installed; install for parity tests")
    for item in items:
        if "diff_tokenspeed" in item.keywords:
            item.add_marker(skip_marker)
