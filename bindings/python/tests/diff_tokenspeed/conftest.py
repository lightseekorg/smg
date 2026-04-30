"""Diff-test infrastructure: validates smg parsers/tokenizer match tokenspeed.

Skipped automatically if tokenspeed is not importable.
"""

from pathlib import Path

import pytest

try:
    import tokenspeed  # noqa: F401
    HAS_TOKENSPEED = True
except ImportError:
    HAS_TOKENSPEED = False


_DIFF_DIR = Path(__file__).parent.resolve()


def pytest_collection_modifyitems(config, items):
    if HAS_TOKENSPEED:
        return
    skip_marker = pytest.mark.skip(reason="tokenspeed not installed; install for parity tests")
    for item in items:
        # Only skip tests that live under this diff_tokenspeed directory; the hook is
        # invoked globally even though it's defined in a sub-conftest.
        try:
            item_path = Path(str(item.fspath)).resolve()
        except Exception:
            continue
        try:
            item_path.relative_to(_DIFF_DIR)
        except ValueError:
            continue
        item.add_marker(skip_marker)
