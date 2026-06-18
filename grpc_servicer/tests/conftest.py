"""Pytest configuration for the gRPC servicer tests.

Some tests load engine-free modules (e.g. ``vllm/kv_events.py``) by file path,
and those modules re-export from the shared ``smg_grpc_servicer.kv_events``
package module. Put this repo's ``grpc_servicer/`` directory at the front of
``sys.path`` so that package import resolves to the in-repo source — without
requiring an editable install, and taking precedence over any stale copy that
might be installed elsewhere on the path. This lets the suite run from the repo
root (as CI does) the same way it runs from inside ``grpc_servicer/``.
"""

import sys
from pathlib import Path

_GRPC_SERVICER_ROOT = Path(__file__).resolve().parent.parent
if sys.path[:1] != [str(_GRPC_SERVICER_ROOT)]:
    sys.path.insert(0, str(_GRPC_SERVICER_ROOT))
