"""Put this repo's ``grpc_servicer/`` at the front of ``sys.path`` so
``smg_grpc_servicer`` imports resolve in-repo when tests run from the repo root
(as CI does), without an editable install.
"""

import sys
from pathlib import Path

_GRPC_SERVICER_ROOT = Path(__file__).resolve().parent.parent
if sys.path[:1] != [str(_GRPC_SERVICER_ROOT)]:
    sys.path.insert(0, str(_GRPC_SERVICER_ROOT))
