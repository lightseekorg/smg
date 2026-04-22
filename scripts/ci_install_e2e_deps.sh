#!/bin/bash
# Install e2e test dependencies
# Usage: ci_install_e2e_deps.sh [extra_deps...]

set -euo pipefail

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Installing e2e test dependencies..."
python3 -m pip install e2e_test/

# Install SmgClient (pure Python client for cross-SDK parity testing)
echo "Installing smg-client..."
python3 -m pip install clients/python/

# Install any extra dependencies passed as arguments
if [ $# -gt 0 ]; then
    # sentence-transformers >= 3.x top-level imports torchcodec via its
    # modality_types module. torchcodec dlopens libtorchcodec_core{4..8}.so,
    # which link against FFmpeg shared libs (libavformat/libavcodec/libavutil/
    # libswresample/libswscale). The runner image does not ship these, so
    # install ffmpeg from apt before pip-installing the extra.
    if [[ " $* " == *" sentence-transformers "* ]]; then
        echo "Installing ffmpeg (libav*) for torchcodec..."
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends ffmpeg
    fi

    echo "Installing extra dependencies: $@"
    python3 -m pip --no-cache-dir install --upgrade "$@"
fi

echo "E2E test dependencies installed"
