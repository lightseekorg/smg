#!/bin/bash

if [ "$1" = "rocm" ]; then
    echo "Running in ROCm mode"

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

else
    # Show current GPU status
    nvidia-smi

    # Clean SGLang processes
    pgrep -f 'sglang::|sglang\.launch_server|sglang\.bench|sglang\.data_parallel|sglang\.srt|sgl_diffusion::' | xargs -r kill -9

    # Clean all GPU processes if any argument is provided
    if [ $# -gt 0 ]; then
        # Install lsof if not already available
        if ! command -v lsof >/dev/null 2>&1; then
            if command -v sudo >/dev/null 2>&1; then
                sudo apt-get update
                sudo apt-get install -y lsof
            else
                apt-get update
                apt-get install -y lsof
            fi
        fi
        kill -9 $(nvidia-smi | sed -n '/Processes:/,$p' | grep "   [0-9]" | awk '{print $5}') 2>/dev/null
        lsof /dev/nvidia* | awk '{print $2}' | xargs kill -9 2>/dev/null
    fi

    # Show GPU status after clean up
    nvidia-smi

    NODE_INFO="${NODE_IP:-$(hostname)}"
    echo "Running on node: $NODE_INFO"
    # Check for remaining GPU processes only when full nuke was attempted
    if [ $# -gt 0 ]; then
        # Give the GPU driver a moment to release compute contexts after kill -9
        sleep 2
        if ! REMAINING_PIDS_RAW=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
            echo "::error::GPU cleanup verification failed on node '$NODE_INFO': unable to query nvidia-smi."
            exit 1
        fi
        REMAINING_PIDS=$(printf '%s\n' "$REMAINING_PIDS_RAW" | awk '/^[0-9]+$/' | sort -u)
        if [ -n "$REMAINING_PIDS" ]; then
            echo "::error::GPU cleanup failed on node '$NODE_INFO'. Remaining processes:"
            for pid in $REMAINING_PIDS; do
                echo "  PID $pid: $(ps -p "$pid" -o user=,args= 2>/dev/null || echo 'process info unavailable')"
            done
            exit 1
        fi
    fi
fi
