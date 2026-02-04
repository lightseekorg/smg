#!/usr/bin/env bash
#
# Launch Prefill-Decode (PD) disaggregated workers for SMG
#
# Usage:
#   ./scripts/launch-pd-workers.sh sglang /path/to/model   # Launch SGLang PD workers
#   ./scripts/launch-pd-workers.sh vllm /path/to/model     # Launch vLLM PD workers
#
# Environment variables:
#   PREFILL_GPU=0          GPU ID for prefill worker (default: 0)
#   DECODE_GPU=1           GPU ID for decode worker (default: 1)
#   PREFILL_PORT=50051     gRPC port for prefill worker (default: 50051)
#   DECODE_PORT=50052      gRPC port for decode worker (default: 50052)
#   BOOTSTRAP_PORT=8998    Bootstrap port for SGLang PD (default: 8998)
#   MAX_MODEL_LEN=4096     Maximum model length (default: 4096)
#   GPU_MEM_UTIL=0.9       GPU memory utilization (default: 0.9)
#   TP_SIZE=1              Tensor parallel size (default: 1)
#
# Examples:
#   # Launch SGLang PD with default settings
#   ./scripts/launch-pd-workers.sh sglang /raid/models/meta-llama/Llama-3.1-8B-Instruct
#
#   # Launch vLLM PD on specific GPUs
#   PREFILL_GPU=2 DECODE_GPU=3 ./scripts/launch-pd-workers.sh vllm /raid/models/meta-llama/Llama-3.1-8B-Instruct
#

set -euo pipefail

# Colors for output
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# Default configuration
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
PREFILL_PORT="${PREFILL_PORT:-50051}"
DECODE_PORT="${DECODE_PORT:-50052}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
TP_SIZE="${TP_SIZE:-1}"

# NIXL side-channel ports for vLLM (must be unique per worker)
NIXL_PREFILL_PORT="${NIXL_PREFILL_PORT:-5600}"
NIXL_DECODE_PORT="${NIXL_DECODE_PORT:-5601}"

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

print_usage() {
    echo "Launch Prefill-Decode (PD) disaggregated workers for SMG"
    echo ""
    echo "Usage: $0 <runtime> <model_path>"
    echo ""
    echo "Arguments:"
    echo "  runtime     Runtime type: 'sglang' or 'vllm'"
    echo "  model_path  Path to the model directory"
    echo ""
    echo "Environment variables:"
    echo "  PREFILL_GPU      GPU ID for prefill worker (default: 0)"
    echo "  DECODE_GPU       GPU ID for decode worker (default: 1)"
    echo "  PREFILL_PORT     gRPC port for prefill worker (default: 50051)"
    echo "  DECODE_PORT      gRPC port for decode worker (default: 50052)"
    echo "  BOOTSTRAP_PORT   Bootstrap port for SGLang PD (default: 8998)"
    echo "  MAX_MODEL_LEN    Maximum model length (default: 4096)"
    echo "  GPU_MEM_UTIL     GPU memory utilization (default: 0.9)"
    echo "  TP_SIZE          Tensor parallel size (default: 1)"
    echo ""
    echo "Examples:"
    echo "  $0 sglang /raid/models/meta-llama/Llama-3.1-8B-Instruct"
    echo "  PREFILL_GPU=2 DECODE_GPU=3 $0 vllm /raid/models/meta-llama/Llama-3.1-8B-Instruct"
}

launch_sglang_pd() {
    local model_path="$1"

    info "Launching SGLang PD workers..."
    info "  Model: $model_path"
    info "  Prefill: GPU $PREFILL_GPU, port $PREFILL_PORT, bootstrap $BOOTSTRAP_PORT"
    info "  Decode:  GPU $DECODE_GPU, port $DECODE_PORT"

    # Launch prefill worker
    info "Starting prefill worker..."
    CUDA_VISIBLE_DEVICES=$PREFILL_GPU python3 -m sglang.launch_server \
        --model-path "$model_path" \
        --host 0.0.0.0 \
        --port "$PREFILL_PORT" \
        --tp-size "$TP_SIZE" \
        --mem-fraction-static "$GPU_MEM_UTIL" \
        --disaggregation-mode prefill \
        --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
        --grpc-mode &

    local prefill_pid=$!
    info "Prefill worker started (PID: $prefill_pid)"

    # Wait for prefill to be ready
    sleep 10

    # Launch decode worker
    info "Starting decode worker..."
    CUDA_VISIBLE_DEVICES=$DECODE_GPU python3 -m sglang.launch_server \
        --model-path "$model_path" \
        --host 0.0.0.0 \
        --port "$DECODE_PORT" \
        --tp-size "$TP_SIZE" \
        --mem-fraction-static "$GPU_MEM_UTIL" \
        --disaggregation-mode decode \
        --base-gpu-id 0 \
        --grpc-mode &

    local decode_pid=$!
    info "Decode worker started (PID: $decode_pid)"

    echo ""
    info "SGLang PD workers launched successfully!"
    echo -e "${BLUE}Prefill:${NC} grpc://localhost:$PREFILL_PORT (bootstrap: $BOOTSTRAP_PORT)"
    echo -e "${BLUE}Decode:${NC}  grpc://localhost:$DECODE_PORT"
    echo ""
    info "To start SMG gateway in PD mode:"
    echo "  smg --pd-disaggregation --prefill grpc://localhost:$PREFILL_PORT $BOOTSTRAP_PORT --decode grpc://localhost:$DECODE_PORT"
    echo ""
    info "Press Ctrl+C to stop workers"

    # Wait for both processes
    wait
}

launch_vllm_pd() {
    local model_path="$1"

    info "Launching vLLM PD workers with NIXL..."
    info "  Model: $model_path"
    info "  Prefill: GPU $PREFILL_GPU, port $PREFILL_PORT, NIXL port $NIXL_PREFILL_PORT"
    info "  Decode:  GPU $DECODE_GPU, port $DECODE_PORT, NIXL port $NIXL_DECODE_PORT"

    # Launch prefill worker (kv_producer)
    info "Starting prefill worker (kv_producer)..."
    CUDA_VISIBLE_DEVICES=$PREFILL_GPU \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PREFILL_PORT \
    python3 -m vllm.entrypoints.grpc_server \
        --model "$model_path" \
        --host 0.0.0.0 \
        --port "$PREFILL_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' &

    local prefill_pid=$!
    info "Prefill worker started (PID: $prefill_pid)"

    # Wait for prefill to be ready
    sleep 10

    # Launch decode worker (kv_consumer)
    info "Starting decode worker (kv_consumer)..."
    CUDA_VISIBLE_DEVICES=$DECODE_GPU \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_DECODE_PORT \
    python3 -m vllm.entrypoints.grpc_server \
        --model "$model_path" \
        --host 0.0.0.0 \
        --port "$DECODE_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' &

    local decode_pid=$!
    info "Decode worker started (PID: $decode_pid)"

    echo ""
    info "vLLM PD workers launched successfully!"
    echo -e "${BLUE}Prefill:${NC} grpc://localhost:$PREFILL_PORT"
    echo -e "${BLUE}Decode:${NC}  grpc://localhost:$DECODE_PORT"
    echo ""
    info "To start SMG gateway in PD mode:"
    echo "  smg --pd-disaggregation --prefill grpc://localhost:$PREFILL_PORT --decode grpc://localhost:$DECODE_PORT"
    echo ""
    info "Press Ctrl+C to stop workers"

    # Wait for both processes
    wait
}

main() {
    if [[ $# -lt 2 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        print_usage
        exit 0
    fi

    local runtime="$1"
    local model_path="$2"

    # Validate model path
    if [[ ! -d "$model_path" ]]; then
        error "Model path does not exist: $model_path"
    fi

    # Validate GPU IDs are different
    if [[ "$PREFILL_GPU" == "$DECODE_GPU" ]]; then
        warn "PREFILL_GPU and DECODE_GPU are the same ($PREFILL_GPU). This may cause issues."
    fi

    # Validate ports are different
    if [[ "$PREFILL_PORT" == "$DECODE_PORT" ]]; then
        error "PREFILL_PORT and DECODE_PORT must be different"
    fi

    # Validate NIXL ports for vLLM
    if [[ "$runtime" == "vllm" && "$NIXL_PREFILL_PORT" == "$NIXL_DECODE_PORT" ]]; then
        error "NIXL_PREFILL_PORT and NIXL_DECODE_PORT must be different"
    fi

    case "$runtime" in
        sglang)
            launch_sglang_pd "$model_path"
            ;;
        vllm)
            launch_vllm_pd "$model_path"
            ;;
        *)
            error "Unknown runtime: $runtime. Use 'sglang' or 'vllm'."
            ;;
    esac
}

main "$@"
