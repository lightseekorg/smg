#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-smg-vllm-grpc-smoke:latest}"
PLATFORM="${PLATFORM:-linux/amd64}"
BASE_IMAGE="${BASE_IMAGE:-python:3.12-slim}"
MODE="install"
BUILD_ONLY=0
NO_BUILD=0
MODEL=""
USE_BASE_VLLM=0
VLLM_SPEC="${VLLM_SPEC:-vllm}"
DOCKER_ARGS=()
RUN_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  scripts/run_vllm_grpc_servicer_smoke.sh [options]

Options:
  --mode install|serve   Smoke mode. `install` verifies fresh installs and the
                         vLLM gRPC entrypoint. `serve` also boots the server.
  --model MODEL          Required for `--mode serve`.
  --image-tag TAG        Docker image tag to build/run.
  --platform PLATFORM    Docker platform. Defaults to linux/amd64 because
                         upstream vLLM wheels are typically published there.
  --base-image IMAGE     Base image to build FROM. Useful when Docker Hub is
                         blocked or you already have a local Python image.
  --use-base-vllm        Reuse the `vllm` installation already present in the
                         base image instead of reinstalling it in the smoke env.
  --vllm-spec SPEC       Python package spec to install for vLLM.
                         Default: `vllm`
  --build-only           Build the image but do not run the smoke test.
  --no-build             Reuse an existing image tag.
  --docker-arg ARG       Extra `docker build` argument. Repeatable.
  --run-arg ARG          Extra `docker run` argument. Repeatable.
  -h, --help             Show this help text.

Examples:
  scripts/run_vllm_grpc_servicer_smoke.sh
  scripts/run_vllm_grpc_servicer_smoke.sh --base-image localhost/python:3.12-slim
  scripts/run_vllm_grpc_servicer_smoke.sh --base-image vllm/vllm-openai:latest --use-base-vllm
  scripts/run_vllm_grpc_servicer_smoke.sh --vllm-spec 'vllm>=0.14.0'
  scripts/run_vllm_grpc_servicer_smoke.sh --mode serve --model meta-llama/Llama-3.1-8B-Instruct --run-arg=--gpus=all
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --base-image)
            BASE_IMAGE="$2"
            shift 2
            ;;
        --use-base-vllm)
            USE_BASE_VLLM=1
            shift
            ;;
        --vllm-spec)
            VLLM_SPEC="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=1
            shift
            ;;
        --no-build)
            NO_BUILD=1
            shift
            ;;
        --docker-arg)
            DOCKER_ARGS+=("$2")
            shift 2
            ;;
        --docker-arg=*)
            DOCKER_ARGS+=("${1#*=}")
            shift
            ;;
        --run-arg)
            RUN_ARGS+=("$2")
            shift 2
            ;;
        --run-arg=*)
            RUN_ARGS+=("${1#*=}")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${MODE}" != "install" && "${MODE}" != "serve" ]]; then
    echo "--mode must be install or serve" >&2
    exit 2
fi

if [[ "${MODE}" == "serve" && -z "${MODEL}" ]]; then
    echo "--model is required for --mode serve" >&2
    exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required" >&2
    exit 1
fi

PROXY_VARS=(HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy)

sanitize_cmd_for_logging() {
    local sanitized=()
    local token
    for token in "$@"; do
        local redacted=0
        local var_name
        for var_name in "${PROXY_VARS[@]}"; do
            if [[ "${token}" == "${var_name}="* ]]; then
                sanitized+=("${var_name}=<redacted>")
                redacted=1
                break
            fi
        done
        if [[ "${redacted}" -eq 0 ]]; then
            sanitized+=("${token}")
        fi
    done
    printf ' %q' "${sanitized[@]}"
}

if [[ "${NO_BUILD}" -eq 0 ]]; then
    BUILD_CMD=(
        docker build
        --platform "${PLATFORM}"
        -t "${IMAGE_TAG}"
        --build-arg "BASE_IMAGE=${BASE_IMAGE}"
        -f "${ROOT_DIR}/grpc_servicer/docker/vllm-smoke.Dockerfile"
    )
    for var_name in "${PROXY_VARS[@]}"; do
        var_value="$(printenv "${var_name}" 2>/dev/null || true)"
        if [[ -n "${var_value}" ]]; then
            BUILD_CMD+=(--build-arg "${var_name}=${var_value}")
        fi
    done
    if [[ ${#DOCKER_ARGS[@]} -gt 0 ]]; then
        for arg in "${DOCKER_ARGS[@]}"; do
            BUILD_CMD+=("${arg}")
        done
    fi
    BUILD_CMD+=("${ROOT_DIR}")

    echo "==> Building smoke image ${IMAGE_TAG}"
    printf 'command:'
    sanitize_cmd_for_logging "${BUILD_CMD[@]}"
    printf '\n'
    "${BUILD_CMD[@]}"
fi

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
    exit 0
fi

RUN_CMD=(
    docker run
    --rm
    --platform "${PLATFORM}"
    -e "WORKSPACE=/workspace"
    -e "VLLM_SPEC=${VLLM_SPEC}"
)
for var_name in "${PROXY_VARS[@]}"; do
    var_value="$(printenv "${var_name}" 2>/dev/null || true)"
    if [[ -n "${var_value}" ]]; then
        RUN_CMD+=(-e "${var_name}=${var_value}")
    fi
done

if [[ -n "${MODEL}" ]]; then
    RUN_CMD+=(-e "SMOKE_MODEL=${MODEL}")
fi

if [[ "${USE_BASE_VLLM}" -eq 1 ]]; then
    RUN_CMD+=(-e "SKIP_VLLM_INSTALL=1")
fi

if [[ -d "${HOME}/.cache/huggingface" ]]; then
    RUN_CMD+=(-v "${HOME}/.cache/huggingface:/root/.cache/huggingface")
fi

if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then
    for arg in "${RUN_ARGS[@]}"; do
        RUN_CMD+=("${arg}")
    done
fi

RUN_CMD+=("${IMAGE_TAG}" "${MODE}")

echo "==> Running ${MODE} smoke test"
printf 'command:'
sanitize_cmd_for_logging "${RUN_CMD[@]}"
printf '\n'
"${RUN_CMD[@]}"
