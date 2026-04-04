ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV WORKSPACE=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN bash -o pipefail -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace
COPY crates/grpc_client/proto /workspace/crates/grpc_client/proto
COPY crates/grpc_client/python /workspace/crates/grpc_client/python
COPY grpc_servicer /workspace/grpc_servicer

ENTRYPOINT ["/workspace/grpc_servicer/docker/vllm-smoke.sh"]
CMD ["install"]
