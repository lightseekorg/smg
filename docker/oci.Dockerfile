######################## BASE IMAGE ##########################
FROM ocr-docker-remote.artifactory.oci.oraclecorp.com/os/oraclelinux:9-slim-fips AS base
COPY --from=odo-docker-signed-local.artifactory.oci.oraclecorp.com/base-image-support/ol9:1.51 / /
ENTRYPOINT ["/sbin/simple_init.py"]

RUN microdnf install io-ol9-container-hardening \
    && rm -rf /var/cache/yum
RUN microdnf update -y && microdnf clean all

ARG PYTHON_VERSION=3.12

ENV PATH="/root/.local/bin:${PATH}"
ENV UV_HTTP_TIMEOUT=500
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_LINK_MODE="copy"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN microdnf install -y ca-certificates curl tar gzip \
    && microdnf clean all

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}

FROM scratch AS local_src
COPY . /src

######################### BUILD IMAGE #########################
FROM base AS build-image

ENV PATH="/root/.cargo/bin:${PATH}"

RUN microdnf install -y git gcc gcc-c++ make openssl-devel pkgconf-pkg-config perl \
    && microdnf install -y protobuf-compiler protobuf-devel --enablerepo=ol9-codeready-builder-yum-remote \
    && microdnf clean all

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rustc --version && cargo --version && protoc --version

COPY --from=local_src /src /opt/smg
WORKDIR /opt/smg

RUN uv pip install maturin \
    && cargo clean \
    && rm -rf bindings/python/dist/ \
    && cd bindings/python \
    && ulimit -n 65536 && maturin build --release --features vendored-openssl --out dist \
    && rm -rf /root/.cache

######################### ROUTER IMAGE #########################
FROM base AS router-image

COPY --from=build-image /opt/smg/bindings/python/dist/*.whl dist/

RUN uv pip install --force-reinstall dist/*.whl

RUN rm -rf /root/.cache dist/ \
    && microdnf clean all

ENTRYPOINT ["python3", "-m", "smg.launch_router"]
