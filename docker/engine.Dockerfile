# Parameterized engine image builder.
#
# Build args:
#   BASE_IMAGE_REF  - full image:tag to start FROM
#   ENGINE          - engine name: vllm | sglang | trtllm | tgl
#   BACKEND         - SMG_DEFAULT_BACKEND value (defaults to ENGINE; tgl overrides to sglang)
#   ENGINE_REPO     - if set, engine source is cloned and install-<ENGINE>.sh runs
#   ENGINE_COMMIT   - commit/ref for ENGINE_REPO ("latest" = HEAD)
#   SMG_REPO        - SMG source repo URL
#   SMG_COMMIT      - commit/ref for SMG_REPO ("latest" = HEAD)
#
# Usage:
#   docker build --build-arg BASE_IMAGE_REF=lmsysorg/sglang:v0.5.10 \
#                --build-arg ENGINE=sglang \
#                --build-arg SMG_REPO=https://github.com/lightseekorg/smg \
#                --build-arg SMG_COMMIT=v1.1.0 \
#                -f docker/engine.Dockerfile .

ARG BASE_IMAGE_REF

# ── sources stage: clone repos, stage install scripts ────────────────────────
FROM alpine:3.19 AS sources
ARG ENGINE_REPO
ARG ENGINE_COMMIT
ARG SMG_REPO
ARG SMG_COMMIT
RUN apk add --no-cache git \
    && if [ -z "${SMG_REPO}" ] || [ -z "${SMG_COMMIT}" ]; then \
         echo "ERROR: SMG_REPO and SMG_COMMIT must be set" >&2; exit 1; fi \
    && if [ -n "${ENGINE_REPO}" ] && [ -n "${ENGINE_COMMIT}" ]; then \
         if [ "${ENGINE_COMMIT}" = "latest" ]; then \
           git clone --depth 1 "${ENGINE_REPO}" /opt/engine-src; \
         else \
           git clone "${ENGINE_REPO}" /opt/engine-src \
           && ( cd /opt/engine-src && git checkout "${ENGINE_COMMIT}" ); \
         fi; \
       else mkdir -p /opt/engine-src; fi \
    && if [ "${SMG_COMMIT}" = "latest" ]; then \
         git clone --depth 1 "${SMG_REPO}" /tmp/smg-src; \
       else \
         git clone "${SMG_REPO}" /tmp/smg-src \
         && ( cd /tmp/smg-src && git checkout "${SMG_COMMIT}" ); \
       fi
COPY scripts/installation/ /tmp/scripts/

# ── final stage: install SMG + conditionally install engine ──────────────────
FROM ${BASE_IMAGE_REF}

ARG ENGINE=sglang
ARG BACKEND
ARG ENGINE_REPO

ENV SMG_DEFAULT_BACKEND=${BACKEND:-${ENGINE}}

COPY --from=sources /opt/engine-src   /opt/engine-src
COPY --from=sources /tmp/smg-src      /opt/smg-src
COPY --from=sources /tmp/scripts/     /tmp/scripts/

RUN bash /tmp/scripts/install-smg.sh /opt/smg-src

RUN case "${ENGINE}" in \
      vllm|sglang|trtllm|tgl) ;; \
      *) echo "ERROR: Unknown ENGINE '${ENGINE}'" >&2; exit 1 ;; \
    esac \
    && if [ -n "${ENGINE_REPO}" ]; then \
         bash /tmp/scripts/install-${ENGINE}.sh /opt/engine-src; \
       fi

ENTRYPOINT ["smg"]
