# Sealed inference cell: mtrxai-icell (TLS proxy + HF pull) + llama-server or ollama serve.
#
# llama.cpp (default):
#   docker build -f docker/Dockerfile --build-arg INFERENCE_BACKEND=llamacpp \
#     --build-arg LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda -t mtrxai-icell .
#
# Ollama:
#   docker build -f docker/Dockerfile --build-arg INFERENCE_BACKEND=ollama \
#     --build-arg OLLAMA_IMAGE=ollama/ollama:latest -t mtrxai-icell-ollama .

ARG INFERENCE_BACKEND=llamacpp
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server
ARG OLLAMA_IMAGE=ollama/ollama:latest

# --- Stage 1: build Rust icell supervisor/proxy ---
FROM rust:1.89-bookworm AS cell-builder
WORKDIR /build
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

# --- Stage 2a: llama.cpp runtime ---
FROM ${LLAMA_IMAGE} AS runtime-llamacpp

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /bin/bash /usr/bin/bash /bin/sh /usr/bin/sh /bin/dash /usr/bin/dash \
    && (getent group celluser >/dev/null 2>&1 || groupadd --system --gid 65532 celluser) \
    && (id -u celluser >/dev/null 2>&1 || useradd --system --uid 65532 --gid 65532 --home-dir /app --shell /usr/sbin/nologin celluser)

COPY --from=cell-builder /build/target/release/mtrxai-icell /usr/local/bin/mtrxai-icell

ENV PATH=/app:/usr/local/bin \
    LD_LIBRARY_PATH=/app \
    INFERENCE_BACKEND=llamacpp \
    LLAMA_SERVER_BIN=/app/llama-server \
    LLAMA_HOST=127.0.0.1 \
    LLAMA_PORT=8080 \
    HTTPS_BIND=0.0.0.0 \
    HTTPS_PORT=8443 \
    MODELS_DIR=/models \
    MODELS_MAX=4 \
    TLS_ROTATE_SECS=3600 \
    TLS_CERT_CN=llama-inference.local \
    AUTO_LOAD_AFTER_PULL=true \
    MODEL_MAX_BYTES=85899345920 \
    HOME=/tmp \
    HF_HOME=/tmp/huggingface \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app
EXPOSE 8443
VOLUME ["/models"]
ENTRYPOINT ["/usr/local/bin/mtrxai-icell"]

# --- Stage 2b: Ollama runtime ---
FROM ${OLLAMA_IMAGE} AS runtime-ollama

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /bin/bash /usr/bin/bash /bin/sh /usr/bin/sh /bin/dash /usr/bin/dash \
    && (getent group celluser >/dev/null 2>&1 || groupadd --system --gid 65532 celluser) \
    && (id -u celluser >/dev/null 2>&1 || useradd --system --uid 65532 --gid 65532 --home-dir /app --shell /usr/sbin/nologin celluser)

COPY --from=cell-builder /build/target/release/mtrxai-icell /usr/local/bin/mtrxai-icell

ENV PATH=/usr/local/bin:/usr/bin:/bin \
    INFERENCE_BACKEND=ollama \
    OLLAMA_BIN=/usr/bin/ollama \
    OLLAMA_HOST=127.0.0.1 \
    OLLAMA_PORT=11434 \
    OLLAMA_MODELS=/models/.ollama \
    OLLAMA_ORIGINS=* \
    HTTPS_BIND=0.0.0.0 \
    HTTPS_PORT=8443 \
    MODELS_DIR=/models \
    TLS_ROTATE_SECS=3600 \
    TLS_CERT_CN=llama-inference.local \
    AUTO_LOAD_AFTER_PULL=true \
    MODEL_MAX_BYTES=85899345920 \
    DELETE_GGUF_AFTER_IMPORT=false \
    HOME=/tmp

WORKDIR /app
EXPOSE 8443
VOLUME ["/models"]
ENTRYPOINT ["/usr/local/bin/mtrxai-icell"]

# --- Final stage selected by INFERENCE_BACKEND build arg ---
FROM runtime-${INFERENCE_BACKEND} AS runtime
