#!/usr/bin/env bash
# Build the mtrxAI inference cell Docker image.
#
# Usage:
#   scripts/build-icell-docker.sh [image-ref] [backend]
#
# Args:
#   image-ref  Docker tag (default: mtrx-icell-llamacpp:local)
#   backend    INFERENCE_BACKEND: llamacpp | ollama (default: llamacpp)
#
# Env:
#   MTRXAI_VERSION (optional, logged only)
#   LLAMA_IMAGE / OLLAMA_IMAGE (optional base image overrides)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_REF="${1:-mtrx-icell-llamacpp:local}"
BACKEND="${2:-llamacpp}"
VERSION="${MTRXAI_VERSION:-unknown}"

case "${BACKEND}" in
  llamacpp|ollama) ;;
  *)
    echo "error: backend must be llamacpp or ollama, got: ${BACKEND}" >&2
    exit 1
    ;;
esac

echo "==> Docker build ${IMAGE_REF} (backend=${BACKEND} version=${VERSION})"

BUILD_ARGS=(
  -f "${ROOT_DIR}/docker/Dockerfile"
  --build-arg "INFERENCE_BACKEND=${BACKEND}"
  -t "${IMAGE_REF}"
)

if [[ -n "${LLAMA_IMAGE:-}" ]]; then
  BUILD_ARGS+=(--build-arg "LLAMA_IMAGE=${LLAMA_IMAGE}")
fi
if [[ -n "${OLLAMA_IMAGE:-}" ]]; then
  BUILD_ARGS+=(--build-arg "OLLAMA_IMAGE=${OLLAMA_IMAGE}")
fi

docker build "${BUILD_ARGS[@]}" "${ROOT_DIR}"

echo "Built ${IMAGE_REF}"
