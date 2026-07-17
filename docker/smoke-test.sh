#!/usr/bin/env bash
# Smoke test for mtrxai-icell against a running container.
# Usage: ./smoke-test.sh [https://127.0.0.1:8443] [CELL_ADMIN_TOKEN]
set -euo pipefail

BASE_URL="${1:-https://127.0.0.1:8443}"
TOKEN="${2:-}"
CURL=(curl -ksS)

if [[ -n "$TOKEN" ]]; then
  AUTH=(-H "Authorization: Bearer ${TOKEN}")
else
  AUTH=()
fi

echo "==> GET /mtrxai/v1/info"
INFO=$("${CURL[@]}" "${BASE_URL}/mtrxai/v1/info")
echo "$INFO"
ENGINE=$(echo "$INFO" | sed -n 's/.*"engine"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
if [[ -z "$ENGINE" ]]; then
  echo "FAIL: missing engine field in /mtrxai/v1/info" >&2
  exit 1
fi
echo "engine=${ENGINE}"

echo "==> GET /health"
"${CURL[@]}" "${BASE_URL}/health" | head -c 200
echo

if [[ "$ENGINE" == "ollama" ]]; then
  echo "==> GET /api/tags (Ollama)"
  "${CURL[@]}" "${BASE_URL}/api/tags" | head -c 200
  echo
else
  echo "==> GET /v1/models (llama.cpp)"
  "${CURL[@]}" "${BASE_URL}/v1/models" | head -c 200
  echo
fi

echo "==> POST /mtrxai/v1/models/unload (admin)"
STATUS=$("${CURL[@]}" -o /dev/null -w "%{http_code}" -X POST "${AUTH[@]}" \
  "${BASE_URL}/mtrxai/v1/models/unload")
if [[ "$STATUS" != "200" && "$STATUS" != "401" ]]; then
  echo "FAIL: unexpected unload status ${STATUS}" >&2
  exit 1
fi
echo "unload status=${STATUS}"

echo "OK: icell smoke test passed (engine=${ENGINE})"
