# mtrxAI icell (inference cell)

Local LLM proxy / cell runtime (`mtrxai-icell`). Production **peer + icell** compose uses this image as the sealed engine behind an attested peer.

**Why a cell (vs raw Ollama):** peer binary attestation does not cover the LLM process. Icell keeps the engine on loopback, publishes only HTTPS `:8443`, and is the intended target of future `ATTESTATION_LLM_SERVER`. Details: [mtrxAI-peer/docs/ATTESTATION.md](../mtrxAI-peer/docs/ATTESTATION.md).

## Build

```bash
cargo build --release
```

## Docker

```bash
docker build -f docker/Dockerfile -t mtrxai-icell .
# or:
scripts/build-icell-docker.sh mtrx-icell-llamacpp:local llamacpp
scripts/build-icell-docker.sh mtrx-icell-ollama:local ollama
```

See `docker/` for compose and smoke tests.

## Release (GitHub Actions)

Tag `vX.Y.Z` or run **Release Container** (`workflow_dispatch`) to build multi-arch images and push:

- `mtrxai/mtrx-icell-llamacpp` (llama.cpp backend)
- `mtrxai/mtrx-icell-ollama` (Ollama backend)

Requires repo secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`. Version comes from the tag, the workflow input, or [`release.toml`](release.toml).
