# mtrxAI icell (inference cell)

Local LLM proxy / cell runtime (`mtrxai-icell`).

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
