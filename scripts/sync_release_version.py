#!/usr/bin/env python3
"""Read release.toml and sync version into icell manifests in this repo."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_TOML = ROOT / "release.toml"

CARGO_MANIFESTS = [
    ROOT / "Cargo.toml",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def read_release_config() -> dict[str, str]:
    text = _read_text(RELEASE_TOML)
    version = _require_match(r'^version\s*=\s*"([^"]+)"', text, "version")
    registry = _require_match(r'^registry\s*=\s*"([^"]+)"', text, "docker.registry")
    icell_llamacpp_image = _require_match(
        r'^icell_llamacpp_image\s*=\s*"([^"]+)"', text, "docker.icell_llamacpp_image"
    )
    ollama_image = _require_match(r'^ollama_image\s*=\s*"([^"]+)"', text, "docker.ollama_image")
    return {
        "version": version,
        "docker_registry": registry.rstrip("/"),
        "docker_icell_llamacpp_image": icell_llamacpp_image,
        "docker_ollama_image": ollama_image,
    }


def _require_match(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise SystemExit(f"release.toml: missing {label}")
    return match.group(1)


def set_release_version(version: str) -> None:
    version = version.removeprefix("v").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+(-[\w.-]+)?(\+[\w.-]+)?", version):
        raise SystemExit(f"invalid semver: {version!r}")
    text = _read_text(RELEASE_TOML)
    text = re.sub(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    _write_text(RELEASE_TOML, text)


def _sync_cargo(path: Path, version: str) -> None:
    text = _read_text(path)
    text = re.sub(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    _write_text(path, text)


def sync_all(version: str | None = None) -> str:
    if version is not None:
        set_release_version(version)
    cfg = read_release_config()
    version = cfg["version"]
    for manifest in CARGO_MANIFESTS:
        if not manifest.is_file():
            raise SystemExit(f"missing manifest: {manifest}")
        _sync_cargo(manifest, version)
    return version


def print_env() -> None:
    cfg = read_release_config()
    registry = cfg["docker_registry"]
    version = cfg["version"]
    icell_llamacpp = cfg["docker_icell_llamacpp_image"]
    ollama = cfg["docker_ollama_image"]
    print(f"MTRXAI_VERSION={version}")
    print(f"MTRXAI_DOCKER_REGISTRY={registry}")
    print(f"MTRXAI_DOCKER_ICELL_LLAMACPP_IMAGE={registry}/{icell_llamacpp}")
    print(f"MTRXAI_DOCKER_OLLAMA_IMAGE={registry}/{ollama}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: sync_release_version.py <get-version|sync|print-env> [version]")

    command = sys.argv[1]
    if command == "get-version":
        print(read_release_config()["version"])
        return
    if command == "print-env":
        print_env()
        return
    if command == "sync":
        version = sys.argv[2] if len(sys.argv) > 2 else None
        print(sync_all(version))
        return
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
