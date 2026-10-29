"""Docker/Podman client discovery.

The docker SDK does not read Docker CLI contexts, so on Docker Desktop (macOS) the default
socket is often missing. Try DOCKER_HOST first, then well-known socket locations.
"""

from __future__ import annotations

import os
from pathlib import Path

import docker
from docker.errors import DockerException


def _candidates() -> list[str | None]:
    home = Path.home()
    paths = [
        home / ".docker/run/docker.sock",  # Docker Desktop (macOS)
        Path("/var/run/docker.sock"),
        Path(f"/run/user/{os.getuid()}/podman/podman.sock"),  # rootless Podman (Linux)
        home / ".local/share/containers/podman/machine/podman.sock",  # Podman machine (macOS)
    ]
    return [None, *(f"unix://{p}" for p in paths if p.exists())]  # None -> from_env()


def get_client() -> docker.DockerClient:
    errors: list[str] = []
    for base_url in _candidates():
        try:
            client = docker.from_env() if base_url is None else docker.DockerClient(base_url)
            client.ping()
            return client
        except DockerException as exc:
            errors.append(f"{base_url or 'env'}: {exc}")
    raise RuntimeError("No container runtime reachable:\n  " + "\n  ".join(errors))
