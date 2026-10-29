from __future__ import annotations

from pathlib import Path

import docker
from docker.errors import ImageNotFound

BASE_IMAGE = "agent-eval-base:latest"
DEFAULT_CONTEXT = Path(__file__).resolve().parents[3] / "images"


def ensure_image(
    client: docker.DockerClient,
    tag: str = BASE_IMAGE,
    context: Path = DEFAULT_CONTEXT,
    rebuild: bool = False,
) -> None:
    """Build the image from `context` unless it already exists."""
    if not rebuild:
        try:
            client.images.get(tag)
            return
        except ImageNotFound:
            pass
    client.images.build(path=str(context), tag=tag, rm=True)
