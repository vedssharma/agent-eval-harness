from __future__ import annotations

import hashlib
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


REPO_ROOT = Path(__file__).resolve().parents[3]


def proxy_image_tag(root: Path = REPO_ROOT) -> str:
    """Tag derived from the proxy sources + Dockerfile, so code changes force a rebuild."""
    digest = hashlib.sha256()
    pkg = root / "src/agent_eval"
    files = sorted(
        [*(pkg / "proxy").rglob("*.py"), pkg / "schemas.py", pkg / "__init__.py"]
    ) + [root / "images/proxy.Dockerfile"]
    for f in files:
        digest.update(f.relative_to(root).as_posix().encode())
        digest.update(f.read_bytes())
    return f"agent-eval-proxy:{digest.hexdigest()[:12]}"


def ensure_proxy_image(client: docker.DockerClient, root: Path = REPO_ROOT) -> str:
    tag = proxy_image_tag(root)
    try:
        client.images.get(tag)
    except ImageNotFound:
        client.images.build(path=str(root), dockerfile="images/proxy.Dockerfile", tag=tag, rm=True)
    return tag
