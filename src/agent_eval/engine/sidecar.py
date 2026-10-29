"""The VCR proxy as a sidecar container on the sandbox's isolated network.

The sandbox network is `internal` (no route to the host or internet), so the proxy must live on
it as a container. In RECORD mode the sidecar is also attached to the default bridge so it can
reach the real upstream; the agent container never gets that route.
"""

from __future__ import annotations

import io
import json
import tarfile
import time
from pathlib import Path

import docker

from ..proxy import MatchPolicy, ProxyMode
from .image import ensure_proxy_image
from .sandbox import Sandbox

ALIAS = "proxy"
PORT = 8080
CASSETTE_PATH = "/data/cassette.jsonl"
EXCHANGE_LOG_PATH = "/data/exchanges.jsonl"
KEY_ENV = "UPSTREAM_API_KEY"


def _tar_with(path: str, data: bytes) -> bytes:
    """A tar that creates the parent dir and the file, for `put_archive("/")`."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        parent = tarfile.TarInfo(path.rsplit("/", 1)[0].lstrip("/"))
        parent.type, parent.mode = tarfile.DIRTYPE, 0o777
        tar.addfile(parent)
        info = tarfile.TarInfo(path.lstrip("/"))
        info.size, info.mode = len(data), 0o666
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class ProxySidecar:
    def __init__(
        self,
        sandbox: Sandbox,
        mode: ProxyMode,
        *,
        cassette: Path | None = None,
        upstream: str | None = None,
        upstream_api_key: str | None = None,
        match: MatchPolicy = MatchPolicy.STRICT,
        stream_speed: float = 0.0,
    ) -> None:
        if mode is ProxyMode.REPLAY and (cassette is None or not cassette.exists()):
            raise FileNotFoundError(f"REPLAY needs an existing cassette, got {cassette}")
        if mode is ProxyMode.RECORD and not upstream:
            raise ValueError("RECORD needs an upstream URL")
        self.sandbox, self.mode = sandbox, mode
        self.cassette, self.upstream = cassette, upstream
        self.upstream_api_key = upstream_api_key
        self.match, self.stream_speed = match, stream_speed
        self.container = None

    @property
    def base_url(self) -> str:
        return f"http://{ALIAS}:{PORT}"

    def start(self, timeout: float = 30) -> None:
        client: docker.DockerClient = self.sandbox.client
        image = ensure_proxy_image(client)
        cmd = [
            "--mode", self.mode.value,
            "--cassette", CASSETTE_PATH,
            "--exchange-log", EXCHANGE_LOG_PATH,
            "--match", self.match.value,
            "--port", str(PORT),
            "--stream-speed", str(self.stream_speed),
        ]  # fmt: skip
        env = {}
        if self.mode is ProxyMode.RECORD:
            cmd += ["--upstream", self.upstream]
            if self.upstream_api_key:
                cmd += ["--inject-header", f"x-api-key={KEY_ENV}"]
                env[KEY_ENV] = self.upstream_api_key

        net = self.sandbox.network.name
        api = client.api
        created = api.create_container(
            image,
            command=cmd,
            environment=env,
            labels=self.sandbox.labels,
            host_config=api.create_host_config(
                extra_hosts={"host.docker.internal": "host-gateway"}, mem_limit="512m"
            ),
            networking_config=api.create_networking_config(
                {net: api.create_endpoint_config(aliases=[ALIAS])}
            ),
        )
        self.container = client.containers.get(created["Id"])
        self.sandbox.adopt(self.container)
        if self.mode is ProxyMode.RECORD:
            client.networks.get("bridge").connect(self.container)  # egress to the upstream
        if self.cassette is not None:
            self.container.put_archive("/", _tar_with(CASSETTE_PATH, self.cassette.read_bytes()))
        else:
            self.container.put_archive("/", _tar_with(CASSETTE_PATH, b""))
        self.container.put_archive("/", _tar_with(EXCHANGE_LOG_PATH, b""))
        self.container.start()
        self._wait_ready(timeout)

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.container.reload()
            logs = self.container.logs().decode(errors="replace")
            if "listening" in logs:
                return
            if self.container.status in {"exited", "dead"}:
                raise RuntimeError(f"proxy sidecar exited during startup:\n{logs}")
            time.sleep(0.2)
        raise TimeoutError("proxy sidecar did not become ready")

    def _read(self, path: str) -> bytes:
        code, out = self.container.exec_run(["cat", path])
        if code != 0:
            raise RuntimeError(f"cannot read {path} from proxy sidecar: {out.decode()}")
        return out

    def fetch_cassette(self) -> bytes:
        return self._read(CASSETTE_PATH)

    def fetch_exchanges(self) -> list[dict]:
        lines = self._read(EXCHANGE_LOG_PATH).decode().splitlines()
        return [json.loads(ln) for ln in lines if ln.strip()]
