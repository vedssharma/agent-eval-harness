"""Ephemeral per-run container with guaranteed teardown."""

from __future__ import annotations

import io
import shlex
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import docker
from docker.models.containers import Container
from docker.models.networks import Network

from .image import BASE_IMAGE

LABEL = "agent_eval"
RUN_LABEL = "agent_eval.run_id"
WORKSPACE = "/workspace"
TASK_DIR = "/task"  # outside the repo so injected task files never show up in the diff
BASELINE_REF = "baseline"
TIMEOUT_EXIT_CODE = 124  # coreutils `timeout`

# Keep build/test noise out of the captured diff without touching the repo's own .gitignore.
_GIT_EXCLUDES = ["__pycache__/", "*.pyc", ".pytest_cache/", ".ruff_cache/"]


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    output: str
    duration_ms: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def timed_out(self) -> bool:
        return self.exit_code == TIMEOUT_EXIT_CODE


def _tar_dir(src: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(src, arcname=".", filter=_as_root)
    return buf.getvalue()


def _tar_bytes(name: str, data: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _as_root(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    return info


class Sandbox:
    """One clean container (plus an isolated network) for one test-case run.

    Use as a context manager; the container and network are removed on exit, including on
    error. Anything leaked by a hard crash can be removed with `reap()`.
    """

    def __init__(
        self,
        client: docker.DockerClient,
        image: str = BASE_IMAGE,
        *,
        isolated: bool = True,
        mem_limit: str = "1g",
        cpus: float = 1.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self.client = client
        self.image = image
        self.isolated = isolated
        self.mem_limit = mem_limit
        self.cpus = cpus
        self.env = env or {}
        self.run_id = uuid.uuid4().hex[:12]
        self.container: Container | None = None
        self.network: Network | None = None
        self._extras: list[Container] = []  # sidecars, removed before the network

    # --- lifecycle ---------------------------------------------------------

    def __enter__(self) -> Self:
        labels = {LABEL: "1", RUN_LABEL: self.run_id}
        try:
            # `internal` networks have no route out: the only reachable peers are other
            # containers on the same network (the proxy, from M2 on).
            self.network = self.client.networks.create(
                f"agent-eval-{self.run_id}", driver="bridge", internal=self.isolated, labels=labels
            )
            self.container = self.client.containers.run(
                self.image,
                command=["sleep", "infinity"],
                detach=True,
                labels=labels,
                network=self.network.name,
                mem_limit=self.mem_limit,
                nano_cpus=int(self.cpus * 1e9),
                environment=self.env,
                working_dir=WORKSPACE,
            )
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        container, network, extras = self.container, self.network, self._extras
        self.container = self.network = None
        self._extras = []
        for extra in extras:
            _quiet(lambda extra=extra: extra.remove(force=True, v=True))
        if container is not None:
            _quiet(lambda: container.remove(force=True, v=True))
        if network is not None:
            _quiet(network.remove)

    def adopt(self, container: Container) -> None:
        """Tie a sidecar container's lifetime to this sandbox."""
        self._extras.append(container)

    @property
    def labels(self) -> dict[str, str]:
        return {LABEL: "1", RUN_LABEL: self.run_id}

    # --- operations --------------------------------------------------------

    def exec(
        self,
        cmd: list[str] | str,
        *,
        timeout: int = 60,
        workdir: str = WORKSPACE,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run a command; a string is run through `bash -lc`. Never raises on non-zero exit."""
        assert self.container is not None, "sandbox not started"
        argv = ["bash", "-lc", cmd] if isinstance(cmd, str) else list(cmd)
        argv = ["timeout", "-k", "5", str(timeout), *argv]
        start = time.monotonic()
        code, out = self.container.exec_run(argv, workdir=workdir, environment=env or {})
        return ExecResult(code, out.decode(errors="replace"), (time.monotonic() - start) * 1000)

    def copy_in(self, src: Path, dest: str) -> None:
        """Copy the contents of local directory `src` into container directory `dest`."""
        assert self.container is not None
        self._check(self.exec(["mkdir", "-p", dest]), f"mkdir {dest}")
        self.container.put_archive(dest, _tar_dir(Path(src)))

    def write_file(self, dest: str, data: str | bytes) -> None:
        assert self.container is not None
        raw = data.encode() if isinstance(data, str) else data
        parent, name = dest.rsplit("/", 1)
        self._check(self.exec(["mkdir", "-p", parent or "/"]), f"mkdir {parent}")
        self.container.put_archive(parent or "/", _tar_bytes(name, raw))

    def read_file(self, path: str) -> str:
        return self._check(self.exec(["cat", path]), f"cat {path}").output

    def seed_repo(self, repo_dir: Path) -> None:
        """Copy the task repo to /workspace and commit it as the diff baseline."""
        self.copy_in(repo_dir, WORKSPACE)
        excludes = "\\n".join(_GIT_EXCLUDES)
        script = (
            "git init -q -b main && "
            "git config user.name harness && git config user.email harness@localhost && "
            f"printf '{excludes}\\n' >> .git/info/exclude && "
            "git add -A && git commit -q --allow-empty -m baseline && "
            f"git tag {BASELINE_REF}"
        )
        self._check(self.exec(script), "seed_repo")

    def inject_task(self, prompt: str) -> None:
        self.write_file(f"{TASK_DIR}/prompt.txt", prompt)

    def diff(self) -> str:
        """Everything the agent changed since the baseline, including new and binary files."""
        self._check(self.exec("git add -A"), "git add")
        return self._check(
            self.exec(f"git diff --cached --binary --no-color {BASELINE_REF}"), "git diff"
        ).output

    def sqlite_dump(self, db_path: str) -> str:
        return self._check(
            self.exec(["sqlite3", db_path, ".dump"]), f"sqlite3 {shlex.quote(db_path)}"
        ).output

    @staticmethod
    def _check(result: ExecResult, what: str) -> ExecResult:
        if not result.ok:
            raise RuntimeError(f"{what} failed (exit {result.exit_code}): {result.output}")
        return result


def _quiet(fn) -> None:
    try:
        fn()
    except docker.errors.DockerException:
        pass


def reap(client: docker.DockerClient, run_id: str | None = None) -> int:
    """Remove leaked harness containers and networks (all of them, or one run's)."""
    selector = f"{RUN_LABEL}={run_id}" if run_id else LABEL
    removed = 0
    for c in client.containers.list(all=True, filters={"label": selector}):
        _quiet(lambda c=c: c.remove(force=True, v=True))
        removed += 1
    for n in client.networks.list(filters={"label": selector}):
        _quiet(n.remove)
        removed += 1
    return removed
