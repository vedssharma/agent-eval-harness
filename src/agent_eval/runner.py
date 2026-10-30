"""Run one task once: sandbox + proxy sidecar + agent + state scoring + trace."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import docker

from .agents import REFERENCE_AGENT
from .engine import ProxySidecar, Sandbox, ensure_image
from .engine.sandbox import TASK_DIR
from .proxy import MatchPolicy, ProxyMode
from .proxy.cassette import Cassette
from .schemas import RunMode, RunResult
from .scoring import score_state, score_trajectory
from .tasks import Task
from .trace import build_trace

AGENT_PATH = "/opt/agent/agent.py"
DEFAULT_UPSTREAM = "https://api.anthropic.com"
AGENT_EXIT_FATAL = 2
OUTPUT_TAIL = 2000


@dataclass
class RunConfig:
    mode: RunMode
    cassette: Path | None = None  # default: <task>/cassettes/default.jsonl
    upstream: str = DEFAULT_UPSTREAM
    api_key: str | None = None  # RECORD only; held by the proxy sidecar, never the agent
    model: str | None = None
    max_turns: int | None = None
    match: MatchPolicy = MatchPolicy.STRICT

    def cassette_for(self, task: Task) -> Path:
        return self.cassette or task.root / "cassettes" / "default.jsonl"


def run_task(
    client: docker.DockerClient, task: Task, cfg: RunConfig, iteration: int = 0
) -> RunResult:
    """Never raises for run-level failures: they come back as `RunResult.error`."""
    if cfg.mode not in (RunMode.RECORD, RunMode.REPLAY):
        raise ValueError(f"mode {cfg.mode.value} is not supported yet")
    result = RunResult(task_id=task.spec.id, iteration=iteration, mode=cfg.mode)
    start = time.monotonic()
    try:
        _run(client, task, cfg, result)
    except Exception as exc:  # noqa: BLE001 - a broken run is data, not a crash
        result.error = result.error or f"{type(exc).__name__}: {exc}"
    result.wall_ms = (time.monotonic() - start) * 1000
    return result


def _run(client: docker.DockerClient, task: Task, cfg: RunConfig, result: RunResult) -> None:
    cassette_path = cfg.cassette_for(task)
    proxy_mode = ProxyMode(cfg.mode.value)
    if proxy_mode is ProxyMode.RECORD and cfg.upstream == DEFAULT_UPSTREAM and not cfg.api_key:
        raise RuntimeError("RECORD against the real API needs an API key (ANTHROPIC_API_KEY)")

    ensure_image(client, task.spec.image)
    with Sandbox(client, task.spec.image) as sb:
        sb.seed_repo(task.repo_path)
        sb.inject_task(task.spec.prompt)
        sb.write_file(AGENT_PATH, REFERENCE_AGENT.read_text())

        sidecar = ProxySidecar(
            sb,
            proxy_mode,
            cassette=cassette_path if proxy_mode is ProxyMode.REPLAY else None,
            upstream=cfg.upstream if proxy_mode is ProxyMode.RECORD else None,
            upstream_api_key=cfg.api_key,
            match=cfg.match,
        )
        sidecar.start()

        env = {"ANTHROPIC_BASE_URL": sidecar.base_url, "ANTHROPIC_API_KEY": "dummy-not-a-real-key"}
        if cfg.model:
            env["AGENT_MODEL"] = cfg.model
        if cfg.max_turns:
            env["AGENT_MAX_TURNS"] = str(cfg.max_turns)
        agent = sb.exec(["python", AGENT_PATH], timeout=task.spec.timeout_s, env=env)
        result.agent_exit_code = agent.exit_code

        # Order matters: diff first, so the hidden tests never appear in the patch.
        result.diff = sb.diff()
        result.state = score_state(sb, task)

        exchanges = sidecar.fetch_exchanges()
        cassette_bytes = sidecar.fetch_cassette()
        agent_log = _read_agent_log(sb)

    if proxy_mode is ProxyMode.RECORD:
        cassette_path.parent.mkdir(parents=True, exist_ok=True)
        cassette_path.write_bytes(cassette_bytes)
        cassette = Cassette.load(cassette_path)
    else:
        cassette = Cassette.load(cassette_path)

    result.trace = build_trace(agent_log, exchanges, cassette)
    reference = task.reference_path()
    if reference:
        result.trajectory = score_trajectory(result.trace, reference)
    result.cassette_misses = sum(1 for e in exchanges if e["miss"])
    if result.cassette_misses:
        result.error = (
            f"{result.cassette_misses} request(s) had no matching recording; "
            "re-record the cassette or check the agent's request determinism"
        )
    elif agent.exit_code == AGENT_EXIT_FATAL:
        result.error = f"agent failed: {agent.output[-OUTPUT_TAIL:].strip()}"
    elif agent.timed_out:
        result.error = f"agent timed out after {task.spec.timeout_s}s"


def _read_agent_log(sb: Sandbox) -> list[dict]:
    res = sb.exec(["cat", f"{TASK_DIR}/agent_log.jsonl"])
    if not res.ok:
        return []
    return [json.loads(ln) for ln in res.output.splitlines() if ln.strip()]


def save_run(result: RunResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(result.model_dump_json(indent=2))
    (out_dir / "diff.patch").write_text(result.diff)
    return out_dir
