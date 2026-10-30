"""`agent-eval` command line."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Annotated

import typer

from .engine import get_client
from .proxy import MatchPolicy
from .runner import DEFAULT_UPSTREAM, RunConfig, run_task, save_run
from .schemas import RunMode
from .tasks import load_task

app = typer.Typer(no_args_is_help=True, help="Reproducible agent evaluation harness.")


@app.command()
def run(
    task_id: str,
    mode: Annotated[RunMode, typer.Option(help="record hits the real API; replay is offline")],
    tasks_dir: Annotated[Path, typer.Option(help="Directory containing task folders")] = Path(
        "tasks"
    ),
    cassette: Annotated[Path | None, typer.Option(help="Cassette file (default per task)")] = None,
    upstream: Annotated[str, typer.Option(help="API base URL used when recording")] = (
        DEFAULT_UPSTREAM
    ),
    model: Annotated[str | None, typer.Option(help="Model id passed to the agent")] = None,
    max_turns: Annotated[int | None, typer.Option(help="Agent turn limit")] = None,
    match: Annotated[MatchPolicy, typer.Option(help="Replay matching policy")] = (
        MatchPolicy.STRICT
    ),
    out: Annotated[Path, typer.Option(help="Where run artifacts are written")] = Path("runs"),
) -> None:
    """Run one task once and print the result."""
    task = load_task(tasks_dir / task_id)
    cfg = RunConfig(
        mode=mode,
        cassette=cassette,
        upstream=upstream,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model=model,
        max_turns=max_turns,
        match=match,
    )
    result = run_task(get_client(), task, cfg)
    run_dir = save_run(result, out / task_id / f"{mode.value}-{time.strftime('%Y%m%d-%H%M%S')}")

    typer.echo(f"task:        {result.task_id}  ({mode.value})")
    typer.echo(f"result:      {'PASS' if result.passed else 'FAIL'}")
    if result.state:
        typer.echo(f"tests:       exit code {result.state.exit_code}")
    typer.echo(f"agent exit:  {result.agent_exit_code}")
    typer.echo(f"tool calls:  {len(result.trace.tool_calls())}")
    if result.trajectory:
        t = result.trajectory
        typer.echo(
            f"trajectory:  {t.score:.2f}  (selection {t.tool_selection:.2f}, "
            f"args {t.arg_conformity:.2f}, efficiency {t.efficiency:.2f}; "
            f"{t.extra_calls} extra, {t.missing_steps} missing, "
            f"{t.redundant_calls} redundant, {t.loops} loops)"
        )
    typer.echo(f"tokens:      {result.trace.input_tokens} in / {result.trace.output_tokens} out")
    typer.echo(f"wall time:   {result.wall_ms / 1000:.1f}s")
    typer.echo(f"misses:      {result.cassette_misses}")
    if result.error:
        typer.echo(f"error:       {result.error}", err=True)
    typer.echo(f"artifacts:   {run_dir}")
    raise typer.Exit(0 if result.passed else 1)


@app.command()
def reap() -> None:
    """Remove leaked harness containers and networks."""
    from .engine import reap as _reap

    typer.echo(f"removed {_reap(get_client())} resource(s)")
