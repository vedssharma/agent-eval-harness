# agent-eval

Reproducible agent test harness. See [PLAN.md](PLAN.md) for the architecture and milestones.

```bash
uv sync
uv run pytest
uv run ruff check .
```

Status: M0-M3 complete (schemas, sandbox engine, VCR proxy, first end-to-end run).

```bash
export ANTHROPIC_API_KEY=...                     # held by the proxy sidecar, never the agent
uv run agent-eval run pager-boundary-001 --mode record   # real API -> tasks/<id>/cassettes/default.jsonl
uv run agent-eval run pager-boundary-001 --mode replay   # offline, deterministic
```

Recording against the real API is not yet exercised; the test suite uses a scripted fake
Anthropic server. Docker/Podman tests are skipped automatically when no container runtime is reachable.
