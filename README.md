# agent-eval

Reproducible agent test harness. See [PLAN.md](PLAN.md) for the architecture and milestones.

```bash
uv sync
uv run pytest
uv run ruff check .
```

Status: M0 (schemas, task format, toy task) and M1 (ephemeral sandbox engine) complete.
Docker/Podman tests are skipped automatically when no container runtime is reachable.
