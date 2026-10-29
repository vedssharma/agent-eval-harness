# Agent Eval Harness: Project Plan

Goal: a reproducible harness that runs a coding agent against tasks in ephemeral containers, controls its network via a VCR/chaos proxy, scores it on three layers, and lets you diff trajectories.

## Key design decisions

1. **The proxy is the trace source.** The agent's LLM calls go through our proxy, so the request/response bodies already contain every tool call, token count, and latency. We rebuild the trajectory from proxy traffic, which keeps the harness agent-agnostic. An optional agent-side JSONL log adds tool *results* and errors.
2. **Hermetic networking.** Put the task container on a Docker `--internal` network whose only reachable peer is the proxy. All egress is then intercepted by construction. The agent gets `ANTHROPIC_BASE_URL=http://proxy:PORT` (reverse-proxy mode). A generic forward-proxy mode for non-LLM HTTP is deferred; HTTPS CONNECT would need MITM certs.
3. **Hidden tests.** The agent never sees the grading tests. We copy them into the container *after* the agent finishes, run them, and check the exit code. This prevents the agent from editing the tests to pass.
4. **Chaos layers on top of REPLAY.** Faults are seeded and declarative (e.g. `{at_request: 3, fault: "429", retry_after: 1}` or `{p: 0.1, fault: "jitter", ms: 2000}`). Upstream stays deterministic, so any change in behaviour is attributable to the fault.
5. **Stats caveat.** In pure REPLAY mode the k=3 runs are identical, so pass@k and pass^k only carry information in RECORD/live mode or CHAOS mode (different seed per iteration). The runner will say so.
   - pass@k = `1 - C(n-c, k) / C(n, k)`
   - pass^k = `C(c, k) / C(n, k)`
   - With n=k=3 these reduce to "any pass" and "all pass".
6. **Stack:** Python 3.12 (image) / 3.12+ (host), `uv`, `pydantic` v2 for every schema, `docker` SDK, `aiohttp` for the proxy, `typer` for the CLI, `pytest`. The visualizer starts as a self-contained static HTML report, with a FastAPI dashboard only if needed.

## Layout

```
src/agent_eval/
  schemas.py        # TaskSpec, Trace, ToolCall, RunResult, Cassette (pydantic)
  engine/           # container lifecycle, task injection, artifact capture
  proxy/            # server.py, cassette.py, chaos.py, sse.py
  agents/           # adapters: reference tool-loop agent, generic command adapter
  scoring/          # state.py, trajectory.py, stats.py
  runner.py         # orchestrates task x k iterations
  cli.py            # run | replay | score | diff | report
  viz/              # trace diff + HTML report
tasks/<task-id>/    # task.yaml, repo/ (seed), hidden_tests/, reference_path.json, cassettes/
images/             # Dockerfile: python + sqlite + git + pytest
tests/
```

## Milestones

Each milestone ends in something runnable. We build a thin vertical slice (M0-M3) before adding breadth.

### M0: Scaffold and schemas
- `uv` project, ruff, pytest, CI.
- Pydantic models for `TaskSpec`, `Trace` (ordered steps: llm_call, tool_call, tool_result, fault), `RunResult`.
- One toy task: a small Python repo with a boundary-condition bug, a visible test, and a hidden test.
- **Done when:** the task loads and validates, and schema round-trip tests pass.

### M1: Ephemeral execution engine
- Build the base image (Python, SQLite, git, pytest).
- `Sandbox` async context manager: create the internal network, start the container, copy in the seed repo, `git init` + baseline commit.
- Run commands with timeout, inject the task prompt, then capture `git diff` against baseline and (optionally) a SQLite dump.
- Teardown always runs (try/finally plus a label-based reaper for crashed runs).
- **Done when:** a scripted "fake agent" applies a known patch, the diff is captured, and no containers or networks are left behind.

### M2: Proxy, RECORD and REPLAY
- aiohttp reverse proxy with streaming (SSE) passthrough.
- Cassette format: JSONL, one entry per exchange, with request key, status, headers, body or chunks, and timings.
- Request key = hash of method + path + canonicalised body, with volatile fields dropped and auth headers redacted. A per-key occurrence counter handles repeated identical calls. Strict and by-order match policies.
- REPLAY returns a clear error on a cassette miss (never silently hits the network).
- **Done when:** record against the real API, replay with the network off, and get byte-identical responses.

### M3: First end-to-end slice and State scorer
- Reference agent adapter: a minimal tool-use loop (bash, read/write file, run tests) running inside the container, talking to the proxy.
- State scorer: copy in hidden tests, run pytest, record the exit code and pass/fail counts.
- `agent-eval run task-001 --mode record|replay`.
- **Done when:** one task runs record -> replay and the replayed run yields the same diff and the same pass/fail.

### M4: Trace extraction and Trajectory scorer
- Parse proxy traffic into `Trace` (tool_use blocks, tool results, token usage, per-call latency).
- Reference path format: ordered tool calls with argument matchers (exact, subset, regex, normalised path).
- Align actual against reference with sequence alignment (Needleman-Wunsch or LCS). Metrics: tool-selection accuracy, argument conformity, redundant-loop count (repeated identical calls and A-B-A-B cycles), and extra/missing steps.
- **Done when:** hand-built traces with known deviations produce the expected scores.

### M5: Statistical profiler
- The runner executes k=3 iterations per task in parallel (bounded concurrency), each in a fresh sandbox.
- Report pass@k, pass^k, total input and output tokens, p50/p95 latency, and wall time. Output as JSON and a table.
- **Done when:** a suite run produces a stable `results.json` and a summary table.

### M6: Chaos mode
- Fault types: connection drop, +2 s jitter, injected 500, injected 429 with `Retry-After`. All seeded and reproducible.
- Fault schedule in task or run config. Each injected fault is recorded in the trace.
- Recovery metrics: retries after fault, time-to-recovery, whether the final result still passed, and whether the agent looped or gave up.
- **Done when:** the same cassette replayed under different seeds shows different recovery trajectories, and the same seed reproduces exactly.

### M7: Trajectory Diff CLI and visualizer
- `agent-eval diff runA runB` and `agent-eval diff run --ref`: aligned side-by-side step table in the terminal (rich), with deviations, errored tool calls, and faults highlighted.
- `agent-eval report run`: a self-contained HTML page with a trace graph, an aligned reference-vs-actual view, a syntax-highlighted environment `git diff`, and a token/latency timeline.
- **Done when:** a deliberately bad run shows its divergence point, its failing call, and its patch at a glance.

### M8: Hardening and regression workflow
- Podman support (the docker SDK against the Podman socket), checked on Linux.
- A task suite of 5-10 tasks (refactor, SQLite migration, git-conflict cases).
- Baseline comparison: `agent-eval compare baseline.json new.json` fails CI on a regression in pass^k, trajectory score, or token budget.
- Cassette hygiene: secret scrubbing checks, cassette-staleness warning when the prompt or tools change.

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Cassettes break whenever the prompt changes | Canonical keys, by-order fallback, and a staleness report. |
| SSE replay timing differs from live | Store chunk timings and replay them with an optional speed factor. |
| macOS host-to-container networking differs from Linux | Run the proxy on the host with an internal network plus a gateway alias, or run the proxy in a sidecar container. Decide in M1/M2. |
| Leaked containers | Labels plus a reaper command and a test that asserts a clean teardown. |
| Agent edits the tests | Hidden tests are injected post-run (decision 3). |

## Open questions
1. Which agent is under test first: the reference tool-loop in this repo, or an existing agent CLI (e.g. Claude Code) run through a command adapter?
2. LLM API shape for the proxy: Anthropic Messages only first, or OpenAI-compatible too?
3. Visualizer: static HTML report (recommended to start) or a live FastAPI dashboard?
4. Do you want this wrapped as a Prime Lab / `verifiers` environment later? The workspace AGENTS.md covers that framework; this plan assumes a standalone package.
