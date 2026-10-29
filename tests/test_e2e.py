"""M3 end-to-end: sandbox + proxy sidecar + reference agent + hidden-test scoring.

The model is a scripted fake Anthropic server; everything else (containers, networks, the
agent process, tool execution, proxy record/replay, scoring) is real.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_eval.cli import app
from agent_eval.engine import reap
from agent_eval.engine.sandbox import LABEL
from agent_eval.proxy.cassette import Cassette
from agent_eval.runner import RunConfig, run_task
from agent_eval.schemas import RunMode
from agent_eval.tasks import Task, load_task

from .fake_anthropic import ScriptedAnthropic

pytestmark = pytest.mark.docker

TASK = load_task(Path(__file__).resolve().parents[1] / "tasks" / "pager-boundary-001")
FIXED = (TASK.solution_path / "src" / "pager.py").read_text()
REAL_KEY = "sk-ant-FAKE-REAL-KEY"


def record(client, cassette: Path, *, fix: bool):
    fake = ScriptedAnthropic(FIXED, fix=fix).start()
    try:
        cfg = RunConfig(
            mode=RunMode.RECORD,
            cassette=cassette,
            upstream=fake.container_url,
            api_key=REAL_KEY,
        )
        result = run_task(client, TASK, cfg)
    finally:
        fake.stop()
    return result, fake


def replay(client, cassette: Path, task: Task = TASK):
    return run_task(client, task, RunConfig(mode=RunMode.REPLAY, cassette=cassette))


@pytest.fixture(scope="module")
def good(docker_client, tmp_path_factory):
    cassette = tmp_path_factory.mktemp("good") / "good.jsonl"
    result, fake = record(docker_client, cassette, fix=True)
    return result, fake, cassette


@pytest.fixture(scope="module")
def bad(docker_client, tmp_path_factory):
    cassette = tmp_path_factory.mktemp("bad") / "bad.jsonl"
    result, fake = record(docker_client, cassette, fix=False)
    return result, fake, cassette


@pytest.fixture(autouse=True)
def no_leaks(docker_client):
    yield
    leftovers = docker_client.containers.list(
        all=True, filters={"label": LABEL}
    ) + docker_client.networks.list(filters={"label": LABEL})
    assert leftovers == [], "run leaked containers or networks"
    reap(docker_client)


# --- RECORD ---------------------------------------------------------------------------------


def test_record_run_passes_hidden_tests(good):
    result, _, _ = good
    assert result.error is None, result.error
    assert result.agent_exit_code == 0
    assert result.state.passed and result.state.exit_code == 0
    assert result.passed


def test_diff_is_the_agents_change_only(good):
    diff = good[0].diff
    assert "+    if page < 1 or per_page < 1:" in diff
    assert "hidden_tests" not in diff  # scored after the diff was taken
    assert "__pycache__" not in diff and "agent_log" not in diff


def test_trace_has_llm_calls_tool_calls_and_tokens(good):
    trace = good[0].trace
    assert [c.name for c in trace.tool_calls()] == ["read_file", "write_file", "bash"]
    llm = [s for s in trace.steps if s.type == "llm_call"]
    assert len(llm) == 4 and all(s.status == 200 and s.model == "claude-scripted" for s in llm)
    assert trace.input_tokens == 100 * (1 + 3 + 5 + 7)
    assert trace.output_tokens == 10 * (1 + 3 + 5 + 7)
    results = [s for s in trace.steps if s.type == "tool_result"]
    assert len(results) == 3 and not any(r.is_error for r in results)
    assert [s.index for s in trace.steps] == list(range(len(trace.steps)))
    assert [s.t_ms for s in trace.steps] == sorted(s.t_ms for s in trace.steps)


def test_agent_only_ever_holds_a_dummy_key(good):
    _, fake, cassette = good
    assert set(fake.seen_keys) == {REAL_KEY}  # the proxy injected the real key upstream
    assert REAL_KEY not in cassette.read_text()  # and it never reached the cassette


def test_cassette_has_one_entry_per_llm_call(good):
    assert len(Cassette.load(good[2])) == 4


# --- REPLAY ---------------------------------------------------------------------------------


def test_replay_reproduces_the_recorded_run_offline(docker_client, good):
    recorded, fake, cassette = good
    assert fake._thread and not fake._thread.is_alive()  # the "API" is gone
    calls_before = fake.calls

    replayed = replay(docker_client, cassette)

    assert fake.calls == calls_before
    assert replayed.error is None, replayed.error
    assert replayed.cassette_misses == 0
    assert replayed.passed == recorded.passed
    assert replayed.diff == recorded.diff
    assert [(c.name, c.arguments) for c in replayed.trace.tool_calls()] == [
        (c.name, c.arguments) for c in recorded.trace.tool_calls()
    ]
    assert replayed.trace.input_tokens == recorded.trace.input_tokens


def test_replay_is_repeatable(docker_client, good):
    a = replay(docker_client, good[2])
    b = replay(docker_client, good[2])
    assert a.diff == b.diff and a.passed and b.passed


def test_replay_with_a_changed_prompt_is_flagged_as_a_miss(docker_client, good):
    other = Task(
        spec=TASK.spec.model_copy(update={"prompt": TASK.spec.prompt + " Be quick."}),
        root=TASK.root,
    )
    result = replay(docker_client, good[2], other)
    assert result.cassette_misses >= 1
    assert result.error and "no matching recording" in result.error
    assert not result.passed


# --- scoring is real: a bad patch must fail ------------------------------------------------


def test_unfixed_code_fails_the_hidden_tests(bad):
    result, _, _ = bad
    assert result.error is None
    assert result.state.exit_code != 0 and not result.passed
    assert "did not fix it" in result.diff


def test_bad_run_replays_as_a_bad_run(docker_client, bad):
    result = replay(docker_client, bad[2])
    assert not result.passed and result.state.exit_code != 0 and result.error is None


# --- CLI ------------------------------------------------------------------------------------


def test_cli_replay_exit_codes(docker_client, good, bad, tmp_path):
    runner = CliRunner()
    base = ["run", TASK.spec.id, "--mode", "replay", "--tasks-dir", str(TASK.root.parent)]
    base += ["--out", str(tmp_path)]

    ok = runner.invoke(app, [*base, "--cassette", str(good[2])])
    assert ok.exit_code == 0, ok.output
    assert "PASS" in ok.output and "misses:      0" in ok.output

    fail = runner.invoke(app, [*base, "--cassette", str(bad[2])])
    assert fail.exit_code == 1 and "FAIL" in fail.output

    saved = list((tmp_path / TASK.spec.id).glob("replay-*/result.json"))
    assert len(saved) == 2
