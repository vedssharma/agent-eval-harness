"""M1 integration tests: need a running Docker/Podman daemon (skipped otherwise)."""

from pathlib import Path

import pytest

from agent_eval.engine import Sandbox, reap
from agent_eval.engine.sandbox import LABEL
from agent_eval.tasks import load_task

pytestmark = pytest.mark.docker

TASK = load_task(Path(__file__).resolve().parents[1] / "tasks" / "pager-boundary-001")


def leaked(client) -> list:
    return client.containers.list(all=True, filters={"label": LABEL}) + client.networks.list(
        filters={"label": LABEL}
    )


@pytest.fixture(autouse=True)
def no_leaks(docker_client):
    reap(docker_client)
    yield
    assert leaked(docker_client) == [], "test leaked containers or networks"


def test_teardown_after_normal_exit(docker_client):
    with Sandbox(docker_client) as sb:
        assert sb.exec("echo hi").output.strip() == "hi"
        assert len(leaked(docker_client)) == 2  # container + network
    assert leaked(docker_client) == []


def test_teardown_after_exception(docker_client):
    with pytest.raises(ZeroDivisionError), Sandbox(docker_client):
        1 / 0  # noqa: B018


def test_reap_removes_leaked_resources(docker_client):
    sb = Sandbox(docker_client).__enter__()  # deliberately never closed
    assert reap(docker_client, run_id=sb.run_id) == 2
    assert leaked(docker_client) == []


def test_exec_timeout_and_nonzero_exit(docker_client):
    with Sandbox(docker_client) as sb:
        slow = sb.exec("sleep 30", timeout=1)
        assert slow.timed_out and slow.duration_ms < 10_000
        assert sb.exec("exit 3").exit_code == 3


def test_isolated_network_has_no_egress(docker_client):
    probe = "import socket; socket.create_connection(('1.1.1.1', 80), timeout=3)"
    with Sandbox(docker_client, isolated=True) as sb:
        assert not sb.exec(["python", "-c", probe], timeout=15).ok


def test_fresh_sandboxes_do_not_share_state(docker_client):
    with Sandbox(docker_client) as a:
        a.write_file("/workspace/marker", "x")
    with Sandbox(docker_client) as b:
        assert not b.exec("test -e /workspace/marker").ok


def test_fake_agent_diff_capture(docker_client):
    with Sandbox(docker_client) as sb:
        sb.seed_repo(TASK.repo_path)
        sb.inject_task(TASK.spec.prompt)
        assert sb.read_file("/task/prompt.txt") == TASK.spec.prompt

        # A run of the visible tests leaves caches behind; they must not pollute the diff.
        assert sb.exec(TASK.spec.test_command).ok
        assert sb.diff() == ""

        # "Fake agent": apply the known-good patch.
        sb.copy_in(TASK.solution_path, "/workspace")
        diff = sb.diff()

    assert "src/pager.py" in diff
    assert "+    if page < 1 or per_page < 1:" in diff
    assert "/task/" not in diff and "prompt.txt" not in diff
    assert "__pycache__" not in diff and ".pytest_cache" not in diff


def test_hidden_tests_distinguish_seed_from_solution(docker_client):
    with Sandbox(docker_client) as sb:
        sb.seed_repo(TASK.repo_path)
        sb.copy_in(TASK.hidden_tests_path, "/workspace/hidden_tests")
        assert not sb.exec("python -m pytest -q hidden_tests").ok

    with Sandbox(docker_client) as sb:
        sb.seed_repo(TASK.repo_path)
        sb.copy_in(TASK.solution_path, "/workspace")
        sb.copy_in(TASK.hidden_tests_path, "/workspace/hidden_tests")
        result = sb.exec("python -m pytest -q hidden_tests")
        assert result.ok, result.output


def test_sqlite_dump(docker_client):
    with Sandbox(docker_client) as sb:
        sb.exec("sqlite3 /tmp/t.db 'create table t(x); insert into t values (7);'")
        assert "INSERT INTO t VALUES(7);" in sb.sqlite_dump("/tmp/t.db")
