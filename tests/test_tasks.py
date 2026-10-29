"""Validate the fixture tasks themselves: seed fails hidden tests, reference solution passes."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_eval.tasks import load_all, load_task

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"
TASKS = load_all(TASKS_DIR)


def run_pytest(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def stage(task, tmp_path: Path, with_solution: bool) -> Path:
    work = tmp_path / "workspace"
    shutil.copytree(task.repo_path, work)
    if with_solution:
        shutil.copytree(task.solution_path, work, dirs_exist_ok=True)
    shutil.copytree(task.hidden_tests_path, work / "hidden_tests")
    return work


def test_at_least_one_task():
    assert TASKS


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.spec.id)
def test_seed_passes_visible_but_fails_hidden(task, tmp_path):
    work = stage(task, tmp_path, with_solution=False)
    assert run_pytest(work, "tests").returncode == 0
    assert run_pytest(work, "hidden_tests").returncode != 0


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.spec.id)
def test_reference_solution_passes_everything(task, tmp_path):
    work = stage(task, tmp_path, with_solution=True)
    result = run_pytest(work, "tests", "hidden_tests")
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.spec.id)
def test_reference_path_loads(task):
    assert task.reference_path()


def test_task_id_must_match_directory(tmp_path):
    bad = tmp_path / "dir-name"
    bad.mkdir()
    (bad / "task.yaml").write_text("id: other\ntitle: t\nprompt: p\n")
    with pytest.raises(ValueError, match="must match"):
        load_task(bad)
