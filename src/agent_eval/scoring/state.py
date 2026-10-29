"""State verification: run the task's tests (including hidden ones) inside the sandbox."""

from __future__ import annotations

from ..engine import Sandbox
from ..engine.sandbox import WORKSPACE
from ..schemas import StateScore
from ..tasks import Task

OUTPUT_TAIL = 4000


def score_state(sandbox: Sandbox, task: Task) -> StateScore:
    """Copy the hidden tests in *after* the agent has finished, then run the test command.

    Call `sandbox.diff()` first: the hidden tests must not appear in the captured patch.
    Pass means exit code 0 (so pytest's "no tests collected", exit 5, is a failure).
    """
    sandbox.copy_in(task.hidden_tests_path, f"{WORKSPACE}/{task.spec.hidden_tests_dir}")
    result = sandbox.exec(task.spec.test_command, timeout=task.spec.timeout_s)
    return StateScore(
        exit_code=result.exit_code,
        passed=result.exit_code == 0,
        output=result.output[-OUTPUT_TAIL:],
    )
