"""M4: trajectory scoring on hand-built traces with known deviations."""

import random

import pytest
from pydantic import ValidationError

from agent_eval.schemas import (
    AlignKind,
    MatchMode,
    ReferenceStep,
    ToolCallStep,
    ToolResultStep,
    Trace,
)
from agent_eval.scoring import TrajectoryWeights, score_trajectory
from agent_eval.scoring.trajectory import align, args_conform, count_loops

READ = ("read_file", {"path": "src/pager.py"})
WRITE = ("write_file", {"path": "src/pager.py", "content": "v1"})
TEST = ("bash", {"command": "python -m pytest -q"})

REFERENCE = [
    ReferenceStep(tool="read_file", args={"path": "src/pager.py"}),
    ReferenceStep(tool="write_file", args={"path": "src/pager.py"}),
    ReferenceStep(tool="bash", args={"command": "pytest"}, match=MatchMode.REGEX),
]


def trace(*calls) -> Trace:
    """calls: (name, args) or (name, args, output)."""
    steps = []
    for i, call in enumerate(calls):
        name, args, *rest = call
        output = rest[0] if rest else f"out-{name}-{i}"  # distinct unless told otherwise
        steps.append(ToolCallStep(index=len(steps), t_ms=i, call_id=f"c{i}", name=name, arguments=args))
        steps.append(ToolResultStep(index=len(steps), t_ms=i, call_id=f"c{i}", output=output))
    return Trace(steps=steps)


def kinds(score) -> list[tuple[str, AlignKind]]:
    return [(p.tool, p.kind) for p in score.alignment]


# --- the reference path itself ------------------------------------------------------------


def test_perfect_run_scores_one():
    s = score_trajectory(trace(READ, WRITE, TEST), REFERENCE)
    assert s.score == pytest.approx(1.0)
    assert (s.tool_precision, s.tool_recall, s.arg_conformity, s.efficiency) == (1, 1, 1, 1)
    assert (s.extra_calls, s.missing_steps, s.redundant_calls, s.loops) == (0, 0, 0, 0)
    assert all(p.kind is AlignKind.MATCHED for p in s.alignment)


def test_scoring_is_deterministic():
    t = trace(READ, READ, WRITE, TEST)
    assert score_trajectory(t, REFERENCE) == score_trajectory(t, REFERENCE)


# --- tool selection ---------------------------------------------------------------------


def test_missing_step_lowers_recall_only():
    s = score_trajectory(trace(READ, TEST), REFERENCE)
    assert s.missing_steps == 1 and s.extra_calls == 0
    assert s.tool_precision == 1 and s.tool_recall == pytest.approx(2 / 3)
    assert s.tool_selection == pytest.approx(0.8)
    assert ("write_file", AlignKind.MISSING) in kinds(s)


def test_wrong_tool_is_one_missing_and_one_extra():
    cat = ("bash", {"command": "cat src/pager.py"})
    s = score_trajectory(trace(cat, WRITE, TEST), REFERENCE)
    assert (s.missing_steps, s.extra_calls) == (1, 1)
    assert ("read_file", AlignKind.MISSING) in kinds(s)
    assert s.tool_precision == pytest.approx(2 / 3) and s.tool_recall == pytest.approx(2 / 3)


def test_extra_exploration_lowers_precision():
    ls = ("bash", {"command": "ls"})
    s = score_trajectory(trace(ls, READ, WRITE, TEST), REFERENCE)
    assert s.extra_calls == 1 and s.missing_steps == 0
    assert s.tool_precision == pytest.approx(0.75) and s.tool_recall == 1


def test_out_of_order_calls_are_not_credited():
    s = score_trajectory(trace(TEST, WRITE, READ), REFERENCE)
    paired = [p for p in s.alignment if p.kind in (AlignKind.MATCHED, AlignKind.ARG_MISMATCH)]
    assert len(paired) == 1  # only a monotone subsequence can align
    assert (s.extra_calls, s.missing_steps) == (2, 2)


# --- argument conformity ----------------------------------------------------------------


def test_wrong_arguments_keep_tool_credit_but_lose_conformity():
    other = ("write_file", {"path": "src/other.py", "content": "x"})
    s = score_trajectory(trace(READ, other, TEST), REFERENCE)
    assert s.tool_selection == 1  # right tools, right order
    assert s.arg_conformity == pytest.approx(2 / 3)
    bad = next(p for p in s.alignment if p.kind is AlignKind.ARG_MISMATCH)
    assert bad.tool == "write_file" and "other.py" in bad.detail


def test_matcher_modes():
    ref = lambda mode, **args: ReferenceStep(tool="t", args=args, match=mode)
    assert args_conform(ref(MatchMode.EXACT, a=1), {"a": 1})
    assert not args_conform(ref(MatchMode.EXACT, a=1), {"a": 1, "b": 2})
    assert args_conform(ref(MatchMode.SUBSET, a=1), {"a": 1, "b": 2})
    assert not args_conform(ref(MatchMode.SUBSET, a=1), {"b": 2})
    assert args_conform(ref(MatchMode.REGEX, cmd=r"pytest\b"), {"cmd": "python -m pytest -q"})
    assert not args_conform(ref(MatchMode.REGEX, cmd=r"pytest\b"), {"cmd": "ls"})
    assert args_conform(ref(MatchMode.ANY, a=1), {})


@pytest.mark.parametrize("actual", ["src/pager.py", "./src/pager.py", "/workspace/src/pager.py"])
def test_paths_are_normalised(actual):
    step = ReferenceStep(tool="read_file", args={"path": "src/pager.py"}, match=MatchMode.EXACT)
    assert args_conform(step, {"path": actual})


def test_invalid_regex_in_a_reference_is_rejected_at_load_time():
    with pytest.raises(ValidationError, match="not a valid regex"):
        ReferenceStep(tool="bash", args={"command": "(unclosed"}, match=MatchMode.REGEX)


# --- redundancy and loops ----------------------------------------------------------------


def test_repeated_identical_call_with_same_output_is_redundant():
    s = score_trajectory(trace(READ, READ, WRITE, TEST), REFERENCE)
    # outputs are distinct per call index in the helper, so force identical ones:
    same = trace((*READ, "same"), (*READ, "same"), WRITE, TEST)
    r = score_trajectory(same, REFERENCE)
    assert s.redundant_calls == 0  # different output -> the second read learned something
    assert r.redundant_calls == 1 and r.loops == 0 and r.extra_calls == 1
    assert r.efficiency == pytest.approx(0.75)


def test_rerunning_tests_after_an_edit_is_not_redundant():
    t = trace(
        READ,
        ("write_file", {"path": "src/pager.py", "content": "v1"}),
        (*TEST, "1 failed"),
        ("write_file", {"path": "src/pager.py", "content": "v2"}),
        (*TEST, "2 passed"),
    )
    s = score_trajectory(t, REFERENCE)
    assert s.redundant_calls == 0 and s.loops == 0
    assert s.efficiency == 1.0  # a legitimate fix-and-retest costs precision, not efficiency
    assert s.extra_calls == 2


def test_alternating_identical_calls_are_a_loop():
    t = trace(READ, (*WRITE, "ok"), (*TEST, "fail"), (*WRITE, "ok"), (*TEST, "fail"))
    s = score_trajectory(t, REFERENCE)
    assert s.loops == 1
    assert s.redundant_calls == 2
    assert s.extra_calls == 2
    assert s.efficiency == pytest.approx(1 - 3 / 5)


def test_count_loops_variants():
    def calls(*names):
        return [
            ToolCallStep(index=i, t_ms=i, call_id=str(i), name=n, arguments={})
            for i, n in enumerate(names)
        ]

    assert count_loops(calls("a", "b", "a", "b")) == 1
    assert count_loops(calls("a", "b", "c", "a", "b", "c")) == 1
    assert count_loops(calls("a", "b", "a", "b", "a", "b")) == 1  # one run, not three
    assert count_loops(calls("a", "b", "a", "c")) == 0
    assert count_loops(calls("a", "a", "a", "a")) == 0  # plain repeats are 'redundant', not loops
    assert count_loops(calls("x", "a", "b", "a", "b", "y", "c", "d", "c", "d")) == 2
    assert count_loops([]) == 0


# --- optional steps, edge cases, weights ---------------------------------------------------


def test_optional_step_can_be_skipped_without_penalty():
    ref = [ReferenceStep(tool="read_file", args={"path": "src/pager.py"}, optional=True), *REFERENCE[1:]]
    skipped = score_trajectory(trace(WRITE, TEST), ref)
    assert skipped.score == pytest.approx(1.0) and skipped.missing_steps == 0
    assert kinds(skipped)[0] == ("read_file", AlignKind.SKIPPED_OPTIONAL)

    done = score_trajectory(trace(READ, WRITE, TEST), ref)
    assert done.score == pytest.approx(1.0) and done.tool_recall == 1


def test_empty_trace_scores_zero_without_dividing_by_zero():
    s = score_trajectory(Trace(), REFERENCE)
    assert s.score == 0 and s.missing_steps == 3 and s.n_calls == 0
    assert s.tool_precision == s.tool_recall == s.efficiency == 0


def test_weights_change_the_blend_but_not_the_components():
    t = trace(READ, ("write_file", {"path": "nope.py", "content": "x"}), TEST)
    default = score_trajectory(t, REFERENCE)
    only_args = score_trajectory(t, REFERENCE, TrajectoryWeights(selection=0, arguments=1, efficiency=0))
    assert only_args.score == pytest.approx(default.arg_conformity)
    assert only_args.tool_selection == default.tool_selection


# --- invariants ---------------------------------------------------------------------------


def test_alignment_is_a_partition_and_scores_stay_in_range():
    rng = random.Random(0)
    tools = ["read_file", "write_file", "bash"]
    paths = ["src/pager.py", "src/other.py"]
    for _ in range(300):
        calls = [
            ToolCallStep(
                index=i,
                t_ms=i,
                call_id=str(i),
                name=rng.choice(tools),
                arguments={"path": rng.choice(paths), "command": rng.choice(["pytest", "ls"])},
            )
            for i in range(rng.randint(0, 9))
        ]
        t = Trace(steps=calls)
        s = score_trajectory(t, REFERENCE)
        assert 0 <= s.score <= 1 and 0 <= s.efficiency <= 1
        call_idx = sorted(p.call_index for p in s.alignment if p.call_index is not None)
        ref_idx = sorted(p.ref_index for p in s.alignment if p.ref_index is not None)
        assert call_idx == list(range(len(calls)))
        assert ref_idx == list(range(len(REFERENCE)))
        # alignment preserves order on both sides
        pairs = [(p.ref_index, p.call_index) for p in align(REFERENCE, calls) if p.ref_index is not None and p.call_index is not None]
        assert pairs == sorted(pairs)
