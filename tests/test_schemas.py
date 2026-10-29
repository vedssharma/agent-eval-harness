import pytest
from pydantic import TypeAdapter, ValidationError

from agent_eval.schemas import (
    FaultKind,
    FaultStep,
    LLMCallStep,
    RunMode,
    RunResult,
    StateScore,
    TaskSpec,
    ToolCallStep,
    ToolResultStep,
    Trace,
    TraceStep,
)


def make_trace() -> Trace:
    return Trace(
        steps=[
            LLMCallStep(index=0, t_ms=0, request_id="r1", input_tokens=100, output_tokens=20),
            ToolCallStep(index=1, t_ms=5, call_id="c1", name="read_file", arguments={"path": "a"}),
            ToolResultStep(index=2, t_ms=9, call_id="c1", output="x", is_error=False),
            FaultStep(index=3, t_ms=10, kind=FaultKind.HTTP_429, request_id="r2"),
            LLMCallStep(index=4, t_ms=12, request_id="r3", input_tokens=50, output_tokens=5),
        ]
    )


def test_trace_roundtrip_preserves_step_types():
    trace = make_trace()
    restored = Trace.model_validate_json(trace.model_dump_json())
    assert restored == trace
    assert [type(s).__name__ for s in restored.steps] == [type(s).__name__ for s in trace.steps]


def test_trace_aggregates():
    trace = make_trace()
    assert trace.input_tokens == 150
    assert trace.output_tokens == 25
    assert [c.name for c in trace.tool_calls()] == ["read_file"]


def test_step_discriminator_rejects_unknown_type():
    with pytest.raises(ValidationError):
        TypeAdapter(TraceStep).validate_python({"type": "nope", "index": 0, "t_ms": 0})


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        TaskSpec(id="x", title="t", prompt="p", bogus=1)


def test_run_result_passed_requires_state_and_no_error():
    base = {"task_id": "t", "iteration": 0, "mode": RunMode.REPLAY}
    assert not RunResult(**base).passed
    assert RunResult(**base, state=StateScore(exit_code=0, passed=True)).passed
    assert not RunResult(**base, state=StateScore(exit_code=1, passed=False)).passed
    assert not RunResult(**base, state=StateScore(exit_code=0, passed=True), error="boom").passed
