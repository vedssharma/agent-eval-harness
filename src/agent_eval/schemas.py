"""Core data models shared by the engine, proxy, scorers and visualizer."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Task definition -------------------------------------------------------


class MatchMode(StrEnum):
    EXACT = "exact"  # actual args must equal expected
    SUBSET = "subset"  # expected keys must match; extra actual keys allowed
    REGEX = "regex"  # each expected value is a regex matched against str(actual value)
    ANY = "any"  # arguments are not checked


class ReferenceStep(Strict):
    """One step of the optimal tool-call path."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    match: MatchMode = MatchMode.SUBSET
    optional: bool = False

    @model_validator(mode="after")
    def _regexes_compile(self) -> ReferenceStep:
        if self.match is MatchMode.REGEX:
            for key, pattern in self.args.items():
                try:
                    re.compile(str(pattern))
                except re.error as exc:
                    raise ValueError(f"args[{key!r}] is not a valid regex: {exc}") from exc
        return self


class TaskSpec(Strict):
    """Declarative description of one test case. Paths are relative to the task dir."""

    id: str
    title: str
    prompt: str
    image: str = "agent-eval-base:latest"
    repo_dir: str = "repo"
    hidden_tests_dir: str = "hidden_tests"
    solution_dir: str | None = "solution"
    reference_path_file: str | None = "reference_path.json"
    test_command: list[str] = Field(default_factory=lambda: ["python", "-m", "pytest", "-q"])
    timeout_s: int = 300


# --- Trace -----------------------------------------------------------------


class _Step(Strict):
    index: int
    t_ms: float = Field(description="Milliseconds since run start")


class LLMCallStep(_Step):
    type: Literal["llm_call"] = "llm_call"
    request_id: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    status: int = 200


class ToolCallStep(_Step):
    type: Literal["tool_call"] = "tool_call"
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultStep(_Step):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    output: str = ""
    is_error: bool = False


class FaultKind(StrEnum):
    DROP = "drop"
    JITTER = "jitter"
    HTTP_500 = "http_500"
    HTTP_429 = "http_429"


class FaultStep(_Step):
    type: Literal["fault"] = "fault"
    kind: FaultKind
    request_id: str | None = None
    detail: str = ""


TraceStep = Annotated[
    LLMCallStep | ToolCallStep | ToolResultStep | FaultStep, Field(discriminator="type")
]


class Trace(Strict):
    steps: list[TraceStep] = Field(default_factory=list)

    def tool_calls(self) -> list[ToolCallStep]:
        return [s for s in self.steps if isinstance(s, ToolCallStep)]

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps if isinstance(s, LLMCallStep))

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps if isinstance(s, LLMCallStep))


# --- Results ---------------------------------------------------------------


class RunMode(StrEnum):
    RECORD = "record"
    REPLAY = "replay"
    CHAOS = "chaos"


class StateScore(Strict):
    exit_code: int
    passed: bool
    output: str = ""


class AlignKind(StrEnum):
    MATCHED = "matched"  # same tool, arguments conform to the reference
    ARG_MISMATCH = "arg_mismatch"  # same tool, arguments do not conform
    MISSING = "missing"  # a required reference step the agent never performed
    SKIPPED_OPTIONAL = "skipped_optional"  # an optional reference step, not penalised
    EXTRA = "extra"  # an agent call with no counterpart in the reference


class AlignedPair(Strict):
    kind: AlignKind
    ref_index: int | None = None  # index into the reference path
    call_index: int | None = None  # index into the trace's tool calls
    tool: str
    detail: str = ""


class TrajectoryScore(Strict):
    """How closely the agent's tool-call sequence follows the reference path."""

    score: float  # weighted blend of the three components below, in [0, 1]
    tool_precision: float  # matched calls / agent calls (penalises extras)
    tool_recall: float  # matched reference steps / required reference steps
    tool_selection: float  # F1 of precision and recall
    arg_conformity: float  # matched calls whose arguments conform / matched calls
    efficiency: float  # 1 - (redundant calls + loops) / agent calls, floored at 0
    redundant_calls: int  # repeats of an identical call that learned nothing new
    loops: int  # back-to-back repeats of a 2-3 call cycle (ABAB, ABCABC)
    extra_calls: int
    missing_steps: int
    n_calls: int
    n_reference: int
    alignment: list[AlignedPair] = Field(default_factory=list)


class RunResult(Strict):
    task_id: str
    iteration: int
    mode: RunMode
    seed: int | None = None
    state: StateScore | None = None
    trajectory: TrajectoryScore | None = None
    diff: str = ""
    trace: Trace = Field(default_factory=Trace)
    wall_ms: float = 0.0
    agent_exit_code: int | None = None
    cassette_misses: int = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.state is not None and self.state.passed and self.error is None
