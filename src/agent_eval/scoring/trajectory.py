"""Trajectory efficiency: compare the agent's tool calls with an optimal reference path.

1. Align the two sequences (order-preserving dynamic programme, in the Needleman-Wunsch family).
   Calls pair up only when they use the same tool; argument conformity then decides how well.
2. From the alignment derive tool-selection precision/recall, argument conformity, and
   efficiency (redundant calls and loops), and blend them into one score.

The blend is a heuristic; every component is reported separately and the weights are
configurable, so nothing is hidden inside the single number.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from ..schemas import (
    AlignedPair,
    AlignKind,
    MatchMode,
    ReferenceStep,
    ToolCallStep,
    ToolResultStep,
    Trace,
    TrajectoryScore,
)

# Alignment scores. Matching a same-tool call must always beat leaving both unmatched
# (-1 extra, -1 missing), otherwise a call with wrong arguments would look like two errors.
MATCH_CONFORMING = 3
MATCH_NONCONFORMING = 1
GAP_EXTRA = -1
GAP_MISSING = -1
GAP_OPTIONAL = 0

PATH_KEYS = frozenset({"path", "file", "filename", "file_path", "filepath"})
WORKSPACE_PREFIXES = ("/workspace/", "./")
MIN_LOOP_PERIOD, MAX_LOOP_PERIOD = 2, 3


class TrajectoryWeights(BaseModel):
    selection: float = 0.4
    arguments: float = 0.3
    efficiency: float = 0.3

    def blend(self, selection: float, arguments: float, efficiency: float) -> float:
        total = self.selection + self.arguments + self.efficiency
        raw = (
            self.selection * selection + self.arguments * arguments + self.efficiency * efficiency
        )
        return raw / total if total else 0.0


# --- argument matching -------------------------------------------------------------------


def _norm(key: str, value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        if key in PATH_KEYS:
            for prefix in WORKSPACE_PREFIXES:
                value = value.removeprefix(prefix)
    return value


def args_conform(step: ReferenceStep, actual: dict[str, Any]) -> bool:
    expected = step.args
    match step.match:
        case MatchMode.ANY:
            return True
        case MatchMode.EXACT:
            return {k: _norm(k, v) for k, v in expected.items()} == {
                k: _norm(k, v) for k, v in actual.items()
            }
        case MatchMode.SUBSET:
            return all(k in actual and _norm(k, actual[k]) == _norm(k, v) for k, v in expected.items())
        case MatchMode.REGEX:
            return all(
                k in actual and re.search(str(v), str(_norm(k, actual[k]))) is not None
                for k, v in expected.items()
            )
    return False  # unreachable for valid enums


# --- alignment ----------------------------------------------------------------------------


def align(reference: Sequence[ReferenceStep], calls: Sequence[ToolCallStep]) -> list[AlignedPair]:
    m, n = len(reference), len(calls)
    conform = [
        [args_conform(reference[i], calls[j].arguments) if reference[i].tool == calls[j].name else None
         for j in range(n)]
        for i in range(m)
    ]  # fmt: skip

    def missing_cost(i: int) -> int:
        return GAP_OPTIONAL if reference[i].optional else GAP_MISSING

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + missing_cost(i - 1)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + GAP_EXTRA
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best = max(dp[i - 1][j] + missing_cost(i - 1), dp[i][j - 1] + GAP_EXTRA)
            c = conform[i - 1][j - 1]
            if c is not None:
                best = max(best, dp[i - 1][j - 1] + (MATCH_CONFORMING if c else MATCH_NONCONFORMING))
            dp[i][j] = best

    # Backtrace. On ties prefer a match, then an extra call, then a missing step; this is
    # deterministic, and matches the *latest* of several identical calls.
    pairs: list[AlignedPair] = []
    i, j = m, n
    while i > 0 or j > 0:
        c = conform[i - 1][j - 1] if i > 0 and j > 0 else None
        if c is not None and dp[i][j] == dp[i - 1][j - 1] + (
            MATCH_CONFORMING if c else MATCH_NONCONFORMING
        ):
            ref = reference[i - 1]
            pairs.append(
                AlignedPair(
                    kind=AlignKind.MATCHED if c else AlignKind.ARG_MISMATCH,
                    ref_index=i - 1,
                    call_index=j - 1,
                    tool=ref.tool,
                    detail="" if c else f"expected {ref.match.value} {ref.args}, got {calls[j - 1].arguments}",
                )
            )
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + GAP_EXTRA:
            pairs.append(AlignedPair(kind=AlignKind.EXTRA, call_index=j - 1, tool=calls[j - 1].name))
            j -= 1
        else:
            ref = reference[i - 1]
            pairs.append(
                AlignedPair(
                    kind=AlignKind.SKIPPED_OPTIONAL if ref.optional else AlignKind.MISSING,
                    ref_index=i - 1,
                    tool=ref.tool,
                )
            )
            i -= 1
    pairs.reverse()
    return pairs


# --- efficiency ---------------------------------------------------------------------------


def _signature(call: ToolCallStep) -> str:
    return call.name + json.dumps(call.arguments, sort_keys=True, default=str)


def count_redundant(trace: Trace) -> int:
    """Repeats of an identical call that returned the same output as last time (nothing new).

    A repeat whose output changed is legitimate (state moved on, e.g. tests re-run after an edit).
    If a result is unavailable the repeat is counted, since nothing shows it taught anything.
    """
    outputs: dict[str, str] = {}
    for step in trace.steps:
        if isinstance(step, ToolResultStep):
            outputs[step.call_id] = step.output
    last_seen: dict[str, str | None] = {}
    redundant = 0
    for call in trace.tool_calls():
        sig = _signature(call)
        out = outputs.get(call.call_id)
        if sig in last_seen:
            prev = last_seen[sig]
            if prev is None or out is None or prev == out:
                redundant += 1
        last_seen[sig] = out
    return redundant


def count_loops(calls: Sequence[ToolCallStep]) -> int:
    """Count runs where a cycle of 2-3 distinct calls repeats back-to-back at least twice.

    Plain repeats of one call (AA) are not loops here; `count_redundant` already covers them.
    """
    seq = [_signature(c) for c in calls]
    loops, i = 0, 0
    while i < len(seq):
        for period in range(MIN_LOOP_PERIOD, MAX_LOOP_PERIOD + 1):
            cycle = seq[i : i + period]
            if len(cycle) < period or len(set(cycle)) < 2:
                continue
            reps = 1
            while seq[i + reps * period : i + (reps + 1) * period] == cycle:
                reps += 1
            if reps >= 2:
                loops += 1
                i += reps * period
                break
        else:
            i += 1
    return loops


# --- scoring ------------------------------------------------------------------------------


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def score_trajectory(
    trace: Trace,
    reference: Sequence[ReferenceStep],
    weights: TrajectoryWeights | None = None,
) -> TrajectoryScore:
    weights = weights or TrajectoryWeights()
    calls = trace.tool_calls()
    alignment = align(reference, calls)

    by_kind = lambda kind: [p for p in alignment if p.kind is kind]
    matched = by_kind(AlignKind.MATCHED)
    mismatched = by_kind(AlignKind.ARG_MISMATCH)
    extra = by_kind(AlignKind.EXTRA)
    missing = by_kind(AlignKind.MISSING)
    optional_done = [
        p for p in matched + mismatched if reference[p.ref_index].optional  # type: ignore[index]
    ]

    paired = len(matched) + len(mismatched)
    required = sum(1 for r in reference if not r.optional)
    precision = _ratio(paired, len(calls))
    recall = _ratio(paired, required + len(optional_done)) if required or optional_done else 1.0
    selection = _ratio(2 * precision * recall, precision + recall)
    conformity = _ratio(len(matched), paired)

    redundant = count_redundant(trace)
    loops = count_loops(calls)
    efficiency = max(0.0, 1.0 - _ratio(redundant + loops, len(calls))) if calls else 0.0

    return TrajectoryScore(
        score=weights.blend(selection, conformity, efficiency),
        tool_precision=precision,
        tool_recall=recall,
        tool_selection=selection,
        arg_conformity=conformity,
        efficiency=efficiency,
        redundant_calls=redundant,
        loops=loops,
        extra_calls=len(extra),
        missing_steps=len(missing),
        n_calls=len(calls),
        n_reference=len(reference),
        alignment=alignment,
    )
