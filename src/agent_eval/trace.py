"""Build a `Trace` from the agent's tool log and the proxy's exchange log + cassette."""

from __future__ import annotations

import json
from typing import Any

from .proxy.cassette import Cassette
from .schemas import LLMCallStep, ToolCallStep, ToolResultStep, Trace, TraceStep

_CLASSES = {
    "llm_call": LLMCallStep,
    "tool_call": ToolCallStep,
    "tool_result": ToolResultStep,
}


def _usage(body: bytes) -> tuple[str | None, int, int]:
    try:
        data = json.loads(body)
    except ValueError:
        return None, 0, 0
    if not isinstance(data, dict):
        return None, 0, 0
    usage = data.get("usage") or {}
    return data.get("model"), usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def build_trace(
    agent_log: list[dict[str, Any]], exchanges: list[dict[str, Any]], cassette: Cassette
) -> Trace:
    """Merge both logs on wall-clock time. Streamed (SSE) usage is not parsed yet."""
    events: list[tuple[float, int, dict[str, Any]]] = []
    for ex in exchanges:
        entry = cassette.lookup(ex["key"], ex["occurrence"])
        model, tin, tout = _usage(entry.response.body) if entry and not ex["miss"] else (None,) * 3
        events.append(
            (
                ex["ts"],
                0,
                {
                    "type": "llm_call",
                    "request_id": f"{ex['key'][:8]}-{ex['occurrence']}",
                    "model": model,
                    "input_tokens": tin or 0,
                    "output_tokens": tout or 0,
                    "latency_ms": ex["latency_ms"],
                    "status": ex["status"],
                },
            )
        )
    for rec in agent_log:
        if rec["type"] == "tool_call":
            events.append(
                (
                    rec["ts"],
                    1,
                    {
                        "type": "tool_call",
                        "call_id": rec["id"],
                        "name": rec["name"],
                        "arguments": rec["arguments"],
                    },
                )
            )
        elif rec["type"] == "tool_result":
            events.append(
                (
                    rec["ts"],
                    2,
                    {
                        "type": "tool_result",
                        "call_id": rec["id"],
                        "output": rec["output"],
                        "is_error": rec["is_error"],
                    },
                )
            )

    events.sort(key=lambda e: (e[0], e[1]))
    t0 = events[0][0] if events else 0.0
    steps: list[TraceStep] = []
    for i, (ts, _, payload) in enumerate(events):
        step = {**payload, "index": i, "t_ms": (ts - t0) * 1000}
        steps.append(_CLASSES[payload["type"]].model_validate(step))
    return Trace(steps=steps)
