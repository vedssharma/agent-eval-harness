"""Build a `Trace` from the proxy's exchange log + cassette, optionally merged with an agent log.

The proxy traffic alone is enough for any agent: tool calls come from `tool_use` blocks in model
responses and tool results from the `tool_result` blocks in the following request. When the agent
also wrote its own log (the reference agent does), that is preferred for tool calls and results,
because it carries exact timestamps.
"""

from __future__ import annotations

from typing import Any

from .anthropic_parse import ParsedResponse, parse_response, request_tool_results
from .proxy.cassette import Cassette, CassetteEntry, unb64
from .schemas import LLMCallStep, ToolCallStep, ToolResultStep, Trace, TraceStep

_CLASSES = {
    "llm_call": LLMCallStep,
    "tool_call": ToolCallStep,
    "tool_result": ToolResultStep,
}
# Tie-break for events sharing a timestamp: the call precedes its result.
_PRIORITY = {"llm_call": 0, "tool_call": 1, "tool_result": 2}


def _entry_for(ex: dict[str, Any], cassette: Cassette) -> CassetteEntry | None:
    return None if ex["miss"] else cassette.lookup(ex["key"], ex["occurrence"])


def build_trace(
    agent_log: list[dict[str, Any]], exchanges: list[dict[str, Any]], cassette: Cassette
) -> Trace:
    events: list[tuple[float, dict[str, Any]]] = []
    use_agent_log = any(rec["type"] == "tool_call" for rec in agent_log)
    seen_results: set[str] = set()

    for i, ex in enumerate(exchanges):
        entry = _entry_for(ex, cassette)
        parsed = (
            parse_response(entry.response.body, entry.response.streamed)
            if entry
            else ParsedResponse()
        )
        events.append(
            (
                ex["ts"],
                {
                    "type": "llm_call",
                    "request_id": f"{ex['key'][:8]}-{ex['occurrence']}",
                    "model": parsed.model,
                    "input_tokens": parsed.input_tokens,
                    "output_tokens": parsed.output_tokens,
                    "latency_ms": ex["latency_ms"],
                    "status": ex["status"],
                },
            )
        )
        if use_agent_log:
            continue

        done = ex["ts"] + ex["latency_ms"] / 1000
        for use in parsed.tool_uses:
            events.append(
                (
                    done,
                    {
                        "type": "tool_call",
                        "call_id": use["id"],
                        "name": use["name"],
                        "arguments": use["input"],
                    },
                )
            )
        # The result of this turn's tool calls arrives in the *next* request.
        nxt = exchanges[i + 1] if i + 1 < len(exchanges) else None
        nxt_entry = _entry_for(nxt, cassette) if nxt else None
        if nxt and nxt_entry:
            for res in request_tool_results(unb64(nxt_entry.request.body_b64)):
                if res.tool_use_id in seen_results:  # a retried request repeats its results
                    continue
                seen_results.add(res.tool_use_id)
                events.append(
                    (
                        nxt["ts"] - 1e-6,
                        {
                            "type": "tool_result",
                            "call_id": res.tool_use_id,
                            "output": res.output,
                            "is_error": res.is_error,
                        },
                    )
                )

    if use_agent_log:
        for rec in agent_log:
            if rec["type"] == "tool_call":
                events.append(
                    (
                        rec["ts"],
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
                        {
                            "type": "tool_result",
                            "call_id": rec["id"],
                            "output": rec["output"],
                            "is_error": rec["is_error"],
                        },
                    )
                )

    events.sort(key=lambda e: (e[0], _PRIORITY[e[1]["type"]]))
    t0 = events[0][0] if events else 0.0
    steps: list[TraceStep] = []
    for i, (ts, payload) in enumerate(events):
        step = {**payload, "index": i, "t_ms": (ts - t0) * 1000}
        steps.append(_CLASSES[payload["type"]].model_validate(step))
    return Trace(steps=steps)


def trace_from_cassette(cassette: Cassette) -> Trace:
    """Rebuild a trace from a cassette alone (no exchange log, no agent log).

    Timestamps are synthesised from the recorded latencies, so only ordering and content are
    meaningful, which is all that trajectory scoring needs.
    """
    exchanges, t = [], 0.0
    for e in cassette.entries:
        exchanges.append(
            {
                "key": e.key,
                "occurrence": e.occurrence,
                "ts": t,
                "latency_ms": e.total_ms,
                "status": e.response.status,
                "miss": False,
            }
        )
        t += e.total_ms / 1000 + 0.001
    return build_trace([], exchanges, cassette)
