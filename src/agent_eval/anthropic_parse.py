"""Extract tool calls, results and token usage from Anthropic Messages API traffic.

Works on raw recorded bodies, so a trace can be built from the proxy's cassette alone, for any
agent. Handles both plain JSON responses and server-sent-event streams.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedResponse:
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    text: str = ""
    tool_uses: list[dict[str, Any]] = field(default_factory=list)  # {id, name, input}
    error: str | None = None


@dataclass
class ToolResultBlock:
    tool_use_id: str
    output: str
    is_error: bool


def _loads(raw: bytes | str) -> Any:
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None


def parse_response(body: bytes, streamed: bool = False) -> ParsedResponse:
    """Parse a response body. Unknown or malformed bodies yield an empty `ParsedResponse`."""
    if streamed or body.lstrip().startswith((b"event:", b"data:")):
        return _parse_sse(body)
    return _parse_message(_loads(body))


def _parse_message(data: Any) -> ParsedResponse:
    out = ParsedResponse()
    if not isinstance(data, dict):
        return out
    if data.get("type") == "error":
        out.error = str((data.get("error") or {}).get("message", "error"))
        return out
    usage = data.get("usage") or {}
    out.model = data.get("model")
    out.input_tokens = usage.get("input_tokens", 0) or 0
    out.output_tokens = usage.get("output_tokens", 0) or 0
    out.stop_reason = data.get("stop_reason")
    for block in data.get("content") or []:
        if block.get("type") == "tool_use":
            out.tool_uses.append(
                {"id": block["id"], "name": block["name"], "input": block.get("input") or {}}
            )
        elif block.get("type") == "text":
            out.text += block.get("text", "")
    return out


def _sse_events(body: bytes) -> list[dict[str, Any]]:
    events = []
    for frame in body.decode(errors="replace").replace("\r\n", "\n").split("\n\n"):
        data_lines = [ln[5:].strip() for ln in frame.split("\n") if ln.startswith("data:")]
        if not data_lines:
            continue
        payload = _loads("".join(data_lines))
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _parse_sse(body: bytes) -> ParsedResponse:
    out = ParsedResponse()
    blocks: dict[int, dict[str, Any]] = {}  # content block index -> accumulating tool_use
    for ev in _sse_events(body):
        kind = ev.get("type")
        if kind == "message_start":
            msg = ev.get("message") or {}
            usage = msg.get("usage") or {}
            out.model = msg.get("model")
            out.input_tokens = usage.get("input_tokens", 0) or 0
            out.output_tokens = usage.get("output_tokens", 0) or 0
        elif kind == "message_delta":
            usage = ev.get("usage") or {}
            out.output_tokens = usage.get("output_tokens", out.output_tokens) or 0
            if "input_tokens" in usage:  # present on some API versions
                out.input_tokens = usage["input_tokens"] or out.input_tokens
            out.stop_reason = (ev.get("delta") or {}).get("stop_reason", out.stop_reason)
        elif kind == "content_block_start":
            block = ev.get("content_block") or {}
            if block.get("type") == "tool_use":
                blocks[ev["index"]] = {
                    "id": block["id"],
                    "name": block["name"],
                    "json": "",
                    "initial": block.get("input") or {},
                }
        elif kind == "content_block_delta":
            delta = ev.get("delta") or {}
            if delta.get("type") == "input_json_delta" and ev.get("index") in blocks:
                blocks[ev["index"]]["json"] += delta.get("partial_json", "")
            elif delta.get("type") == "text_delta":
                out.text += delta.get("text", "")
        elif kind == "error":
            out.error = str((ev.get("error") or {}).get("message", "stream error"))
    for index in sorted(blocks):
        b = blocks[index]
        args = _loads(b["json"]) if b["json"] else b["initial"]
        out.tool_uses.append(
            {"id": b["id"], "name": b["name"], "input": args if isinstance(args, dict) else {}}
        )
    return out


def request_tool_results(body: bytes) -> list[ToolResultBlock]:
    """Tool results the agent sent back in a request (the last user message's blocks)."""
    data = _loads(body)
    if not isinstance(data, dict):
        return []
    messages = data.get("messages") or []
    if not messages or messages[-1].get("role") != "user":
        return []
    content = messages[-1].get("content")
    if not isinstance(content, list):
        return []
    results = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            results.append(
                ToolResultBlock(
                    tool_use_id=block.get("tool_use_id", ""),
                    output=_flatten(block.get("content")),
                    is_error=bool(block.get("is_error", False)),
                )
            )
    return results


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""
