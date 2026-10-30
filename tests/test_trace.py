"""M4: Messages API parsing and proxy-derived traces (no Docker needed)."""

import json

import pytest

from agent_eval.anthropic_parse import parse_response, request_tool_results
from agent_eval.proxy.cassette import (
    Cassette,
    CassetteEntry,
    Chunk,
    RecordedRequest,
    RecordedResponse,
    b64,
)
from agent_eval.schemas import LLMCallStep, ToolCallStep, ToolResultStep
from agent_eval.trace import build_trace, trace_from_cassette


def sse(*events: tuple[str, dict]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


STREAM = sse(
    ("message_start", {"type": "message_start", "message": {
        "model": "claude-x", "usage": {"input_tokens": 25, "output_tokens": 1}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0,
                             "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                             "delta": {"type": "text_delta", "text": "Let me look."}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("content_block_start", {"type": "content_block_start", "index": 1,
                             "content_block": {"type": "tool_use", "id": "toolu_s", "name": "bash",
                                               "input": {}}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 1,
                             "delta": {"type": "input_json_delta", "partial_json": '{"comm'}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 1,
                             "delta": {"type": "input_json_delta", "partial_json": 'and": "ls"}'}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                       "usage": {"output_tokens": 40}}),
    ("message_stop", {"type": "message_stop"}),
)  # fmt: skip


def message(content: list[dict], tokens=(10, 5)) -> bytes:
    return json.dumps(
        {
            "type": "message",
            "model": "claude-x",
            "content": content,
            "stop_reason": "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn",
            "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
        }
    ).encode()


def tool_use(id_: str, name: str, **args) -> dict:
    return {"type": "tool_use", "id": id_, "name": name, "input": args}


def request_with_results(*results: tuple[str, str, bool]) -> bytes:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": i, "content": out, "is_error": err}
                        for i, out, err in results
                    ],
                },
            ]
        }
    ).encode()


def entry(n: int, req: bytes, resp: bytes, *, status=200, streamed=False, ms=100.0) -> CassetteEntry:
    chunks = [Chunk(data_b64=b64(resp))]
    return CassetteEntry(
        key=f"key{n}",
        occurrence=0,
        request=RecordedRequest(method="POST", path="/v1/messages", body_b64=b64(req)),
        response=RecordedResponse(status=status, chunks=chunks, streamed=streamed),
        total_ms=ms,
    )


# --- parsing -------------------------------------------------------------------------------


def test_parse_json_response():
    r = parse_response(message([{"type": "text", "text": "hi"}, tool_use("t1", "bash", command="ls")]))
    assert (r.model, r.input_tokens, r.output_tokens) == ("claude-x", 10, 5)
    assert r.tool_uses == [{"id": "t1", "name": "bash", "input": {"command": "ls"}}]
    assert r.text == "hi" and r.stop_reason == "tool_use"


def test_parse_sse_stream_reassembles_tool_arguments_and_usage():
    r = parse_response(STREAM, streamed=True)
    assert r.tool_uses == [{"id": "toolu_s", "name": "bash", "input": {"command": "ls"}}]
    assert (r.model, r.input_tokens, r.output_tokens) == ("claude-x", 25, 40)
    assert r.text == "Let me look." and r.stop_reason == "tool_use"


def test_sse_is_detected_without_the_streamed_flag():
    assert parse_response(STREAM).tool_uses[0]["name"] == "bash"


def test_parse_error_and_garbage_bodies():
    err = parse_response(json.dumps({"type": "error", "error": {"message": "overloaded"}}).encode())
    assert err.error == "overloaded" and not err.tool_uses
    for junk in (b"", b"not json", b"[1,2]", b"\xff\xfe"):
        assert parse_response(junk).tool_uses == []


def test_request_tool_results_handles_string_and_block_content():
    body = request_with_results(("a", "plain", False), ("b", "boom", True))
    got = request_tool_results(body)
    assert [(r.tool_use_id, r.output, r.is_error) for r in got] == [
        ("a", "plain", False),
        ("b", "boom", True),
    ]
    blocks = json.dumps(
        {"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c",
             "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]}]}]}
    ).encode()  # fmt: skip
    assert request_tool_results(blocks)[0].output == "xy"
    assert request_tool_results(b'{"messages": [{"role": "user", "content": "text only"}]}') == []


# --- proxy-derived trace -------------------------------------------------------------------


def two_turn_cassette() -> Cassette:
    return Cassette(
        "mem.jsonl",
        [
            entry(1, b'{"messages":[{"role":"user","content":"task"}]}',
                  message([tool_use("t1", "read_file", path="a.py")], (100, 10))),
            entry(2, request_with_results(("t1", "contents", False)),
                  message([{"type": "text", "text": "done"}], (150, 5))),
        ],
    )  # fmt: skip


def test_trace_from_cassette_orders_calls_results_and_llm_steps():
    trace = trace_from_cassette(two_turn_cassette())
    assert [s.type for s in trace.steps] == ["llm_call", "tool_call", "tool_result", "llm_call"]
    call = trace.steps[1]
    assert isinstance(call, ToolCallStep) and (call.name, call.arguments) == ("read_file", {"path": "a.py"})
    result = trace.steps[2]
    assert isinstance(result, ToolResultStep) and result.call_id == "t1" and result.output == "contents"
    assert (trace.input_tokens, trace.output_tokens) == (250, 15)


def test_trace_from_cassette_parses_streamed_entries():
    c = Cassette("mem.jsonl", [entry(1, b'{"messages":[]}', STREAM, streamed=True)])
    trace = trace_from_cassette(c)
    llm = trace.steps[0]
    assert isinstance(llm, LLMCallStep) and (llm.input_tokens, llm.output_tokens) == (25, 40)
    assert [(s.name, s.arguments) for s in trace.tool_calls()] == [("bash", {"command": "ls"})]


def test_retried_request_does_not_duplicate_tool_results():
    req = request_with_results(("t1", "contents", False))
    c = Cassette(
        "mem.jsonl",
        [
            entry(1, b'{"messages":[]}', message([tool_use("t1", "bash", command="ls")])),
            entry(2, req, b'{"type":"error","error":{"message":"slow down"}}', status=429),
            entry(3, req, message([{"type": "text", "text": "ok"}])),
        ],
    )
    trace = trace_from_cassette(c)
    assert sum(isinstance(s, ToolResultStep) for s in trace.steps) == 1
    assert [s.status for s in trace.steps if isinstance(s, LLMCallStep)] == [200, 429, 200]


def test_agent_log_is_preferred_over_proxy_inference_when_present():
    c = two_turn_cassette()
    exchanges = [
        {"key": e.key, "occurrence": 0, "ts": 100.0 + i, "latency_ms": 10.0, "status": 200, "miss": False}
        for i, e in enumerate(c.entries)
    ]
    log = [
        {"ts": 100.5, "type": "tool_call", "id": "t1", "name": "read_file", "arguments": {"path": "a.py"}},
        {"ts": 100.6, "type": "tool_result", "id": "t1", "output": "from the agent log", "is_error": False},
    ]
    trace = build_trace(log, exchanges, c)
    assert [s.type for s in trace.steps] == ["llm_call", "tool_call", "tool_result", "llm_call"]
    assert trace.steps[2].output == "from the agent log"
    assert trace.steps[2].t_ms == pytest.approx(600.0)  # exact agent timestamp, not inferred


def test_missed_exchange_yields_an_llm_step_without_content():
    ex = [{"key": "k", "occurrence": 0, "ts": 1.0, "latency_ms": 1.0, "status": 424, "miss": True}]
    trace = build_trace([], ex, Cassette("mem.jsonl"))
    assert [s.type for s in trace.steps] == ["llm_call"] and trace.steps[0].status == 424
