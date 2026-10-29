"""M2: VCR proxy record/replay against an in-process fake upstream (no network, no Docker)."""

import asyncio
import time
from pathlib import Path

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from agent_eval.proxy import MatchPolicy, ProxyMode, VCRProxy, request_key
from agent_eval.proxy.cassette import Cassette

SSE_EVENTS = [b"event: a\ndata: 1\n\n", b"event: b\ndata: 2\n\n", b"event: c\ndata: 3\n\n"]


class FakeUpstream:
    """Stands in for the real API. Responses change on every call, so a replayed response
    that equals the recorded one proves the upstream was not consulted."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_headers: list[dict[str, str]] = []
        self._runner: web.AppRunner | None = None
        self.url = ""

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/v1/messages", self.messages)
        app.router.add_post("/v1/stream", self.stream)
        app.router.add_post("/v1/limited", self.limited)
        app.router.add_get("/v1/models", self.models)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        self.url = f"http://127.0.0.1:{self._runner.addresses[0][1]}"

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    def _count(self, request: web.Request) -> int:
        n = self.calls
        self.calls += 1
        self.seen_headers.append(dict(request.headers))
        return n

    async def messages(self, request):
        n = self._count(request)
        return web.json_response({"id": f"msg_{n}", "echo": await request.json()})

    async def stream(self, request):
        self._count(request)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for ev in SSE_EVENTS:
            await resp.write(ev)
            await asyncio.sleep(0.05)
        return resp

    async def limited(self, request):
        self._count(request)
        return web.json_response({"error": "slow down"}, status=429, headers={"Retry-After": "1"})

    async def models(self, request):
        n = self._count(request)
        return web.json_response({"n": n, "query": dict(request.query)})


@pytest_asyncio.fixture
async def upstream():
    up = FakeUpstream()
    await up.start()
    yield up
    await up.stop()


@pytest.fixture
def cassette_path(tmp_path):
    return str(tmp_path / "cassettes" / "t.jsonl")


async def send(proxy: VCRProxy, method: str, path: str, **kw):
    async with aiohttp.ClientSession() as s, s.request(method, proxy.url + path, **kw) as r:
        return r.status, dict(r.headers), await r.read()


def recorder(upstream, cassette_path, **kw) -> VCRProxy:
    return VCRProxy(ProxyMode.RECORD, cassette_path, upstream=upstream.url, **kw)


def replayer(cassette_path, **kw) -> VCRProxy:
    return VCRProxy(ProxyMode.REPLAY, cassette_path, **kw)


# --- request keys ---------------------------------------------------------------------------


def test_key_ignores_json_key_order_and_metadata():
    a = request_key("POST", "/v1/messages", "", b'{"a":1,"b":2,"metadata":{"u":"x"}}')
    b = request_key("POST", "/v1/messages", "", b'{ "b": 2, "a": 1, "metadata": {"u":"y"} }')
    assert a == b


def test_key_distinguishes_method_path_query_body():
    base = request_key("POST", "/p", "a=1&b=2", b'{"x":1}')
    assert base == request_key("POST", "/p", "b=2&a=1", b'{"x":1}')
    assert base != request_key("GET", "/p", "a=1&b=2", b'{"x":1}')
    assert base != request_key("POST", "/q", "a=1&b=2", b'{"x":1}')
    assert base != request_key("POST", "/p", "a=1&b=3", b'{"x":1}')
    assert base != request_key("POST", "/p", "a=1&b=2", b'{"x":2}')


def test_key_handles_non_json_bodies():
    assert request_key("POST", "/p", "", b"raw") != request_key("POST", "/p", "", b"other")
    assert request_key("POST", "/p", "", b"\xff\xfe") == request_key("POST", "/p", "", b"\xff\xfe")


# --- record -> replay ---------------------------------------------------------------------


async def test_record_then_replay_is_byte_identical_without_upstream(upstream, cassette_path):
    requests = [
        ("POST", "/v1/messages", {"json": {"q": "hello"}}),
        ("POST", "/v1/stream", {"json": {"q": "stream"}}),
        ("POST", "/v1/limited", {"json": {"q": "limit"}}),
        ("GET", "/v1/models?b=2&a=1", {}),
    ]
    recorded = []
    async with recorder(upstream, cassette_path) as proxy:
        for method, path, kw in requests:
            recorded.append(await send(proxy, method, path, **kw))
        assert [e.status for e in proxy.exchanges] == [200, 200, 429, 200]
        assert not any(e.replayed for e in proxy.exchanges)

    calls_after_record = upstream.calls
    await upstream.stop()  # replay must not need the network

    async with replayer(cassette_path) as proxy:
        replayed = [await send(proxy, m, p, **kw) for m, p, kw in requests]
        assert proxy.misses == []
        assert all(e.replayed for e in proxy.exchanges)

    assert upstream.calls == calls_after_record
    for (rs, rh, rb), (ps, ph, pb) in zip(recorded, replayed, strict=True):
        assert (ps, pb) == (rs, rb)
        assert ph.get("Retry-After") == rh.get("Retry-After")
        assert ph["Content-Type"] == rh["Content-Type"]
        assert ph["x-agent-eval-replay"] == "1"


async def test_streaming_chunks_and_timing_are_preserved(upstream, cassette_path):
    async with recorder(upstream, cassette_path) as proxy:
        _, _, live = await send(proxy, "POST", "/v1/stream", json={})
    assert live == b"".join(SSE_EVENTS)

    entry = Cassette.load(cassette_path).entries[0]
    assert entry.response.streamed
    assert entry.response.body == live

    async with replayer(cassette_path, stream_speed=0.0) as proxy:
        t = time.monotonic()
        _, _, body = await send(proxy, "POST", "/v1/stream", json={})
        fast = time.monotonic() - t
    assert body == live and fast < 0.1

    gaps = sum(c.delay_ms for c in entry.response.chunks) / 1000
    async with replayer(cassette_path, stream_speed=1.0) as proxy:
        t = time.monotonic()
        await send(proxy, "POST", "/v1/stream", json={})
        paced = time.monotonic() - t
    assert paced >= gaps * 0.8


async def test_duplicate_requests_replay_in_recorded_order(upstream, cassette_path):
    async with recorder(upstream, cassette_path) as proxy:
        live = [(await send(proxy, "POST", "/v1/messages", json={"q": 1}))[2] for _ in range(3)]
    assert len(set(live)) == 3  # upstream answered differently each time

    async with replayer(cassette_path) as proxy:
        got = [(await send(proxy, "POST", "/v1/messages", json={"q": 1}))[2] for _ in range(3)]
        assert got == live
        status, _, _ = await send(proxy, "POST", "/v1/messages", json={"q": 1})  # 4th: exhausted
        assert status == 424 and len(proxy.misses) == 1


async def test_equivalent_requests_match_despite_key_order_and_metadata(upstream, cassette_path):
    async with recorder(upstream, cassette_path) as proxy:
        _, _, live = await send(
            proxy, "POST", "/v1/messages", json={"a": 1, "b": 2, "metadata": {"user": "u1"}}
        )
    async with replayer(cassette_path) as proxy:
        status, _, body = await send(
            proxy, "POST", "/v1/messages", json={"b": 2, "metadata": {"user": "u2"}, "a": 1}
        )
    assert status == 200 and body == live


# --- misses -------------------------------------------------------------------------------


async def test_cassette_miss_returns_424_and_is_logged(upstream, cassette_path):
    async with recorder(upstream, cassette_path) as proxy:
        await send(proxy, "POST", "/v1/messages", json={"q": "known"})
    async with replayer(cassette_path) as proxy:
        status, headers, body = await send(proxy, "POST", "/v1/messages", json={"q": "unknown"})
        assert status == 424
        assert headers["x-agent-eval-miss"] == "1"
        assert b"cassette_miss" in body
        assert len(proxy.misses) == 1 and proxy.misses[0].miss and not proxy.misses[0].replayed


async def test_by_order_policy_ignores_request_content(upstream, cassette_path):
    async with recorder(upstream, cassette_path) as proxy:
        _, _, first = await send(proxy, "POST", "/v1/messages", json={"q": "one"})
        _, _, second = await send(proxy, "POST", "/v1/messages", json={"q": "two"})
    async with replayer(cassette_path, match=MatchPolicy.BY_ORDER) as proxy:
        got = [(await send(proxy, "POST", "/v1/messages", json={"q": "changed"}))[2] for _ in "ab"]
        assert got == [first, second]
        assert (await send(proxy, "POST", "/v1/messages", json={}))[0] == 424


# --- credentials ----------------------------------------------------------------------------


async def test_secrets_are_forwarded_but_never_written_to_the_cassette(upstream, cassette_path):
    headers = {"x-api-key": "sk-ant-SECRET-123", "Authorization": "Bearer tok-SECRET-456"}
    async with recorder(upstream, cassette_path) as proxy:
        await send(proxy, "POST", "/v1/messages", json={"q": 1}, headers=headers)
    assert upstream.seen_headers[0]["x-api-key"] == "sk-ant-SECRET-123"
    text = Path(cassette_path).read_text()
    assert "SECRET" not in text and "REDACTED" in text


async def test_proxy_can_inject_real_key_so_agent_only_holds_a_dummy(upstream, cassette_path):
    async with recorder(
        upstream, cassette_path, upstream_headers={"x-api-key": "sk-REAL-789"}
    ) as proxy:
        await send(proxy, "POST", "/v1/messages", json={}, headers={"x-api-key": "dummy"})
    assert upstream.seen_headers[0]["x-api-key"] == "sk-REAL-789"
    assert "REAL" not in Path(cassette_path).read_text()


# --- failure modes --------------------------------------------------------------------------


async def test_unreachable_upstream_returns_502_and_is_not_recorded(cassette_path):
    dead = VCRProxy(ProxyMode.RECORD, cassette_path, upstream="http://127.0.0.1:1")
    async with dead:
        status, _, body = await send(dead, "POST", "/v1/messages", json={})
    assert status == 502 and b"upstream_unreachable" in body
    assert len(Cassette.load(cassette_path)) == 0


def test_record_mode_requires_upstream(cassette_path):
    with pytest.raises(ValueError, match="upstream"):
        VCRProxy(ProxyMode.RECORD, cassette_path)


async def test_replay_with_missing_cassette_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        await replayer(str(tmp_path / "nope.jsonl")).start()


async def test_recording_is_flushed_per_exchange(upstream, cassette_path):
    async with recorder(upstream, cassette_path) as proxy:
        await send(proxy, "POST", "/v1/messages", json={"q": 1})
        assert len(Cassette.load(cassette_path)) == 1  # readable before the proxy stops
        await send(proxy, "POST", "/v1/messages", json={"q": 2})
        assert len(Cassette.load(cassette_path)) == 2
