"""asyncio reverse proxy with VCR record/replay.

RECORD forwards to the real upstream and saves each exchange. REPLAY serves saved exchanges
and never touches the network; a request with no match gets HTTP 424 and is logged in
`VCRProxy.misses` so the runner can fail the run instead of silently diverging.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

import aiohttp
from aiohttp import web
from multidict import CIMultiDict

from .cassette import (
    DEFAULT_IGNORED_BODY_FIELDS,
    Cassette,
    CassetteEntry,
    Chunk,
    RecordedRequest,
    RecordedResponse,
    b64,
    redact,
    request_key,
    unb64,
)

MISS_STATUS = 424  # not retried by the Anthropic/OpenAI SDKs, unlike 5xx
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
# Dropped from forwarded requests (aiohttp recomputes them) and from stored responses.
REQUEST_STRIP = HOP_BY_HOP | {"host", "content-length", "accept-encoding"}
RESPONSE_STRIP = HOP_BY_HOP | {"content-length", "content-encoding"}


class ProxyMode(StrEnum):
    RECORD = "record"
    REPLAY = "replay"


class MatchPolicy(StrEnum):
    STRICT = "strict"  # n-th request with a given key gets the n-th recording with that key
    BY_ORDER = "by_order"  # n-th request overall gets the n-th recording, whatever it asks


@dataclass
class Exchange:
    """One proxied request, kept in memory for trace extraction and assertions."""

    index: int
    key: str
    occurrence: int
    method: str
    path: str
    status: int
    started_ms: float  # since proxy start
    latency_ms: float
    replayed: bool
    miss: bool = False
    ts: float = 0.0  # wall-clock epoch seconds at request start, for merging with agent logs


class VCRProxy:
    def __init__(
        self,
        mode: ProxyMode,
        cassette_path: str,
        *,
        upstream: str | None = None,
        match: MatchPolicy = MatchPolicy.STRICT,
        ignore_body_fields: frozenset[str] = DEFAULT_IGNORED_BODY_FIELDS,
        upstream_headers: dict[str, str] | None = None,
        stream_speed: float = 0.0,
        exchange_log: str | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        """
        upstream_headers: set on every forwarded request (e.g. the real API key), so the
            sandboxed agent only ever holds a dummy credential.
        stream_speed: replay SSE chunks with their recorded delays x this factor (0 = instant).
        exchange_log: if set, every exchange (hits, misses, recordings) is appended to this
            JSONL file as it happens.
        """
        self.mode = ProxyMode(mode)
        if self.mode is ProxyMode.RECORD and not upstream:
            raise ValueError("RECORD mode needs an upstream URL")
        self.cassette = Cassette(cassette_path)
        self.upstream = upstream.rstrip("/") if upstream else None
        self.match = match
        self.ignore_body_fields = ignore_body_fields
        self.upstream_headers = upstream_headers or {}
        self.stream_speed = stream_speed
        self.exchange_log = Path(exchange_log) if exchange_log else None
        self.host, self.port = host, port

        self.exchanges: list[Exchange] = []
        self.misses: list[Exchange] = []
        self._seen: Counter[str] = Counter()
        self._t0 = 0.0
        self._runner: web.AppRunner | None = None
        self._session: aiohttp.ClientSession | None = None

    # --- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self.mode is ProxyMode.REPLAY:
            self.cassette = Cassette.load(self.cassette.path)
        else:
            self.cassette.start_recording()
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)
            )
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self.port = self._runner.addresses[0][1]
        self._t0 = time.monotonic()

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
        if self._session:
            await self._session.close()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # --- request handling --------------------------------------------------

    def _now_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.read()
        query = request.query_string
        key = request_key(request.method, request.path, query, body, self.ignore_body_fields)
        occurrence = self._seen[key]
        self._seen[key] += 1
        index = len(self.exchanges)
        started = self._now_ms()
        wall = time.time()

        if self.mode is ProxyMode.REPLAY:
            response, status, miss = await self._replay(request, key, occurrence, index)
        else:
            response, status, miss = await self._record(
                request, body, query, key, occurrence
            )
        ex = Exchange(
            index=index,
            key=key,
            occurrence=occurrence,
            method=request.method,
            path=request.path,
            status=status,
            started_ms=started,
            latency_ms=self._now_ms() - started,
            replayed=self.mode is ProxyMode.REPLAY and not miss,
            miss=miss,
            ts=wall,
        )
        self.exchanges.append(ex)
        if self.exchange_log:
            with self.exchange_log.open("a") as f:
                f.write(json.dumps(asdict(ex)) + "\n")
        if miss:
            self.misses.append(ex)
        return response

    # --- REPLAY ------------------------------------------------------------

    async def _replay(
        self, request: web.Request, key: str, occurrence: int, index: int
    ) -> tuple[web.StreamResponse, int, bool]:
        if self.match is MatchPolicy.BY_ORDER:
            entry = self.cassette.at(index)
        else:
            entry = self.cassette.lookup(key, occurrence)
        if entry is None:
            payload = {
                "type": "error",
                "error": {
                    "type": "cassette_miss",
                    "message": (
                        f"no recording for {request.method} {request.path} "
                        f"(key={key[:12]}, occurrence={occurrence}, policy={self.match.value})"
                    ),
                },
            }
            resp = web.json_response(
                payload, status=MISS_STATUS, headers={"x-agent-eval-miss": "1"}
            )
            return resp, MISS_STATUS, True

        rec = entry.response
        headers = CIMultiDict(rec.headers)
        headers["x-agent-eval-replay"] = "1"
        if not rec.streamed:
            return web.Response(body=rec.body, status=rec.status, headers=headers), rec.status, False

        resp = web.StreamResponse(status=rec.status, headers=headers)
        await resp.prepare(request)
        for chunk in rec.chunks:
            if self.stream_speed > 0 and chunk.delay_ms > 0:
                await asyncio.sleep(chunk.delay_ms / 1000 * self.stream_speed)
            await resp.write(unb64(chunk.data_b64))
        await resp.write_eof()
        return resp, rec.status, False

    # --- RECORD ------------------------------------------------------------

    async def _record(
        self, request: web.Request, body: bytes, query: str, key: str, occurrence: int
    ) -> tuple[web.StreamResponse, int, bool]:
        assert self._session is not None and self.upstream is not None
        fwd = CIMultiDict(
            (k, v) for k, v in request.headers.items() if k.lower() not in REQUEST_STRIP
        )
        fwd["Accept-Encoding"] = "identity"  # store raw bytes, not gzip
        for name, value in self.upstream_headers.items():
            fwd[name] = value
        url = self.upstream + request.path + (f"?{query}" if query else "")

        start = time.monotonic()
        try:
            upstream = await self._session.request(
                request.method, url, data=body or None, headers=fwd, allow_redirects=False
            )
        except aiohttp.ClientError as exc:
            resp = web.json_response(
                {"type": "error", "error": {"type": "upstream_unreachable", "message": str(exc)}},
                status=502,
            )
            return resp, 502, False

        async with upstream:
            resp_headers = [
                (k, v) for k, v in upstream.headers.items() if k.lower() not in RESPONSE_STRIP
            ]
            ttfb_ms = (time.monotonic() - start) * 1000
            streamed = upstream.headers.get("content-type", "").startswith("text/event-stream")
            chunks: list[Chunk] = []

            if streamed:
                out: web.StreamResponse = web.StreamResponse(
                    status=upstream.status, headers=CIMultiDict(resp_headers)
                )
                await out.prepare(request)
                last = time.monotonic()
                async for data in upstream.content.iter_any():
                    now = time.monotonic()
                    chunks.append(Chunk(delay_ms=(now - last) * 1000, data_b64=b64(data)))
                    last = now
                    await out.write(data)
                await out.write_eof()
            else:
                data = await upstream.read()
                chunks.append(Chunk(delay_ms=0.0, data_b64=b64(data)))
                out = web.Response(
                    body=data, status=upstream.status, headers=CIMultiDict(resp_headers)
                )

            self.cassette.append(
                CassetteEntry(
                    key=key,
                    occurrence=occurrence,
                    request=RecordedRequest(
                        method=request.method,
                        path=request.path,
                        query=query,
                        headers=redact(request.headers.items()),
                        body_b64=b64(body),
                    ),
                    response=RecordedResponse(
                        status=upstream.status,
                        headers=resp_headers,
                        chunks=chunks,
                        streamed=streamed,
                    ),
                    ttfb_ms=ttfb_ms,
                    total_ms=(time.monotonic() - start) * 1000,
                )
            )
            return out, upstream.status, False


def _main() -> None:
    """`python -m agent_eval.proxy`: run standalone (e.g. as a sidecar container)."""
    import argparse
    import os

    p = argparse.ArgumentParser(prog="agent_eval.proxy")
    p.add_argument("--mode", choices=[m.value for m in ProxyMode], required=True)
    p.add_argument("--cassette", required=True)
    p.add_argument("--upstream")
    p.add_argument("--match", choices=[m.value for m in MatchPolicy], default="strict")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--stream-speed", type=float, default=0.0)
    p.add_argument("--exchange-log", help="append every exchange to this JSONL file")
    p.add_argument(
        "--inject-header",
        action="append",
        default=[],
        metavar="HEADER=ENVVAR",
        help="send the value of $ENVVAR upstream as HEADER (keeps real keys out of the sandbox)",
    )
    args = p.parse_args()
    injected = {}
    for spec in args.inject_header:
        header, _, var = spec.partition("=")
        injected[header] = os.environ[var]

    async def run() -> None:
        proxy = VCRProxy(
            ProxyMode(args.mode),
            args.cassette,
            upstream=args.upstream,
            match=MatchPolicy(args.match),
            upstream_headers=injected,
            stream_speed=args.stream_speed,
            exchange_log=args.exchange_log,
            host=args.host,
            port=args.port,
        )
        async with proxy:
            print(json.dumps({"listening": proxy.url, "mode": args.mode}), flush=True)
            await asyncio.Event().wait()

    asyncio.run(run())


if __name__ == "__main__":
    _main()
