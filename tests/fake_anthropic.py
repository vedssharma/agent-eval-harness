"""A scripted stand-in for the Anthropic Messages API, served from a background thread.

It plays a fixed conversation keyed on the number of messages the agent has sent, so the
agent really runs its tools in the sandbox while the "model" side is deterministic.
"""

import asyncio
import threading

from aiohttp import web

BAD_CONTENT = "def page_slice(items, page, per_page):\n    return items  # did not fix it\n"


class ScriptedAnthropic:
    def __init__(self, fixed_source: str, *, fix: bool = True) -> None:
        self.source = fixed_source if fix else BAD_CONTENT
        self.calls = 0
        self.seen_keys: list[str] = []
        self.port = 0
        self._loop = asyncio.new_event_loop()
        self._runner: web.AppRunner | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    # sidecar containers reach the host through this name (Docker Desktop and host-gateway)
    @property
    def container_url(self) -> str:
        return f"http://host.docker.internal:{self.port}"

    def _script(self, n_messages: int) -> tuple[list[dict], str, int]:
        steps = {
            1: ([{"type": "tool_use", "id": "toolu_1", "name": "read_file",
                  "input": {"path": "src/pager.py"}}], "tool_use"),
            3: ([{"type": "tool_use", "id": "toolu_2", "name": "write_file",
                  "input": {"path": "src/pager.py", "content": self.source}}], "tool_use"),
            5: ([{"type": "tool_use", "id": "toolu_3", "name": "bash",
                  "input": {"command": "python -m pytest -q"}}], "tool_use"),
            7: ([{"type": "text", "text": "Added boundary validation to page_slice."}], "end_turn"),
        }  # fmt: skip
        content, stop = steps.get(n_messages, steps[7])
        return content, stop, n_messages

    async def messages(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.calls += 1
        self.seen_keys.append(request.headers.get("x-api-key", ""))
        content, stop, n = self._script(len(body["messages"]))
        return web.json_response(
            {
                "id": f"msg_{n}",
                "type": "message",
                "role": "assistant",
                "model": "claude-scripted",
                "content": content,
                "stop_reason": stop,
                "usage": {"input_tokens": 100 * n, "output_tokens": 10 * n},
            }
        )

    def start(self) -> "ScriptedAnthropic":
        async def setup() -> None:
            app = web.Application()
            app.router.add_post("/v1/messages", self.messages)
            self._runner = web.AppRunner(app, access_log=None)
            await self._runner.setup()
            site = web.TCPSite(self._runner, "0.0.0.0", 0)
            await site.start()
            self.port = self._runner.addresses[0][1]

        def serve() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(setup())
            self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=serve, daemon=True)
        self._thread.start()
        assert self._ready.wait(10)
        return self

    def stop(self) -> None:
        if self._runner:
            fut = asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop)
            fut.result(10)
            self._runner = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(10)
