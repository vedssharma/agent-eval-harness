"""Cassette storage and request keying for the VCR proxy.

A cassette is a JSONL file with one `CassetteEntry` per recorded exchange. Bodies are base64 so
replay is byte-identical; streamed (SSE) responses keep their chunk boundaries and timings.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

from pydantic import Field

from ..schemas import Strict

# Never persisted to a cassette.
REDACTED_HEADERS = frozenset({"authorization", "x-api-key", "proxy-authorization", "cookie"})
REDACTED_VALUE = "REDACTED"
# Body fields that vary between otherwise-identical requests.
DEFAULT_IGNORED_BODY_FIELDS = frozenset({"metadata"})
# Run-specific noise that tools echo back into the conversation (matched on the JSON text).
DEFAULT_SCRUB = (
    re.compile(r"\bin \d+(?:\.\d+)?s\b"),  # pytest: "2 passed in 0.01s"
    re.compile(r"\(\d+(?:\.\d+)?s\)"),  # pytest durations: "(0.02s)"
    re.compile(r"/tmp/pytest-of-[^\s\"\\]+"),
)

Headers = list[tuple[str, str]]


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(data: str) -> bytes:
    return base64.b64decode(data)


def redact(headers: Iterable[tuple[str, str]]) -> Headers:
    return [(k, REDACTED_VALUE if k.lower() in REDACTED_HEADERS else v) for k, v in headers]


class RecordedRequest(Strict):
    method: str
    path: str
    query: str = ""
    headers: Headers = Field(default_factory=list)  # redacted
    body_b64: str = ""


class Chunk(Strict):
    delay_ms: float = 0.0  # since the previous chunk (or since response headers)
    data_b64: str


class RecordedResponse(Strict):
    status: int
    headers: Headers = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    streamed: bool = False

    @property
    def body(self) -> bytes:
        return b"".join(unb64(c.data_b64) for c in self.chunks)


class CassetteEntry(Strict):
    key: str
    occurrence: int  # n-th request with this key within the recording (0-based)
    request: RecordedRequest
    response: RecordedResponse
    ttfb_ms: float = 0.0
    total_ms: float = 0.0


def request_key(
    method: str,
    path: str,
    query: str,
    body: bytes,
    ignore_body_fields: Iterable[str] = DEFAULT_IGNORED_BODY_FIELDS,
    scrub: Iterable[re.Pattern[str]] = DEFAULT_SCRUB,
) -> str:
    """Stable hash of what makes two requests "the same call".

    Headers are excluded (auth and SDK user-agents vary). JSON bodies are canonicalised: key
    order is irrelevant and `ignore_body_fields` are dropped. Query parameters are sorted.
    `scrub` patterns are replaced with a placeholder in the canonical body, so tool output that
    embeds run-specific noise (e.g. pytest timings echoed back to the model) still matches.
    """
    canon_query = urlencode(sorted(parse_qsl(query, keep_blank_values=True)))
    try:
        parsed = json.loads(body) if body else None
    except (ValueError, UnicodeDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        ignored = set(ignore_body_fields)
        parsed = {k: v for k, v in parsed.items() if k not in ignored}
    if parsed is not None or not body:
        canon_body = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
    else:
        canon_body = body
    if scrub:
        text = canon_body.decode(errors="replace")
        for pattern in scrub:
            text = pattern.sub("<scrubbed>", text)
        canon_body = text.encode()
    h = hashlib.sha256()
    for part in (method.upper().encode(), path.encode(), canon_query.encode(), canon_body):
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


class Cassette:
    def __init__(self, path: Path | str, entries: list[CassetteEntry] | None = None) -> None:
        self.path = Path(path)
        self.entries: list[CassetteEntry] = entries or []
        self._by_key: dict[tuple[str, int], CassetteEntry] = {
            (e.key, e.occurrence): e for e in self.entries
        }

    @classmethod
    def load(cls, path: Path | str) -> Cassette:
        lines = Path(path).read_text().splitlines()
        return cls(path, [CassetteEntry.model_validate_json(ln) for ln in lines if ln.strip()])

    def start_recording(self) -> None:
        """Truncate the file; subsequent `append` calls write through immediately."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.entries.clear()
        self._by_key.clear()

    def append(self, entry: CassetteEntry) -> None:
        self.entries.append(entry)
        self._by_key[(entry.key, entry.occurrence)] = entry
        with self.path.open("a") as f:  # crash-safe: every exchange is on disk once complete
            f.write(entry.model_dump_json() + "\n")

    def lookup(self, key: str, occurrence: int) -> CassetteEntry | None:
        return self._by_key.get((key, occurrence))

    def at(self, index: int) -> CassetteEntry | None:
        return self.entries[index] if 0 <= index < len(self.entries) else None

    def __len__(self) -> int:
        return len(self.entries)
