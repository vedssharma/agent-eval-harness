"""Reference tool-use agent. Runs INSIDE the sandbox container; standard library only.

Talks to the Anthropic Messages API at $ANTHROPIC_BASE_URL (the VCR proxy sidecar). Every tool
call and result is appended to /task/agent_log.jsonl so the harness can build a trace.

Requests are deterministic (fixed system prompt, tools, message order) so cassettes replay.
Exit codes: 0 finished, 2 fatal API/protocol error, 3 turn limit reached.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", "/workspace"))
LOG = Path(os.environ.get("AGENT_LOG", "/task/agent_log.jsonl"))
PROMPT = Path(os.environ.get("AGENT_PROMPT", "/task/prompt.txt"))
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "http://proxy:8080").rstrip("/")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "dummy")
MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-5-5")
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "20"))
MAX_RETRIES = int(os.environ.get("AGENT_MAX_RETRIES", "5"))
RETRYABLE = {408, 429, 500, 502, 503, 504, 529}
OUTPUT_LIMIT = 10_000

SYSTEM = (
    "You are a software engineering agent working in the git repository at /workspace. "
    "Use the tools to inspect and edit files and to run commands. Make the smallest change "
    "that fully solves the task, run the tests, and when you are done reply with a one-line "
    "summary and no tool calls."
)

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in /workspace and return its output and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file (path relative to /workspace).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a text file (path relative to /workspace).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
]


class Fatal(Exception):
    pass


def log(**entry):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": time.time(), **entry}) + "\n")


def call_api(messages):
    body = json.dumps(
        {
            "model": MODEL,
            "max_tokens": 4096,
            "system": SYSTEM,
            "tools": TOOLS,
            "messages": messages,
        }
    ).encode()
    headers = {
        "content-type": "application/json",
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
    }
    for attempt in range(MAX_RETRIES + 1):
        req = urllib.request.Request(
            f"{BASE_URL}/v1/messages", data=body, headers=headers, method="POST"
        )
        retry_after = None
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            if exc.code not in RETRYABLE:
                raise Fatal(f"HTTP {exc.code}: {detail}") from None
            retry_after = exc.headers.get("retry-after")
            problem = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            problem = f"connection error: {exc}"
        if attempt == MAX_RETRIES:
            raise Fatal(f"giving up after {MAX_RETRIES + 1} attempts ({problem})")
        delay = float(retry_after) if retry_after else min(2**attempt, 30)
        log(type="retry", attempt=attempt + 1, problem=problem, delay_s=delay)
        time.sleep(delay)


def truncate(text):
    if len(text) <= OUTPUT_LIMIT:
        return text
    return text[:OUTPUT_LIMIT] + f"\n[truncated {len(text) - OUTPUT_LIMIT} chars]"


def run_tool(name, args):
    """Returns (output, is_error)."""
    try:
        if name == "bash":
            proc = subprocess.run(
                ["bash", "-lc", args["command"]],
                cwd=WORKSPACE,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            out = proc.stdout + proc.stderr
            return truncate(f"{out}\n[exit code {proc.returncode}]"), proc.returncode != 0
        if name == "read_file":
            return truncate((WORKSPACE / args["path"]).read_text()), False
        if name == "write_file":
            path = WORKSPACE / args["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"])
            return f"wrote {len(args['content'])} chars to {args['path']}", False
        return f"unknown tool: {name}", True
    except subprocess.TimeoutExpired:
        return "command timed out after 120s", True
    except (OSError, KeyError, UnicodeDecodeError) as exc:
        return f"{type(exc).__name__}: {exc}", True


def main():
    messages = [{"role": "user", "content": PROMPT.read_text()}]
    for _ in range(MAX_TURNS):
        resp = call_api(messages)
        content = resp.get("content", [])
        messages.append({"role": "assistant", "content": content})
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if not tool_uses:
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            log(type="final", text=text)
            print(text)
            return 0
        results = []
        for use in tool_uses:
            log(type="tool_call", id=use["id"], name=use["name"], arguments=use["input"])
            output, is_error = run_tool(use["name"], use["input"])
            log(type="tool_result", id=use["id"], output=output, is_error=is_error)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": use["id"],
                    "content": output,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})
    log(type="error", message=f"turn limit ({MAX_TURNS}) reached")
    return 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fatal as exc:
        log(type="error", message=str(exc))
        print(f"fatal: {exc}", file=sys.stderr)
        sys.exit(2)
