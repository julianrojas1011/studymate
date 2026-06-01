"""
Conversation memory — short-term list + long-term JSON persistence.

Two-layer design:
  - Short-term: a plain Python list that grows during the session.
  - Long-term:  a JSON file written at the end of each exchange.

System prompts and few-shot examples are NOT stored here.  They live in
prompts.py and are injected fresh on every session start.  This means we
can improve the prompt wording or add new examples without touching or
migrating any saved history file.

Atomic writes: we write to a .tmp file then atomically rename it over the
real file.  os.replace() is atomic on POSIX and transactional on Windows,
so a process killed mid-write leaves the previous good file intact rather
than a half-written, unparseable JSON document.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

MEMORY_FILE = Path(os.environ.get("STUDYMATE_MEMORY_FILE", "memory.json"))

# Keys we persist for each message.  We drop None values and anything else
# (e.g. temporary rendering hints) so the file stays lean and portable.
_KEEP_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serialise(msg: dict) -> dict:
    """Return a clean copy of msg containing only the keys we persist."""
    return {k: v for k, v in msg.items() if k in _KEEP_KEYS and v is not None}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _extract_text(content: object) -> str:
    """Best-effort extraction of a readable string from any content shape.

    content can be:
      - a plain string  (most user and assistant messages)
      - a list of blocks  (assistant tool-call turns, tool-result user turns)
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            # Text blocks from assistant turns
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            # tool_result blocks from user turns — show the result value
            elif block.get("type") == "tool_result":
                parts.append(f"[tool result: {block.get('content', '')}]")
            # tool_use blocks — show what was called
            elif block.get("type") == "tool_use":
                parts.append(f"[called {block.get('name', 'tool')}]")
        return " ".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Memory class
# ---------------------------------------------------------------------------

class Memory:
    """Manages the conversation history for one StudyMate session."""

    def __init__(self, path: Path = MEMORY_FILE) -> None:
        self.path = path
        self._history: list[dict] = []
        self.created_at: str = _now_iso()

    # ------------------------------------------------------------------
    # Core list operations
    # ------------------------------------------------------------------

    def append(self, message: dict) -> None:
        """Serialise and append a message to the in-memory history."""
        self._history.append(_serialise(message))

    def messages(self) -> list[dict]:
        """Return a shallow copy of the history list.

        Callers get a copy so they can prepend system prompts or few-shot
        examples without mutating the stored history.
        """
        return list(self._history)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary_text(self, limit: int = 8) -> str:
        """Return a plain-text recap of the last *limit* user/assistant turns.

        Tool messages are skipped — they're noise for a conversational recap.
        Content is truncated at 240 characters so the summary stays compact.
        """
        turns = [m for m in self._history if m.get("role") in ("user", "assistant")]
        recent = turns[-limit:]

        if not recent:
            return "No prior conversation recorded."

        lines: list[str] = []
        for msg in recent:
            role = msg["role"]
            raw = _extract_text(msg.get("content", ""))
            snippet = raw[:240] + "…" if len(raw) > 240 else raw
            label = "You" if role == "user" else "StudyMate"
            lines.append(f"- {label}: {snippet}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load history from disk.

        Failures are handled gracefully so a missing or corrupt file never
        crashes the REPL at startup — the session simply starts fresh.
        """
        try:
            text = self.path.read_text(encoding="utf-8")
            data = json.loads(text)
            self._history = data.get("history", [])
            self.created_at = data.get("created_at", _now_iso())
        except FileNotFoundError:
            # First run — perfectly normal.
            pass
        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupt file.  Start fresh rather than crashing.
            self._history = []
            self.created_at = _now_iso()

    def save(self) -> None:
        """Persist history to disk using an atomic write.

        We write to <path>.tmp first, then call os.replace() to swap it
        into place.  os.replace() is atomic on POSIX (rename syscall) and
        transactional on Windows, so the real file is never in a partial
        state even if the process is killed during the write.
        """
        payload = {
            "created_at": self.created_at,
            "saved_at": _now_iso(),
            "history": self._history,
        }
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def clear(self) -> None:
        """Wipe in-memory history, reset the creation timestamp, and delete the file."""
        self._history = []
        self.created_at = _now_iso()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
