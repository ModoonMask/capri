"""Context assembly: keeps prompts inside budget while preserving continuity."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def history_path(data_dir: Path, session: str = "main") -> Path:
    return Path(data_dir) / "memory" / f"conversation-{session}.jsonl"


class ConversationLog:
    """Persistent conversation history, stored as JSONL."""

    def __init__(self, data_dir: Path, session: str = "main"):
        self.path = history_path(data_dir, session)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                if obj.get("role") in ("user", "assistant") and obj.get("content"):
                    self.entries.append(obj)
            except (json.JSONDecodeError, TypeError):
                continue

    def append(self, role: str, content: str) -> None:
        entry = {"role": role, "content": content, "ts": round(time.time(), 3)}
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def window(self, max_messages: int) -> list[dict[str, str]]:
        return [{"role": e["role"], "content": e["content"]}
                for e in self.entries[-max_messages:]]

    def last_exchange(self) -> tuple[str, str] | None:
        """(last user msg, last assistant msg) for the extraction pass."""
        last_user = last_assistant = None
        for e in reversed(self.entries):
            if last_assistant is None and e["role"] == "assistant":
                last_assistant = e["content"]
            elif last_user is None and e["role"] == "user":
                last_user = e["content"]
            if last_user and last_assistant:
                return last_user, last_assistant
        return None

    def minutes_since_last_user(self) -> float | None:
        if not self.entries:
            return None
        last_ts = self.entries[-1].get("ts", 0)
        return max(0.0, (time.time() - last_ts) / 60.0)


def build_messages(personality_prompt: str, history: list[dict[str, str]]) -> list:
    from .providers import system as sys_msg
    return [sys_msg(personality_prompt), *history]
