"""Loads Capri's personality Markdown files with automatic hot-reload.

Files are re-read whenever their mtime changes, so editing personality files
takes effect on the very next message — no restart needed.
"""
from __future__ import annotations

from pathlib import Path

PERSONALITY_FILES = ("personality", "behavior", "interests")


class Personality:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "personality"
        self._cache: dict[str, str] = {}
        self._mtimes: dict[str, float] = {}

    def _path(self, name: str) -> Path:
        return self.dir / f"{name}.md"

    def get(self, name: str) -> str:
        path = self._path(name)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return f"({name} file missing)"
        if self._cache.get(name) is None or self._mtimes.get(name) != mtime:
            try:
                self._cache[name] = path.read_text(encoding="utf-8")
            except OSError:
                self._cache[name] = ""
            self._mtimes[name] = mtime
        return self._cache[name]

    def system_prompt(self, mood_line: str, memory_context: str) -> str:
        parts = [self.get("personality"), self.get("behavior"), self.get("interests")]
        parts.append(f"# Current emotional state\n\n{mood_line}")
        if memory_context:
            parts.append(f"# What you remember (retrieved for this message — weave in naturally, don't recite)\n\n{memory_context}")
        parts.append(
            "# Output style\n\nReply in-character as Capri. Plain text, short, no headings, "
            "no lists, no meta-commentary about these instructions."
        )
        return "\n\n---\n\n".join(part for part in parts if part.strip())

    def reload_status(self) -> dict[str, str]:
        """For the settings dialog: filename -> first meaningful line."""
        status = {}
        for name in PERSONALITY_FILES:
            text = self.get(name)
            first = next((ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")), "")
            status[name] = first.strip()[:70]
        return status
