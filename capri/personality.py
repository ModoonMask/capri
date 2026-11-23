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

    def system_prompt(self, mood_line: str, memory_context: str,
                      language: str = "pt-BR",
                      recent_openers: list[str] | None = None) -> str:
        parts = [self.get("personality"), self.get("behavior"), self.get("interests")]
        parts.append(f"# Current emotional state\n\n{mood_line}")
        if memory_context:
            parts.append(f"# What you remember (retrieved for this message — weave in naturally, don't recite)\n\n{memory_context}")
        if recent_openers:
            listed = "\n".join(f"- {op}" for op in recent_openers)
            parts.append(
                "# Suas últimas aberturas (NÃO comece a resposta com nada parecido com isto)\n\n"
                + listed
            )
        parts.append(
            "# Como responder\n\n"
            f"Idioma: escreva SEMPRE em {language}, coloquial e natural, mesmo que o "
            "usuário escreva em outro idioma. Nunca se reapresente nem explique quem "
            "você é (regras completas de formato e anti-repetição estão em 'Regras de "
            "conversa' — valem)."
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
