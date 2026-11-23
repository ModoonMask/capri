"""Persistent multi-category memory system for Capri.

Storage layout (all under the data dir, survives restarts):

    data/memory/user.md          long-term facts about the user
    data/memory/relationship.md  evolving state of the relationship
    data/memory/memories.md      important long-term memories / running jokes
    data/memory/episodes.jsonl   one JSON object per notable moment (episodic log)
    data/memory/state.json       emotional state persistence

Memory is *not* dumped wholesale into every prompt: `relevant_context()`
scores memories by keyword overlap, recency, and importance, and returns only
the most relevant lines within a character budget.

New memories are written *selectively*: after each turn a small LLM extraction
(delegated to ChatExtract by the caller) proposes facts; `apply_extraction`
deduplicates and merges them.
"""
from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

STOPWORDS = frozenset(
    "a an the and or but if then so of to in on at for with my your her his their "
    "i you he she it we they me him them this that these those is are was were be "
    "been being do does did have has had will would can could should about from "
    "not no yes ok okay hey hi yeah".split()
)

TOKEN_RE = re.compile(r"[\w'+#]+", re.UNICODE)


def tokenize(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return {t for t in TOKEN_RE.findall(text) if t not in STOPWORDS and len(t) > 1}


VALID_EPISODE_KINDS = {"moment", "joke", "topic", "milestone", "followup"}


@dataclass
class Episode:
    """One notable moment from a conversation."""

    ts: float
    kind: str          # "moment" | "joke" | "topic" | "milestone" | "followup"
    summary: str       # one sentence, written in third person
    keywords: list[str]
    importance: int    # 1..5

    def to_json(self) -> str:
        return json.dumps(
            {"ts": self.ts, "kind": self.kind, "summary": self.summary,
             "keywords": self.keywords, "importance": self.importance},
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(line: str) -> "Episode | None":
        try:
            obj = json.loads(line)
            return Episode(
                ts=float(obj["ts"]), kind=str(obj.get("kind", "moment")),
                summary=str(obj.get("summary", "")), keywords=list(obj.get("keywords", [])),
                importance=int(obj.get("importance", 2)),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None


class MemoryManager:
    """Thread-safe store of Capri's persistent memory."""

    MAX_EPISODES = 500

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "memory"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._episodes_path = self.dir / "episodes.jsonl"
        self._episodes_cache: list[Episode] | None = None
        self._mtimes: dict[str, float] = {}
        self._texts: dict[str, str] = {}

    # ---- raw file access -------------------------------------------------
    def _path(self, name: str) -> Path:
        names = {
            "user": "user.md",
            "relationship": "relationship.md",
            "memories": "memories.md",
        }
        return self.dir / names[name]

    def read(self, name: str) -> str:
        with self._lock:
            path = self._path(name)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            if name not in self._texts or self._mtimes.get(name) != mtime:
                try:
                    self._texts[name] = path.read_text(encoding="utf-8")
                except OSError:
                    self._texts[name] = ""
                self._mtimes[name] = mtime
            return self._texts[name]

    def write(self, name: str, content: str) -> None:
        with self._lock:
            path = self._path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            self._texts[name] = content
            self._mtimes[name] = path.stat().st_mtime

    def append(self, name: str, text: str) -> None:
        block = text.strip()
        if not block:
            return
        current = self.read(name).rstrip("\n")
        if current.strip():
            separator = "\n" if name == "user" else "\n\n"
            self.write(name, current + separator + block + "\n")
        else:
            self.write(name, block + "\n")

    # ---- episodic memory ---------------------------------------------------
    def add_episode(self, ep: Episode) -> None:
        with self._lock:
            with self._episodes_path.open("a", encoding="utf-8") as f:
                f.write(ep.to_json() + "\n")
            self._episodes_cache = None

    def episodes(self) -> list[Episode]:
        with self._lock:
            if self._episodes_cache is None:
                eps: list[Episode] = []
                if self._episodes_path.exists():
                    for line in self._episodes_path.read_text(encoding="utf-8").splitlines():
                        ep = Episode.from_json(line)
                        if ep:
                            eps.append(ep)
                self._episodes_cache = eps
            return list(self._episodes_cache)

    # ---- retrieval ----------------------------------------------------------
    def relevant_memories(self, user_message: str, char_budget: int = 2500,
                          now: float | None = None) -> str:
        """Score and assemble the most relevant memory context for a turn."""
        now = now if now is not None else time.time()
        sections: list[tuple[str, str]] = []

        user_md = self.read("user").strip()
        if user_md:
            sections.append(("About the user", _strip_heading(user_md)))

        rel_md = self.read("relationship").strip()
        if rel_md:
            sections.append(("Relationship", _tail(_strip_heading(rel_md), 1200)))

        core_md = self.read("memories").strip()
        if core_md:
            sections.append(("Long-term memories", _tail(_strip_heading(core_md), 1200)))

        query_tokens = tokenize(user_message)
        scored = self._score_episodes(query_tokens, user_message, now)
        picked: list[Episode] = []
        used = sum(len(s) + 30 for _, s in sections)
        for ep in scored:
            block = f"- ({ep.kind}) {ep.summary}"
            if used + len(block) > char_budget:
                break
            picked.append(ep)
            used += len(block)
        if picked:
            sections.append(("Recent relevant moments", "\n".join(f"- {e.summary}" for e in picked)))

        rendered = "\n\n".join(f"### {title}\n{body}" for title, body in sections if body.strip())
        return rendered.strip()

    def _score_episodes(self, query_tokens: set[str], raw_message: str, now: float) -> list[Episode]:
        eps = self.episodes()
        if not eps:
            return []
        scored: list[tuple[float, Episode]] = []
        for ep in eps:
            ep_tokens = set(ep.keywords) | tokenize(ep.summary)
            overlap = len(query_tokens & ep_tokens)
            score = overlap * 2.0
            age_days = max(0.0, (now - ep.ts) / 86400.0)
            score += max(0.0, 2.0 - age_days / 7.0)          # recency wave
            score += 0.5 * ep.importance                      # importance boost
            if overlap == 0 and age_days > 21:
                continue  # stale and unrelated: don't pad the prompt
            scored.append((score, ep))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].ts))
        return [ep for _, ep in scored[:12]]

    # ---- selective updating -------------------------------------------------
    def apply_extraction(self, extraction: dict[str, Any]) -> list[str]:
        """Merge facts proposed by the post-turn extraction into the memory files.

        Returns human-readable notes about what was stored (for debug).
        """
        notes: list[str] = []
        with self._lock:
            facts = extraction.get("user_facts") or []
            if isinstance(facts, str):
                facts = [facts]
            for fact in facts[:4]:
                if _add_unique_line(self.read("user"), self._path("user"), str(fact)):
                    notes.append(f"user: {fact}")
                    self._texts.pop("user", None)

            rel_facts = extraction.get("relationship") or []
            if isinstance(rel_facts, str):
                rel_facts = [rel_facts]
            for fact in rel_facts[:3]:
                self.append("relationship", str(fact))
                notes.append(f"relationship: {fact}")

            memories = extraction.get("memories") or []
            if isinstance(memories, str):
                memories = [memories]
            for fact in memories[:3]:
                self.append("memories", str(fact))
                notes.append(f"memory: {fact}")

            ep = extraction.get("episode")
            if isinstance(ep, dict) and str(ep.get("summary", "")).strip():
                kind = str(ep.get("kind", "moment"))
                if kind not in VALID_EPISODE_KINDS:
                    kind = "moment"
                try:
                    importance = int(ep.get("importance", 2))
                except (TypeError, ValueError):
                    importance = 2
                episode = Episode(
                    ts=time.time(),
                    kind=kind,
                    summary=str(ep["summary"])[:300],
                    keywords=[str(k)[:30] for k in (ep.get("keywords") or [])][:8],
                    importance=max(1, min(5, importance)),
                )
                self.add_episode(episode)
                notes.append(f"episode: {episode.summary}")
        return notes


def _add_unique_line(current_text: str, path: Path, fact: str) -> bool:
    fact = re.sub(r"\s+", " ", str(fact).strip().rstrip("."))
    if not fact:
        return False
    norm = fact.lower()
    if any(norm in line.lower() or line.lower() in norm
           for line in current_text.splitlines() if line.strip() and not line.startswith("#")):
        return False
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    return True


def _strip_heading(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#") and not ln.strip().startswith("_")]
    return "\n".join(lines).strip()


def _tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return "…\n" + text[-max_chars:]


def dedupe_facts(facts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for fact in facts:
        key = " ".join(sorted(tokenize(str(fact))))
        if key and key not in seen:
            seen.add(key)
            out.append(str(fact).strip())
    return out
