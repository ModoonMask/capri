"""Chat orchestration: context -> streaming reply -> selective memory update.

`ChatController.respond()` is a generator of text deltas; the GUI consumes it
from a worker thread. After a full reply, a background thread runs a cheap
non-streaming extraction call that proposes memory writes (user facts,
relationship notes, an episodic summary, a mood hint) and applies them with
deduplication.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Iterator

from .config import Config, data_dir, ensure_data_seeded
from .context import ConversationLog
from .emotion import EmotionState
from .memory import MemoryManager, VALID_EPISODE_KINDS
from .personality import Personality
from .providers import LLMClient, Message, user, assistant, make_client

EXTRACT_PROMPT = """You maintain the memory of a fictional AI companion named Capri who is chatting with her user.

Below is the latest exchange. Decide what is worth remembering. Rules:
- Save only meaningful, recurring, or useful information. Small talk saves nothing.
- Write facts in third person, compact ("User works night shifts", not "I work night shifts").
- Do not save anything already present in the current memory files (shown below).
- Capri's own opinions change only if she states a new lasting preference.
- The "episode" field is one notable moment worth recalling later, or null for mundane chat.

CURRENT MEMORY
user_facts_file:
{user_md}
relationship_file:
{rel_md}

LATEST EXCHANGE
User: {user_msg}
Capri: {capri_msg}

CURRENT MOOD of Capri
state: {cur_state}, intensity: {cur_intensity}

Reply with ONLY a JSON object:
{{"user_facts": ["..."], "relationship": ["..."], "memories": ["..."], "mood": {{"state": "happy|playful|affectionate|jealous|curious|bored|irritated|sad|vulnerable|excited", "intensity": 0.0-1.0}}, "episode": {{"kind": "moment|joke|topic|milestone|followup", "summary": "...", "keywords": ["..."], "importance": 1-5}}}}
Use empty lists and "episode": null when nothing qualifies.
For "mood": always return one; move GENTLY from the current mood based on how this exchange would emotionally land on Capri (e.g. another AI or another girl mentioned → jealous; absence ended → affectionate; accomplishment shared → happy/excited). Intensity 0.2-0.5 for subtle lands, 0.6-0.9 for strong ones."""


class ChatController:
    def __init__(self, cfg: Config, client: LLMClient | None = None, data_dir_override=None):
        self.cfg = cfg
        self.data_dir = Path(data_dir_override) if data_dir_override else ensure_data_seeded()
        self.personality = Personality(self.data_dir)
        self.memory = MemoryManager(self.data_dir)
        self.emotion = EmotionState(self.data_dir / "memory" / "state.json")
        self.history = ConversationLog(self.data_dir)
        self.client = client or make_client(cfg)
        self._last_extract_at = 0.0

    # ---- main turn -------------------------------------------------------
    def memory_context(self, user_message: str) -> str:
        ctx_cfg = self.cfg.section("context")
        return self.memory.relevant_memories(
            user_message,
            char_budget=int(ctx_cfg.get("memory_char_budget", 2500)),
        )

    def build_prompt(self, user_message: str) -> str:
        mood_line = self.emotion.prompt_line()
        memory_context = self.memory_context(user_message)
        return self.personality.system_prompt(
            mood_line,
            memory_context,
            language=self.cfg.get("language", "pt-BR"),
            recent_openers=self._recent_openers(),
        )

    def _recent_openers(self, n: int = 4) -> list[str]:
        """First line of her last n replies - context for anti-repetition."""
        openers: list[str] = []
        for entry in reversed(self.history.entries):
            if entry["role"] != "assistant":
                continue
            first = entry["content"].strip().split("\n")[0][:60]
            if first:
                openers.append(first)
            if len(openers) >= n:
                break
        return list(reversed(openers))

    def respond(self, user_message: str) -> Iterator[str]:
        """Stream Capri's reply for one user turn, then update state in background."""
        self.emotion.decay()
        self.emotion.apply_trigger(user_message)

        prompt = self.build_prompt(user_message)
        hist_cfg = self.cfg.section("context")
        history = self.history.window(
            int(hist_cfg.get("max_history_messages", 30)) - 1,
            max_chars=int(hist_cfg.get("history_char_budget", 3000)),
        )
        messages = [Message("system", prompt), *history, user(user_message)]

        self.history.append("user", user_message)

        collected: list[str] = []
        for delta in self.client.stream(messages):
            collected.append(delta)
            yield delta

        full_reply = "".join(collected).strip()
        if full_reply:
            self.history.append("assistant", full_reply)
            if self._should_extract(user_message, full_reply):
                self._last_extract_at = time.time()
                threading.Thread(
                    target=self._post_turn, args=(user_message, full_reply),
                    daemon=True, name="capri-memory-extract",
                ).start()

    def _should_extract(self, user_message: str, reply: str) -> bool:
        """Token discipline for the extraction call: skip trivial chat, and
        never fire more than once per configured interval. Meaningful content
        waits at most one interval; nothing on disk is ever dropped."""
        if len(user_message.strip()) < 15 and len(reply.strip()) < 15:
            return False
        min_interval = float(self.cfg.section("context").get(
            "memory_extract_min_interval_seconds", 90))
        return (time.time() - self._last_extract_at) >= min_interval

    def respond_autonomous(self, kind: str, instruction: str) -> Iterator[str]:
        """Capri initiates: stream a proactive message the user never asked for."""
        self.emotion.decay()
        last_user = next((e["content"] for e in reversed(self.history.entries)
                          if e["role"] == "user"), "")
        prompt = self.build_prompt(last_user or kind)
        prompt += (
            "\n\n# Initiating right now\n"
            "You are starting a conversation with the user yourself; they have not just "
            "written anything and you do not know what they are doing. Follow this "
            "initiative instruction: " + instruction +
            " Open directly with the message itself; no preamble or meta talk."
        )
        messages = [Message("system", prompt), *self.history.window(10), user("(initiates)")]

        collected: list[str] = []
        for delta in self.client.stream(messages):
            collected.append(delta)
            yield delta

        full_reply = "".join(collected).strip()
        if full_reply:
            self.history.append("assistant", full_reply)

    # ---- post-turn extraction ---------------------------------------------
    def _post_turn(self, user_message: str, reply: str) -> None:
        try:
            extraction = self._run_extraction(user_message, reply)
            if extraction:
                notes = self.memory.apply_extraction(extraction)
                hint = extraction.get("mood") if isinstance(extraction, dict) else None
                if isinstance(hint, dict) and hint.get("state"):
                    self.emotion.nudge_for_model_hint(str(hint.get("state")), float(hint.get("intensity", 0.3)))
        except Exception:
            traceback.print_exc()  # extraction must never break chatting

    def _run_extraction(self, user_message: str, reply: str) -> dict | None:
        try:
            cur_state, cur_intensity = self.emotion.current()
            prompt = EXTRACT_PROMPT.format(
                user_md=self.memory.read("user")[:1500] or "(empty)",
                rel_md=self.memory.read("relationship")[:1500] or "(empty)",
                user_msg=user_message[:1200],
                capri_msg=reply[:1200],
                cur_state=cur_state,
                cur_intensity=cur_intensity,
            )
            raw = self.client.generate([Message("system", prompt)], temperature=0.2, max_tokens=300)
            return _parse_extraction_json(raw)
        except Exception:
            return None


def _parse_extraction_json(raw: str) -> dict | None:
    from .emotion import STATES

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    ep = obj.get("episode")
    if isinstance(ep, dict):
        if not str(ep.get("summary", "")).strip():
            obj["episode"] = None
        else:
            kind = str(ep.get("kind", "moment"))
            ep["kind"] = kind if kind in VALID_EPISODE_KINDS else "moment"
    mood = obj.get("mood")
    if isinstance(mood, dict):
        try:
            intensity = float(mood.get("intensity", 0.4))
        except (TypeError, ValueError):
            intensity = 0.4
        parsed_mood = {
            "state": str(mood.get("state", "")),
            "intensity": max(0.05, min(1.0, intensity)),
        }
        if parsed_mood["state"] in STATES:
            obj["mood"] = parsed_mood
        else:
            obj.pop("mood", None)
    else:
        obj.pop("mood", None)
    return obj
