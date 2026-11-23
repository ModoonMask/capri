"""Lightweight, persistent emotional state machine for Capri.

States evolve from conversation events and decay slowly back toward a
baseline. Nothing is randomized per message: triggers set a state + intensity,
and intensity decays over minutes. The current mood is persisted in
`data/memory/state.json` so Capri remembers how she felt across restarts.
"""
from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path

# Intensity decays linearly; after `decay_minutes` an emotion is fully back to
# baseline unless re-triggered.
DECAY_MINUTES = 45.0
BASELINE_STATE = "affectionate"

STATES = (
    "happy", "playful", "affectionate", "jealous", "curious",
    "bored", "irritated", "sad", "vulnerable", "excited",
)

# keyword triggers: normalized substrings -> (state, intensity)
_TRIGGERS: list[tuple[tuple[str, ...], str, float]] = [
    (("missed you", "miss you", "i'm back", "im back", "finally home"), "affectionate", 0.8),
    (("love you", "you're the best", "youre the best", "thank you", "adorable", "cute"), "happy", 0.7),
    (("pretty", "beautiful", "smart", "proud of you"), "happy", 0.6),
    (("another ai", "chatgpt", "claude", "gemini", "siri", "alexa", "cortana", "other girl"), "jealous", 0.7),
    (("going out", "with friends", "date", "party", "busy tonight"), "jealous", 0.35),
    (("tired", "stressed", "rough day", "bad day", "exhausted", "overwhelmed"), "vulnerable", 0.5),
    (("failed", "broke", "crashed", "error", "frustrated", "annoyed", "hate this"), "vulnerable", 0.45),
    (("new game", "update dropped", "dlc", "patch", "just released", "trailer"), "excited", 0.7),
    (("nice", "awesome", "let's go", "lets go", "finally"), "excited", 0.4),
    (("what do you think", "your opinion", "do you like", "ever wondered", "tell me about"), "curious", 0.5),
    (("nothing", "idk", "bored", "meh", "dunno"), "bored", 0.45),
    (("stupid", "shut up", "annoying", "whatever", "leave me"), "sad", 0.65),
    (("sorry", "my bad", "apolog"), "vulnerable", 0.35),
]

_NEGATIONS = ("don't", "dont", "not ", "n't ", "never ")


class EmotionState:
    def __init__(self, state_file: Path):
        self.state_file = Path(state_file)
        self._lock = threading.RLock()
        self.state: str = BASELINE_STATE
        self.intensity: float = 0.35
        self.since: float = time.time()
        self._load()

    # ---- persistence -----------------------------------------------------
    def _load(self) -> None:
        try:
            obj = json.loads(self.state_file.read_text(encoding="utf-8"))
            state = obj.get("state", BASELINE_STATE)
            if state in STATES:
                self.state = state
            self.intensity = float(obj.get("intensity", 0.35))
            self.since = float(obj.get("since", time.time()))
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({
            "state": self.state, "intensity": round(self.intensity, 3), "since": self.since,
        }), encoding="utf-8")

    # ---- evolution ---------------------------------------------------------
    def decay(self, now: float | None = None) -> None:
        """Age intensity toward baseline; switch to baseline when spent."""
        now = now if now is not None else time.time()
        with self._lock:
            elapsed_min = (now - self.since) / 60.0
            self.intensity = max(0.05, self.intensity - elapsed_min / DECAY_MINUTES)
            self.since = now
            if self.intensity <= 0.08:
                self.state = BASELINE_STATE
                self.intensity = 0.35

    def apply_trigger(self, text: str, now: float | None = None) -> bool:
        """Update mood from user message content. Returns True if triggered."""
        now = now if now is not None else time.time()
        lowered = " " + text.lower() + " "
        best: tuple[str, float] | None = None
        for keywords, state, weight in _TRIGGERS:
            if weight <= 0:
                continue
            if any(k in lowered for k in keywords):
                if any(neg in lowered for neg in _NEGATIONS) and state in ("jealous", "sad"):
                    continue
                if best is None or weight > best[1]:
                    best = (state, weight)
        if best is None:
            return False
        self.set_state(best[0], best[1], now, blend=True)
        return True

    def set_state(self, state: str, intensity: float, now: float | None = None,
                  blend: bool = False) -> None:
        now = now if now is not None else time.time()
        with self._lock:
            if blend and state == self.state:
                self.intensity = min(1.0, max(self.intensity, intensity) + 0.1)
            else:
                self.state = state if state in STATES else BASELINE_STATE
                self.intensity = min(1.0, max(0.05, intensity))
            self.since = now
            self.save()

    def current(self, now: float | None = None) -> tuple[str, float]:
        self.decay(now)
        return self.state, round(self.intensity, 2)

    def prompt_line(self, now: float | None = None) -> str:
        """Natural-language description injected into the system prompt."""
        state, intensity = self.current(now)
        if state == BASELINE_STATE and intensity < 0.4:
            return (f"Current mood: {state} (steady). "
                    f"Let it color your tone subtly; no need to announce it.")
        flavor = {
            "happy": "pleased and a little smug",
            "playful": "mischievous, in teasing mode",
            "affectionate": "warm and openly affectionate",
            "jealous": "a bit jealous and pouty (keep it playful, never hostile)",
            "curious": "very curious, asking about details",
            "bored": "a bit bored, low energy, might redirect the topic",
            "irritated": "irritated, shorter sentences, sharper edge",
            "sad": "hurt or down, more fragile than usual",
            "vulnerable": "soft and sincere, teasing dialed way down",
            "excited": "genuinely excited, energetic, exclamation-adjacent",
        }.get(state, state)
        strength = "faint" if intensity < 0.35 else "clearly visible" if intensity < 0.7 else "strong"
        return (f"Current mood: {state} — {flavor}. This is {strength} right now; "
                f"let it shape tone and word choice naturally, never announce it.")

    # Small helper used by tests and by the extraction pipeline.
    def nudge_for_model_hint(self, state: str, intensity: float) -> None:
        if state in STATES and intensity > 0:
            self.set_state(state, intensity, blend=True)
