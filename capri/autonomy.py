"""Autonomous initiative: Capri occasionally starts conversations herself.

A background thread wakes up, checks gating conditions (enabled, user idle,
cooldown expired, not in quiet hours), picks a random interval within the
configured range, and fires a callback. The callback is responsible for
generating and displaying the proactive message.

Anti-spam guarantees:
- fire only after `idle_minutes_before_proactive` of user silence
- hard cooldown between proactive messages
- optional quiet hours (hour-of-day, wrapped at midnight)
- can be disabled entirely from config or the settings panel
"""
from __future__ import annotations

import random
import threading
import time
from typing import Callable

CHECK_INTERVAL_SEC = 30.0

SPARK_KINDS = (
    "followup",     # ask how something previously discussed turned out
    "memory",       # bring up a past moment
    "thought",      # share her own thought/opinion
    "affection",    # small unprompted warmth
)

SPARK_PROMPTS = {
    "followup": (
        "Pick one open thread from your episodic memories (something the user said they "
        "would do, was playing, was excited about, or was worried about) and ask how it "
        "turned out — naturally, in one or two short sentences. If nothing fits, say nothing "
        "like 'never mind': instead pick the 'thought' behavior."
    ),
    "memory": (
        "Bring up one specific past moment between you two that you find genuinely "
        "interesting or sweet, with a short personal comment. One or two sentences."
    ),
    "thought": (
        "Share a small, concrete thought of your own right now: an opinion, something "
        "about your day as a program, something you noticed about this conversation's "
        "topics. One or two sentences, no question attached necessarily."
    ),
    "affection": (
        "A short unprompted moment of affection, specific and understated, not a template "
        "compliment. You can tease the user's current topics instead."
    ),
}


class AutonomyScheduler:
    def __init__(self, cfg, on_proactive: Callable[[str, str], None],
                 minutes_since_last_activity: Callable[[], float]):
        """`on_proactive(kind, instruction)` is called from the scheduler thread.

        `minutes_since_last_activity()` must report user inactivity in minutes.
        """
        self.cfg = cfg
        self.on_proactive = on_proactive
        self.minutes_since_last_activity = minutes_since_last_activity
        self._stop = threading.Event()
        self._last_proactive: float = 0.0
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="capri-autonomy")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def nudge(self) -> None:
        """Re-evaluate soon (e.g., after settings change)."""
        self._wake.set()

    # ---- gating ------------------------------------------------------------
    def should_fire(self, now: float | None = None) -> bool:
        a = self.cfg.section("autonomy")
        if not a.get("enabled", True):
            return False
        now = now if now is not None else time.time()
        cooldown_min = float(a.get("cooldown_minutes", 30))
        if self._last_proactive and (now - self._last_proactive) < cooldown_min * 60:
            return False
        idle_min = self.minutes_since_last_activity()
        required_idle = float(a.get("idle_minutes_before_proactive", 20))
        if idle_min is None or idle_min < required_idle:
            return False
        if self._in_quiet_hours(now):
            return False
        return True

    def _in_quiet_hours(self, now: float) -> bool:
        quiet = self.cfg.section("autonomy").get("quiet_hours", [23, 8])
        try:
            start, end = int(quiet[0]) % 24, int(quiet[1]) % 24
        except (TypeError, ValueError, IndexError):
            return False
        if start == end:
            return False
        hour = time.localtime(now).tm_hour
        return start < hour < end or (start > end and (hour >= start or hour < end))

    # ---- loop ----------------------------------------------------------------
    def _run(self) -> None:
        self._last_proactive = time.time()   # grace period on startup
        next_fire = self._next_interval()
        while not self._stop.is_set():
            self._wake.wait(timeout=CHECK_INTERVAL_SEC)
            self._wake.clear()
            if self._stop.is_set():
                return
            now = time.time()
            if now >= next_fire and self.should_fire(now):
                self._last_proactive = now
                next_fire = self._next_interval()
                try:
                    kind = random.choice(SPARK_KINDS)
                    self.on_proactive(kind, SPARK_PROMPTS[kind])
                except Exception:
                    pass  # never let proactivity kill the app
            elif now >= next_fire:
                next_fire = now + 60.0  # gated; re-check in a minute

    def _next_interval(self) -> float:
        a = self.cfg.section("autonomy")
        lo = max(5.0, float(a.get("min_interval_minutes", 45)))
        hi = max(lo, float(a.get("max_interval_minutes", 150)))
        return time.time() + random.uniform(lo, hi) * 60.0
