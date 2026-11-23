"""Tests for Capri's core (non-GUI) systems. Run: pytest tests/ -q"""
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capri.config import Config, DEFAULT_CONFIG, PROVIDER_PRESETS
from capri.emotion import EmotionState
from capri.memory import MemoryManager, Episode, tokenize
from capri.context import ConversationLog
from capri.autonomy import AutonomyScheduler
from capri.providers import LLMClient, ProviderError, Message
from capri.chat import ChatController, _parse_extraction_json


@pytest.fixture()
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPRI_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("CAPRI_HOME", str(tmp_path / "home"))
    return tmp_path


def make_config(tmp_home, provider="groq", key="test-key-123") -> Config:
    cfg = Config.load()
    cfg.set("provider", provider)
    cfg.data["api_keys"] = {provider: key}
    return cfg


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_defaults_and_merge(self, tmp_home):
        cfg = Config.load()
        assert cfg.provider in PROVIDER_PRESETS
        assert cfg.get("temperature") == DEFAULT_CONFIG["temperature"]

    def test_save_and_reload(self, tmp_home):
        cfg = Config.load()
        cfg.set("temperature", 1.23)
        cfg.save()
        reloaded = Config.load()
        assert reloaded.get("temperature") == 1.23

    def test_env_key_wins_over_disk(self, tmp_home, monkeypatch):
        cfg = make_config(tmp_home)
        monkeypatch.setenv("GROQ_API_KEY", "env-key")
        assert cfg.api_key() == "env-key"

    def test_disk_key_fallback(self, tmp_home, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        cfg = make_config(tmp_home)
        assert cfg.api_key() == "test-key-123"

    def test_unknown_provider_falls_back(self, tmp_home, monkeypatch):
        cfg = Config.load()
        cfg.set("provider", "does-not-exist")
        assert cfg.provider == "groq"


# ---------------------------------------------------------------------------
# emotion
# ---------------------------------------------------------------------------
class TestEmotion:
    def test_compliment_makes_happy(self, tmp_home):
        state = EmotionState(tmp_home / "state.json")
        assert state.apply_trigger("you are so cute, capri")
        s, i = state.current()
        assert s == "happy"

    def test_other_ai_triggers_jealousy(self, tmp_home):
        state = EmotionState(tmp_home / "state.json")
        assert state.apply_trigger("gemini answered this for me")
        assert state.current()[0] == "jealous"

    def test_negated_jealousy_ignored(self, tmp_home):
        state = EmotionState(tmp_home / "state.json")
        state.set_state("affectionate", 0.3)
        state.apply_trigger("I don't use chatgpt at all")
        assert state.current()[0] != "jealous"

    def test_decays_back_to_baseline(self, tmp_home):
        state = EmotionState(tmp_home / "state.json")
        state.set_state("excited", 0.9)
        future = time.time() + 10 * 3600
        assert state.current(future)[0] == "affectionate"  # baseline

    def test_persists_across_restart(self, tmp_home):
        state = EmotionState(tmp_home / "state.json")
        state.set_state("jealous", 0.8)
        state2 = EmotionState(tmp_home / "state.json",)
        assert state2.state == "jealous"


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------
class TestMemory:
    def test_append_and_read(self, tmp_home):
        mem = MemoryManager(tmp_home / "data")
        mem.append("user", "loves roguelikes")
        assert "roguelikes" in mem.read("user")

    def test_relevant_memories_keyword_match(self, tmp_home):
        mem = MemoryManager(tmp_home / "data")
        mem.add_episode(Episode(time.time() - 100, "topic",
                                "User asked about Elden Ring DLC bosses",
                                ["elden", "ring", "dlc", "boss"], 3))
        older = time.time() - 30 * 86400
        mem.add_episode(Episode(older, "moment",
                                "User cooked pasta for the first time",
                                ["pasta", "cooking"], 1))
        ctx = mem.relevant_memories("how do I beat that elden ring boss?", char_budget=1000)
        assert "Elden Ring" in ctx and "pasta" not in ctx

    def test_stale_and_unrelated_excluded(self, tmp_home):
        mem = MemoryManager(tmp_home / "data")
        weeks_ago = time.time() - 40 * 86400
        mem.add_episode(Episode(weeks_ago, "topic", "Talked about the weather widget",
                                ["weather", "widget"], 1))
        ctx = mem.relevant_memories("tell me about your taste in music")
        assert "weather widget" not in ctx

    def test_extraction_dedupes_user_facts(self, tmp_home):
        mem = MemoryManager(tmp_home / "data")
        first = mem.apply_extraction({"user_facts": ["User plays bass guitar"]})
        assert first
        second = mem.apply_extraction({"user_facts": ["User plays bass guitar"]})
        assert second == []
        assert mem.read("user").count("bass guitar") == 1

    def test_extraction_saves_episode(self, tmp_home):
        mem = MemoryManager(tmp_home / "data")
        mem.apply_extraction({"episode": {
            "kind": "joke", "summary": "They named a bug garrett together",
            "keywords": ["garrett", "bug"], "importance": 4}})
        assert len(mem.episodes()) == 1
        assert mem.episodes()[0].kind == "joke"

    def test_invalid_kind_sanitized(self, tmp_home):
        mem = MemoryManager(tmp_home / "data")
        mem.apply_extraction({"episode": {"kind": "weird", "summary": "s", "keywords": [], "importance": 9}})
        assert mem.episodes()[0].kind == "moment"
        assert mem.episodes()[0].importance == 5


# ---------------------------------------------------------------------------
# conversation log
# ---------------------------------------------------------------------------
class TestConversationLog:
    def test_persistence_and_window(self, tmp_home):
        log = ConversationLog(tmp_home)
        log.append("user", "hi capri")
        log.append("assistant", "hey choom 💜")
        log2 = ConversationLog(tmp_home)
        assert len(log2.entries) == 2
        window = log2.window(1)
        assert window[0]["role"] == "assistant"

    def test_last_exchange(self, tmp_home):
        log = ConversationLog(tmp_home)
        log.append("user", "u1")
        log.append("assistant", "a1")
        log.append("user", "u2")
        log.append("assistant", "a2")
        assert log.last_exchange() == ("u2", "a2")


# ---------------------------------------------------------------------------
# providers (mock transport)
# ---------------------------------------------------------------------------
def _sse_response(chunks, status=200):
    sse_body = "".join(
        f'data: {json.dumps({"choices": [{"delta": {"content": c}}]})}\n\n' for c in chunks
    ) + "data: [DONE]\n\n"
    return httpx.Response(status, headers={"content-type": "text/event-stream"}, content=sse_body.encode())


class TestProviders:
    def test_missing_key_raises(self, tmp_home, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ProviderError):
            LLMClient("groq", "", "m", "https://x.test/v1")

    def test_streaming_parses_deltas(self):
        def handler(request):
            assert request.headers["authorization"] == "Bearer k"
            return _sse_response(["hey ", "choom"])

        client = LLMClient("groq", "k", "model-x", "https://x.test/v1",
                           transport=httpx.MockTransport(handler))
        assert "".join(client.stream([Message("user", "hi")])) == "hey choom"

    def test_rate_limit_retry_then_success(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, text="rate limited")
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        client = LLMClient("groq", "k", "m", "https://x.test/v1",
                           transport=httpx.MockTransport(handler))
        assert client.generate([Message("user", "hi")]) == "ok"
        assert calls["n"] == 2

    def test_auth_error_not_retried(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda *_: None)
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(401, text="bad key")

        client = LLMClient("groq", "k", "m", "https://x.test/v1",
                           transport=httpx.MockTransport(handler))
        with pytest.raises(ProviderError) as ei:
            client.generate([Message("user", "hi")])
        assert ei.value.status == 401
        assert calls["n"] == 1

    def test_gemini_preset_uses_openai_compat(self):
        presets = PROVIDER_PRESETS["gemini"]
        assert "openai" in presets["base_url"]


# ---------------------------------------------------------------------------
# chat controller end-to-end with fake client
# ---------------------------------------------------------------------------
class FakeClient:
    def __init__(self, reply_chunks=("cpus are ", "warm ", "tonight"), extraction=None):
        self.reply_chunks = reply_chunks
        self.extraction = extraction or {}

    def stream(self, messages, **kwargs):
        yield from self.reply_chunks
        self.last_messages = messages

    def generate(self, messages, **kwargs):
        return json.dumps(self.extraction)


class TestChatController:
    def test_language_enforced_in_prompt(self, tmp_home):
        cfg = make_config(tmp_home)
        cfg.set("language", "pt-BR")
        engine = ChatController(cfg, client=FakeClient(), data_dir_override=tmp_home / "data")
        prompt = engine.build_prompt("oi")
        assert "pt-BR" in prompt
        assert "English" in engine.personality.system_prompt("mood", "", language="English")

    def test_recent_openers_anti_repetition(self, tmp_home):
        cfg = make_config(tmp_home)
        engine = ChatController(cfg, client=FakeClient(), data_dir_override=tmp_home / "data")
        engine.history.append("assistant", "oi de novo, tudo certo? 💜")
        engine.history.append("user", "hm")
        engine.history.append("assistant", "hmm. sabia disso")
        prompt = engine.build_prompt("continua aí")
        assert "NÃO comece a resposta com nada parecido" in prompt
        assert "sabia disso" in prompt

    def test_no_openers_section_on_fresh_start(self, tmp_home):
        cfg = make_config(tmp_home)
        engine = ChatController(cfg, client=FakeClient(), data_dir_override=tmp_home / "data")
        prompt = engine.build_prompt("oi")
        assert "NÃO comece a resposta" not in prompt

    def test_history_preserved_between_turns(self, tmp_home):
        cfg = make_config(tmp_home)
        fake = FakeClient()
        engine = ChatController(cfg, client=fake, data_dir_override=tmp_home / "data")
        list(engine.respond("primeira mensagem"))
        list(engine.respond("segunda mensagem"))
        sent = fake.last_messages
        contents = [m["content"] for m in sent]
        assert "primeira mensagem" in contents
        assert any(m.get("role") == "assistant" for m in sent)
        assert sent[0]["role"] == "system"
        assert sent[-1]["content"] == "segunda mensagem"

    def test_extraction_mood_updates_emotion(self, tmp_home):
        cfg = make_config(tmp_home)
        fake = FakeClient(extraction={
            "mood": {"state": "irritated", "intensity": 0.7},
        })
        engine = ChatController(cfg, client=fake, data_dir_override=tmp_home / "data")
        list(engine.respond("meu código não compila de novo"))
        deadline = time.time() + 10
        while time.time() < deadline and engine.emotion.state != "irritated":
            time.sleep(0.2)
        state, intensity = engine.emotion.current()
        assert state == "irritated"
        assert intensity >= 0.7

    def test_extract_prompt_includes_current_mood(self, tmp_home):
        cfg = make_config(tmp_home)
        engine = ChatController(cfg, client=FakeClient(), data_dir_override=tmp_home / "data")
        engine.emotion.set_state("jealous", 0.6)
        raw_prompt = None

        def capture(messages, **kwargs):
            nonlocal raw_prompt
            raw_prompt = messages[0]["content"]
            return "{}"

        engine.client.generate = capture
        engine._run_extraction("oi", "oi!")
        assert raw_prompt is not None
        assert "state: jealous" in raw_prompt

    def test_full_turn_streams_and_saves(self, tmp_home):
        cfg = make_config(tmp_home)
        fake = FakeClient()
        engine = ChatController(cfg, client=fake, data_dir_override=tmp_home / "data")
        chunks = list(engine.respond("hey capri, miss you"))
        assert "".join(chunks) == "cpus are warm tonight"
        assert engine.history.entries[-1]["role"] == "assistant"
        assert engine.history.entries[-2]["content"] == "hey capri, miss you"

    def test_context_contains_mood_and_memory(self, tmp_home):
        cfg = make_config(tmp_home)
        engine = ChatController(cfg, client=FakeClient(), data_dir_override=tmp_home / "data")
        engine.emotion.set_state("playful", 0.7)
        engine.memory.add_episode(Episode(time.time(), "topic", "User is reinstalling Windows again",
                                          ["windows", "reinstall"], 3))
        prompt = engine.build_prompt("about that windows reinstall")
        assert "playful" in prompt
        assert "reinstalling Windows" in prompt
        assert "Capri" in prompt

    def test_extraction_applied_after_turn(self, tmp_home):
        cfg = make_config(tmp_home)
        fake = FakeClient(extraction={
            "user_facts": ["User runs a homelab"],
            "episode": {"kind": "topic", "summary": "User set up a homelab rack",
                        "keywords": ["homelab"], "importance": 3},
        })
        engine = ChatController(cfg, client=fake, data_dir_override=tmp_home / "data")
        list(engine.respond("I spent the weekend on my homelab"))
        deadline = time.time() + 10
        while time.time() < deadline and "homelab" not in engine.memory.read("user"):
            time.sleep(0.2)
        assert "homelab" in engine.memory.read("user")
        assert any("homelab" in ep.summary for ep in engine.memory.episodes())

    def test_autonomous_reply_not_treated_as_user_turn(self, tmp_home):
        cfg = make_config(tmp_home)
        engine = ChatController(cfg, client=FakeClient(), data_dir_override=tmp_home / "data")
        chunks = list(engine.respond_autonomous("thought", "say something warm"))
        assert "".join(chunks) == "cpus are warm tonight"
        assert engine.history.entries[-1]["role"] == "assistant"
        assert engine.history.entries[-1]["content"] == "cpus are warm tonight"


# ---------------------------------------------------------------------------
# autonomy
# ---------------------------------------------------------------------------
class TestAutonomy:
    def make_scheduler(self, cfg, **overrides):
        fired = []
        sched = AutonomyScheduler(
            cfg,
            on_proactive=lambda kind, instr: fired.append((kind, instr)),
            minutes_since_last_activity=lambda: overrides.get("idle", 30),
        )
        return sched, fired

    def make_cfg(self, enabled=True, quiet=[23, 8], cooldown=30, idle_req=20):
        cfg = Config.load()
        a = cfg.section("autonomy")
        a.update({"enabled": enabled, "min_interval_minutes": 45, "max_interval_minutes": 150,
                  "quiet_hours": quiet, "cooldown_minutes": cooldown,
                  "idle_minutes_before_proactive": idle_req})
        return cfg

    def test_disabled_never_fires(self, tmp_home):
        sched, fired = self.make_scheduler(self.make_cfg(enabled=False))
        assert sched.should_fire() is False
        assert fired == []

    AWAKE = time.mktime(time.struct_time((2024, 1, 15, 12, 0, 0, 0, 15, -1)))

    def test_fires_when_idle(self, tmp_home):
        sched, fired = self.make_scheduler(self.make_cfg())
        assert sched.should_fire(now=self.AWAKE) is True

    def test_blocked_when_user_active(self, tmp_home):
        cfg = self.make_cfg()
        sched, fired = self.make_scheduler(cfg)
        sched.minutes_since_last_activity = lambda: 1.0
        assert sched.should_fire(now=self.AWAKE) is False

    def test_cooldown_respected(self, tmp_home):
        cfg = self.make_cfg(cooldown=30)
        sched, _ = self.make_scheduler(cfg)
        sched._last_proactive = self.AWAKE - 120
        assert sched.should_fire(now=self.AWAKE) is False
        sched._last_proactive = self.AWAKE - 3600
        assert sched.should_fire(now=self.AWAKE) is True

    def test_quiet_hours(self, tmp_home):
        cfg = self.make_cfg(quiet=[2, 4])
        cfg.section("autonomy")
        sched, _ = self.make_scheduler(cfg)
        busy = time.mktime(time.struct_time((2024, 1, 15, 3, 0, 0, 0, 15, -1)))
        awake = time.mktime(time.struct_time((2024, 1, 15, 12, 0, 0, 0, 15, -1)))
        assert sched._in_quiet_hours(busy) is True
        assert sched._in_quiet_hours(awake) is False

    def test_quiet_hours_wrap_midnight(self, tmp_home):
        cfg = self.make_cfg(quiet=[23, 8])
        sched, _ = self.make_scheduler(cfg)
        busy = time.mktime(time.struct_time((2024, 1, 15, 1, 0, 0, 0, 15, -1)))
        assert sched._in_quiet_hours(busy) is True


# ---------------------------------------------------------------------------
# extraction json parsing
# ---------------------------------------------------------------------------
class TestExtractionParsing:
    def test_plain_json(self):
        obj = _parse_extraction_json('{"user_facts": ["a"], "episode": null}')
        assert obj == {"user_facts": ["a"], "episode": None}

    def test_json_with_wrapping_text(self):
        raw = 'Sure! Here it is:\n```json\n{"memories": ["x"]}\n```\n'
        obj = _parse_extraction_json(raw)
        assert obj is not None and obj["memories"] == ["x"]

    def test_garbage_returns_none(self):
        assert _parse_extraction_json("hmm?") is None

    def test_mood_validated(self):
        obj = _parse_extraction_json('{"mood": {"state": "not-a-state", "intensity": 2.0}}')
        assert obj is not None and "mood" not in obj
        obj = _parse_extraction_json('{"mood": {"state": "jealous", "intensity": 9}}')
        assert obj is not None and obj["mood"] == {"state": "jealous", "intensity": 1.0}
