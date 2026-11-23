"""Tests for voice (TTS + STT) and vision (screen capture). No network, no audio."""
import io
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capri.config import Config
from capri.voice_out import SentenceSplicer, Speaker, spoken_text


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPRI_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("CAPRI_HOME", str(tmp_path / "home"))
    return Config.load()


class FakePlayer:
    def __init__(self):
        self.played: list[bytes] = []
        self.blocking = False

    def play(self, audio_bytes):
        while self.blocking:
            time.sleep(0.01)
        self.played.append(audio_bytes)

    def stop(self):
        self.blocking = False


class TestSentenceSplicer:
    def test_releases_complete_sentences(self):
        s = SentenceSplicer()
        assert s.feed("você chegou j") == []
        out = s.feed("antar? estava saudade ")
        assert out == ["você chegou jantar?"]
        assert s.flush() == "estava saudade"

    def test_multiline_counts_as_boundary(self):
        s = SentenceSplicer()
        assert s.feed("opa\n") == ["opa"]

    def test_tiny_fragment_goes_back_to_buffer(self):
        s = SentenceSplicer(min_chars=6)
        assert s.feed("ok.") == []          # too short: held back
        assert s.flush() == "ok."


class TestSpokenText:
    def test_strips_markdown_emoji_and_urls(self):
        raw = "💜 [link](https://x.test) *vou* contar! 😊 https://tes.te/x"
        assert "https" not in spoken_text(raw)
        assert "💜" not in spoken_text(raw)
        assert "*" not in spoken_text(raw)
        assert "link" in spoken_text(raw)

    def test_keeps_punctuation(self):
        assert spoken_text("oi, tudo bem?!") == "oi, tudo bem?!"


class TestSpeaker:
    def make(self, cfg):
        player = FakePlayer()
        sp = Speaker(cfg, player=player, voice_name="pt-BR-FranciscaNeural")
        sp._synthesize = lambda sentence: f"mp3:{sentence}".encode()
        return sp, player

    def test_streams_sentences_soon_after_chunk(self, cfg):
        sp, player = self.make(cfg)
        try:
            sp.feed("oi. você voltou ")
            sp.feed("tar. ")
            sp.finish()
            deadline = time.time() + 5
            while time.time() < deadline and len(player.played) < 2:
                time.sleep(0.02)
            texts = [b.decode() for b in player.played]
            assert "mp3:oi." in texts
            assert any("você voltou tar" in t for t in texts)
        finally:
            sp.close()

    def test_stop_interrupts_pending_speech(self, cfg):
        sp, player = self.make(cfg)
        try:
            sp.speak("frase um. frase dois. frase três.")
            deadline = time.time() + 5
            while time.time() < deadline and sp._queue.empty():
                time.sleep(0.02)
            sp.stop()
            time.sleep(0.3)
            assert sp._queue.qsize() <= 1  # at most the sentinel
        finally:
            sp.close()

    def test_is_speaking_true_during_playback(self, cfg):
        sp, player = self.make(cfg)
        try:
            release = threading.Event()
            shine = threading.Event()

            def slow_play(audio_bytes):
                release.set()
                shine.wait(5)          # keep her "speaking" until the test looks
                player.played.append(audio_bytes)

            sp._player.play = slow_play
            sp.speak("algo longo o bastante")
            assert release.wait(5), "speech never started"
            assert sp.is_speaking() is True
            shine.set()
        finally:
            sp.close()

    def test_disabled_speaker_ignores_feed(self, cfg):
        sp, player = self.make(cfg)
        sp.enabled = False
        sp.feed("nada.")
        sp.finish()
        sp.speak("também não.")
        assert player.played == []
        sp.close()


class TestVoiceConfigDefaults:
    def test_voice_and_vision_defaults_exist(self, cfg):
        v = cfg.section("voice")
        assert v["tts_engine"] == "edge_tts"
        assert v["stt_backend"] in ("groq_api", "local_whisper")
        assert "push_to_talk_key" in v
        assert "vad_enabled" in v
        vis = cfg.section("vision")
        assert vis["max_width"] == 768
        assert vis["jpeg_quality"] == 60
        assert vis["enabled"] is False

    def test_stored_config_survives_new_defaults(self, cfg, tmp_path):
        cfg.data["voice"] = {"push_to_talk_key": "f4"}
        cfg.save()
        cfg2 = Config.load()
        assert cfg2.data["voice"]["push_to_talk_key"] == "f4"
        assert cfg2.data["voice"]["tts_engine"] == "edge_tts"
