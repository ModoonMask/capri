"""Tests for voice (TTS + STT) and vision (screen capture). No network, no audio."""
import io
import json
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


# ---------------------------------------------------------------------------
# vision: image format + provider fallback (stub transport, no network)
# ---------------------------------------------------------------------------
from test_core import FakeClient, make_config, tmp_home  # noqa: F401 (fixture re-export)
from capri.chat import ChatController
from capri.providers import LLMClient, Message, supports_vision, strip_images
from capri.vision import capture_jpeg, data_uri, frames_differ, image_parts
from PIL import Image
import httpx


def _jpeg(width=1600, height=900, color=(30, 20, 40)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, "JPEG")
    return buf.getvalue()


class TestImageFormat:
    def test_image_parts_shape(self):
        parts = image_parts("olha isso", b"jpeg-ish")
        assert parts[0] == {"type": "text", "text": "olha isso"}
        url = parts[1]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        import base64
        assert base64.b64decode(url.split(",", 1)[1]) == b"jpeg-ish"

    def test_frame_signature_bounds(self):
        from capri.vision import frame_signature
        buf = io.BytesIO()
        Image.new("RGB", (1600, 900), (10, 10, 10)).save(buf, "JPEG", quality=60)
        assert 0.0 <= frame_signature(buf.getvalue()) <= 255.0

    def test_frames_differ_threshold(self):
        a = _jpeg(color=(0, 0, 0))
        assert frames_differ(a, b"") is True
        assert frames_differ(a, a) is False
        b = _jpeg(color=(240, 240, 240))
        assert frames_differ(a, b, threshold=8.0) is True

    def test_data_uri_reversible(self):
        jpeg = _jpeg()
        head, b64 = data_uri(jpeg).split(",", 1)
        assert head == "data:image/jpeg;base64"
        import base64
        assert base64.b64decode(b64) == jpeg


def _gemini_client(handler):
    return LLMClient("gemini", "k", "gemini-2.0-flash",
                     "https://x.test/v1", transport=httpx.MockTransport(handler))


def _groq_client(handler):
    return LLMClient("groq", "k", "llama-3.1-8b-instant",
                     "https://x.test/v1", transport=httpx.MockTransport(handler))


class TestProviderVisionSupport:
    def test_gemini_supports_vision(self):
        assert supports_vision("gemini", "gemini-2.0-flash") is True
        assert supports_vision("groq", "llama-3.1-8b-instant") is False
        assert supports_vision("openrouter", "google/gemini-2.0-flash-001") is True

    def test_groq_strips_image_and_keeps_text(self, capsys):
        sent = []
        def handler(request):
            sent.append(json.loads(request.content))
            sse = 'data: ' + json.dumps({"choices": [{"delta": {"content": "hmm"}}]}) + "\n\n" + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=sse.encode())
        client = _groq_client(handler)
        msgs = [Message("user", image_parts("vejo isso?", _jpeg()))]
        out = "".join(client.stream(msgs))
        assert out == "hmm"
        content = sent[0]["messages"][-1]["content"]
        assert content == "vejo isso?"            # text survived
        assert "image_url" not in json.dumps(sent)  # image never left the machine
        err = capsys.readouterr().err
        assert "does not support vision" in err

    def test_gemini_keeps_image(self):
        sent = []
        def handler(request):
            sent.append(json.loads(request.content))
            sse = 'data: ' + json.dumps({"choices": [{"delta": {"content": "panela?"}}]}) + "\n\n" + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=sse.encode())
        client = _gemini_client(handler)
        list(client.stream([Message("user", image_parts("olha", _jpeg()))]))
        content = sent[0]["messages"][-1]["content"]
        assert isinstance(content, list)
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_strip_images_text_only_messages_untouched(self):
        msgs = [Message("system", "s"), Message("user", "hi")]
        assert strip_images(msgs, "groq", "llama-3.1-8b-instant") == msgs


class TestChatWithImage:
    def test_turn_sends_image_but_history_stores_text_only(self, tmp_home):
        cfg = make_config(tmp_home)
        fake = FakeClient()
        engine = ChatController(cfg, client=fake, data_dir_override=tmp_home / "data")
        chunks = list(engine.respond("o que é isso?", image=_jpeg()))
        assert "".join(chunks) == "cpus are warm tonight"
        last_sent = fake.last_messages[-1]
        assert isinstance(last_sent["content"], list)              # multimodal to the model
        assert last_sent["content"][0]["text"] == "o que é isso?"
        assert engine.history.entries[-2]["content"] == "o que é isso?"  # disk: text only
        assert "image_url" not in json.dumps(engine.history.entries)

    def test_prompt_mentions_screen_when_image_attached(self, tmp_home):
        cfg = make_config(tmp_home)
        engine = ChatController(cfg, client=FakeClient(), data_dir_override=tmp_home / "data")
        assert "imagem da tela" in engine.build_prompt("oi", sees_screen=True)
        assert "imagem da tela" not in engine.build_prompt("oi")


# ---------------------------------------------------------------------------
# STT: wav/rms helpers, gate, transcriber dispatch (no network)
# ---------------------------------------------------------------------------
import struct
import wave

from capri.voice_in import MicGate, Transcriber, language_code, rms, wav_bytes


class TestSttHelpers:
    def test_language_code(self):
        assert language_code("pt-BR") == "pt"
        assert language_code("English") == "english"
        assert language_code("") == "pt"

    def test_rms_silence_vs_signal(self):
        quiet = struct.pack("<h", 0) * 1024
        loud = struct.pack("<h", 16000) * 1024
        assert rms(b"") == 0.0
        assert rms(quiet) == 0.0
        assert rms(loud) > rms(quiet)

    def test_wav_bytes_roundtrip(self):
        frames = [struct.pack("<h", i % 3000) * 16 for i in range(8)]
        audio = wav_bytes(frames)
        with wave.open(io.BytesIO(audio), "rb") as w:
            assert w.getframerate() == 16000
            assert w.getnchannels() == 1
            assert w.getnframes() == 8 * 16
            assert w.readframes(w.getnframes()) == b"".join(
                struct.pack("<h", i % 3000) * 16 for i in range(8))


class TestMicGate:
    def test_blocked_while_capri_speaks(self):
        class FakeSpeaker:
            def is_speaking(self):
                return self._on
        sp = FakeSpeaker()
        sp._on = True
        gate = MicGate(sp)
        assert gate.should_listen() is False
        sp._on = False
        assert gate.should_listen() is True

    def test_open_without_speaker(self):
        assert MicGate().should_listen() is True


class TestTranscriber:
    def test_groq_requires_key(self, cfg, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        cfg.data["api_keys"] = {}
        t = Transcriber(cfg)
        with pytest.raises(RuntimeError, match="Groq API key"):
            t.transcribe(b"\x00" * 32000)

    def test_groq_dispatch_uses_api(self, cfg, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "k-test")
        captured = {}
        class FakeResp:
            status_code = 200
            text = "ola capri"
        def fake_post(url, **kw):
            captured["url"] = url
            captured["data"] = kw.get("data", {})
            return FakeResp()
        import httpx as httpx_mod
        monkeypatch.setattr(httpx_mod, "post", fake_post)
        t = Transcriber(cfg)
        assert t.transcribe(b"\x00" * 32000) == "ola capri"
        assert "audio/transcriptions" in captured["url"]
        assert captured["data"]["model"] == "whisper-large-v3-turbo"
        assert captured["data"]["language"] == "pt"

    def test_dispatch_to_local_backend(self, cfg, monkeypatch):
        cfg.section("voice")["stt_backend"] = "local_whisper"
        t = Transcriber(cfg)
        calls = []
        monkeypatch.setattr(t, "transcribe_local", lambda a: calls.append(a) or "local")
        assert t.transcribe(b"\x00" * 32000) == "local"
        assert calls
