"""Speech-to-text for Capri.

Two switchable backends:
- "groq_api":      Groq's hosted Whisper transcription (reuses GROQ_API_KEY;
                   https://api.groq.com/openai/v1/audio/transcriptions).
- "local_whisper": faster-whisper running on CPU (lazy import; the model files
                   download on first use, so nothing here hard-fails offline).

Listening modes:
- push-to-talk: hold cfg.voice.push_to_talk_key to record; releasing stops and
  transcribes. Pressing the key while Capri is speaking interrupts her speech.
- VAD (optional): an energy-based voice-activity loop auto-transcribes short
  utterances; it is silent (never captures) while Capri speaks (MicGate).

No recordings are ever written to disk: audio lives in RAM only, and the
transcription is treated exactly like typed text.
"""
from __future__ import annotations

import io
import os
import struct
import sys
import threading
import wave
from typing import Callable

RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16


def language_code(language: str) -> str:
    return str(language or "pt").split("-")[0].lower()


class MicGate:
    """Decides whether the mic may listen; blocked while Capri talks."""

    def __init__(self, speaker=None):
        self.speaker = speaker

    def should_listen(self) -> bool:
        if self.speaker is not None and self.speaker.is_speaking():
            return False
        return True


def wav_bytes(frames: list[bytes]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(RATE)
        for frame in frames:
            w.writeframes(frame)
    return buf.getvalue()


def rms(frame: bytes) -> float:
    """RMS amplitude of PCM16 mono data, scaled to a 0..1000-ish band."""
    if not frame:
        return 0.0
    total = 0
    count = len(frame) // 2
    for (value,) in struct.iter_unpack("<h", frame[: count * 2]):
        total += value * value
    return (total / max(1, count)) ** 0.5 / 32768.0 * 1000.0


class MicRecorder:
    """Records PCM16 frames from the default input device until stopped."""

    def __init__(self):
        import sounddevice
        self.sd = sounddevice
        self._stream = None
        self._frames: list[bytes] = []
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG001
        with self._lock:
            if not self._stop_flag.is_set():
                self._frames.append(bytes(indata))

    def start(self) -> None:
        self._stop_flag = threading.Event()
        self._frames = []
        self._stream = self.sd.InputStream(
            samplerate=RATE, channels=CHANNELS, dtype="int16",
            callback=self._callback, blocksize=1024)
        self._stream.start()

    def stop(self) -> bytes:
        self._stop_flag.set()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            frames, self._frames = self._frames, []
        return wav_bytes(frames)


class PushToTalk:
    """Polls a global key (keyboard lib) and drives record->transcribe.

    Pressing the key at any time stops Capri's speech first (interruption),
    then opens the mic.
    """

    def __init__(self, cfg, transcriber: "Transcriber",
                 on_text: Callable[[str], None],
                 on_interrupt: Callable[[], None] | None = None,
                 gate: MicGate | None = None, poll_s: float = 0.05):
        self.cfg = cfg
        self.transcriber = transcriber
        self.on_text = on_text
        self.on_interrupt = on_interrupt or (lambda: None)
        self.gate = gate or MicGate()
        self.poll_s = poll_s
        self.key = str(cfg.section("voice").get("push_to_talk_key", "f9"))
        self.enabled = True
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._recorder: MicRecorder | None = None

    # ---- lifecycle -----------------------------------------------------------
    def start(self) -> bool:
        try:
            import keyboard  # lazy; Windows-only-friendly
        except Exception as exc:  # noqa: BLE001
            print(f"[capri] push-to-talk unavailable ({exc}); typing or VAD still work",
                  file=sys.stderr)
            return False
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="capri-ptt")
        self._thread.start()
        return True

    def close(self) -> None:
        self._stop.set()

    # ---- loop ------------------------------------------------------------------
    def _pressed(self) -> bool:
        import keyboard
        return bool(keyboard.is_pressed(self.key))

    def _loop(self) -> None:
        recorder: MicRecorder | None = None
        while not self._stop.is_set():
            try:
                pressed = self._pressed()
            except Exception:  # noqa: BLE001
                return
            if pressed and recorder is None:
                if self._recorder is None:
                    try:
                        self._recorder = MicRecorder()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[capri] mic unavailable ({exc})", file=sys.stderr)
                        return
                self.on_interrupt()          # she stops talking; user has the floor
                recorder = self._recorder
                recorder.start()
            elif not pressed and recorder is not None:
                audio = recorder.stop()
                recorder = None
                threading.Thread(target=self._transcribe, args=(audio,),
                                 daemon=True, name="capri-stt").start()
            self._stop.wait(self.poll_s)

    def _transcribe(self, audio: bytes) -> None:
        if len(audio) < RATE // 4:  # under 250ms: ignore clicks
            return
        try:
            text = self.transcriber.transcribe(audio)
        except Exception as exc:  # noqa: BLE001
            print(f"[capri] stt failed: {exc}", file=sys.stderr)
            return
        text = text.strip()
        if text and self.on_text:
            self.on_text(text)


class VadListener:
    """Optional always-on listener using an RMS energy threshold."""

    def __init__(self, cfg, transcriber: "Transcriber",
                 on_text: Callable[[str], None],
                 on_speech_start: Callable[[], None] | None = None,
                 gate: MicGate | None = None):
        import sounddevice
        self.sd = sounddevice
        self.cfg = cfg
        self.transcriber = transcriber
        self.on_text = on_text
        self.on_speech_start = on_speech_start or (lambda: None)
        self.gate = gate or MicGate()
        self.threshold = float(cfg.section("voice").get("vad_threshold", 400))
        self.hangover_s = 0.7
        self.max_utterance_s = 20.0
        self._stop = threading.Event()

    def start(self) -> bool:
        try:
            self.sd.query_devices()  # sanity: some audio backend exists
        except Exception as exc:  # noqa: BLE001
            print(f"[capri] VAD unavailable ({exc})", file=sys.stderr)
            return False
        threading.Thread(target=self._loop, daemon=True, name="capri-vad").start()
        return True

    def close(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        chunk_frames = RATE // 10  # 100ms
        try:
            stream = self.sd.InputStream(samplerate=RATE, channels=CHANNELS,
                                         dtype="int16", blocksize=chunk_frames)
            stream.start()
        except Exception as exc:  # noqa: BLE001
            print(f"[capri] VAD mic failed: {exc}", file=sys.stderr)
            return
        speaking = False
        frames: list[bytes] = []
        quiet_for = 0.0
        try:
            while not self._stop.is_set():
                if not self.gate.should_listen():
                    if speaking:
                        speaking, frames, quiet_for = False, [], 0.0
                    self._stop.wait(0.2)
                    continue
                data, _ = stream.read(chunk_frames)
                frame = bytes(data)
                level = rms(frame)
                if level >= self.threshold:
                    if not speaking:
                        speaking = True
                        frames = []
                        self.on_speech_start()
                    frames.append(frame)
                    quiet_for = 0.0
                elif speaking:
                    frames.append(frame)
                    quiet_for += len(frame) / (2 * RATE)
                    total_s = sum(len(f) for f in frames) / (2 * RATE)
                    if quiet_for >= self.hangover_s or total_s > self.max_utterance_s:
                        speaking = False
                        quiet_for = 0.0
                        audio = wav_bytes(frames)
                        frames = []
                        if len(audio) > RATE // 2:  # >= 500ms of audio
                            threading.Thread(
                                target=self._transcribe, args=(audio,),
                                daemon=True, name="capri-stt").start()
                else:
                    self._stop.wait(0.02)
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    def _transcribe(self, audio: bytes) -> None:
        try:
            text = self.transcriber.transcribe(audio)
        except Exception as exc:  # noqa: BLE001
            print(f"[capri] stt failed: {exc}", file=sys.stderr)
            return
        text = text.strip()
        if text and self.on_text:
            self.on_text(text)


class Transcriber:
    """Backend dispatch: Groq hosted Whisper or local faster-whisper."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._local_model = None

    # ---- backend: Groq API ---------------------------------------------------
    def _groq_key(self) -> str:
        key = str(self.cfg.data.get("api_keys", {}).get("groq", "") or "")
        if not key:
            key = os.environ.get("GROQ_API_KEY", "")
        return key

    def transcribe_groq(self, audio: bytes) -> str:
        import httpx

        key = self._groq_key()
        if not key:
            raise RuntimeError(
                "Groq API key required for speech recognition "
                "(set $GROQ_API_KEY or paste it in Settings)")
        r = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": ("audio.wav", audio, "audio/wav")},
            data={
                "model": "whisper-large-v3-turbo",
                "language": language_code(str(self.cfg.get("language", "pt"))),
                "response_format": "text",
            },
            timeout=60.0,
        )
        if r.status_code != 200:
            raise RuntimeError(f"groq stt returned {r.status_code}: {r.text[:120]}")
        return r.text

    # ---- backend: local faster-whisper ----------------------------------------
    def transcribe_local(self, audio: bytes) -> str:
        if self._local_model is None:
            from faster_whisper import WhisperModel  # lazy: heavy, downloads model
            model_name = str(self.cfg.section("voice").get("stt_model", "small"))
            self._local_model = WhisperModel(model_name, device="cpu",
                                             compute_type="int8")
        lang = language_code(str(self.cfg.get("language", "pt")))
        with wave.open(io.BytesIO(audio), "rb") as w:
            pcm = w.readframes(w.getnframes())
        segments, _info = self._local_model.transcribe(
            pcm, language=lang if lang in ("pt", "en", "es", "fr", "de", "it") else None,
            beam_size=1)
        return " ".join(seg.text for seg in segments)

    # ---- dispatch --------------------------------------------------------------
    def transcribe(self, audio: bytes) -> str:
        backend = str(self.cfg.section("voice").get("stt_backend", "groq_api"))
        if backend == "local_whisper":
            return self.transcribe_local(audio)
        return self.transcribe_groq(audio)
