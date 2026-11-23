"""Text-to-speech for Capri: speaks her streaming replies sentence by sentence.

Design notes:
- edge-tts is the default engine (free, Microsoft neural voices, pt-BR aware).
  The engine is isolated behind Speaker._synthesize so another one can be
  plugged in later.
- Audio playback goes through a small player abstraction (sounddevice +
  soundfile decode the mp3 edge-tts produces). If no audio backend exists,
  the Speaker degrades to a logged no-op instead of crashing the app.
- The GUI only ever calls feed()/finish()/stop()/speak(), which are cheap:
  synthesis and playback happen on a private worker thread, never on a
  Qt thread.
- is_speaking() drives the mic gate in voice_in so the microphone goes quiet
  while Capri talks (echo suppression), and user speech can interrupt her
  via stop().
Never store audio or spoken text anywhere: speech exists only in RAM.
"""
from __future__ import annotations

import asyncio
import io
import queue
import re
import sys
import threading
import unicodedata

DEFAULT_VOICE = "pt-BR-FranciscaNeural"

_SENTENCE_END = re.compile(r"(?:[.!?。…]|\.\.\.|\n)+\s*")
_MARKDOWN = re.compile(r"[*_`~#>`]|\[([^\]]*)\]\([^)]*\)|https?://\S+")
_EMOJIISH = re.compile("[\U0001F000-\U0001FAFF\u2190-\u21FF\u2600-\u27BF\uFE0F]")


def spoken_text(text: str) -> str:
    """Clean chat formatting out of text before a speech engine reads it."""
    text = _MARKDOWN.sub(lambda m: m.group(1) or " ", text)
    text = text.replace("💜", " ").replace("♡", " ")
    text = _EMOJIISH.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


class SentenceSplicer:
    """Accumulates streaming deltas and releases complete sentences early."""

    def __init__(self, min_chars: int = 2):
        self.min_chars = min_chars
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        ready: list[str] = []
        start = 0
        pos = 0
        while True:
            m = _SENTENCE_END.search(self._buf, pos)
            if not m:
                break
            end = m.end()
            chunk = self._buf[start:end].strip()
            if len(chunk) >= self.min_chars:
                ready.append(chunk)
                start = end
            pos = end
        self._buf = self._buf[start:]
        return ready

    def flush(self) -> str:
        rest = self._buf.strip()
        self._buf = ""
        return rest


class SilencePlayer:
    """Fallback player used when no audio backend is importable."""

    def __init__(self, reason: str = ""):
        self.reason = reason

    def play(self, audio_bytes: bytes) -> None:
        return

    def stop(self) -> None:
        return


class SoundFilePlayer:
    """mp3 playback via soundfile (libsndfile) + sounddevice blocks."""

    def __init__(self):
        import sounddevice  # noqa: F401  (raises if PortAudio is missing)
        import soundfile
        self._sd = sounddevice
        self._sf = soundfile
        self._stop = threading.Event()
        self._current = None

    def play(self, audio_bytes: bytes) -> None:
        self._stop.clear()
        try:
            with self._sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                rate = f.samplerate
                channels = len(f.read(0, dtype="float32", always_2d=True)[0])
                f.seek(0)

                def callback(outdata, frames, time_info, status):  # noqa: ARG001
                    if self._stop.is_set():
                        raise self._sd.CallbackStop
                    block = f.read(frames, dtype="float32", always_2d=True)
                    n = len(block)
                    outdata[:n] = block
                    if n < frames:
                        outdata[n:] = 0
                        raise self._sd.CallbackStop

                stream = self._sd.OutputStream(
                    samplerate=rate, channels=channels, dtype="float32",
                    callback=callback)
                stream.start()
                self._current = stream
                while not self._stop.is_set() and stream.active:
                    threading.Event().wait(0.05)
        except Exception as exc:  # noqa: BLE001
            if not self._stop.is_set():
                print(f"[capri] tts playback failed: {exc}", file=sys.stderr)
        finally:
            self._current = None

    def stop(self) -> None:
        self._stop.set()


def make_default_player():
    try:
        return SoundFilePlayer()
    except Exception as exc:  # noqa: BLE001
        print(f"[capri] audio output unavailable ({exc}); Capri will be written-only",
              file=sys.stderr)
        return SilencePlayer(reason=str(exc))


class Speaker:
    """Sentence-queued speaker: feed streaming deltas, she talks as they land."""

    def __init__(self, cfg, player=None, voice_name: str | None = None):
        v = cfg.section("voice")
        self.engine = str(v.get("tts_engine", "edge_tts"))
        if not voice_name:
            voice_name = str(v.get("voice", "")) or _lang_to_edge_voice(
                str(cfg.get("language", "pt-BR"))) or DEFAULT_VOICE
        self.voice_name = voice_name
        self.enabled = True
        self._player = player if player is not None else make_default_player()
        self._queue: queue.Queue = queue.Queue()
        self._splicer = SentenceSplicer()
        self._splicing_lock = threading.Lock()
        self._speaking = threading.Event()
        self._stop_speech = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="capri-tts")
        self._worker.start()

    # ---- streaming API (cheap, safe from any thread) ------------------------
    def feed(self, delta: str) -> None:
        if not self.enabled:
            return
        with self._splicing_lock:
            for sentence in self._splicer.feed(delta):
                self._queue.put(spoken_text(sentence))

    def finish(self) -> None:
        """End of stream: flush whatever sentence fragment is pending."""
        with self._splicing_lock:
            rest = self._splicer.flush()
        if rest:
            self._queue.put(spoken_text(rest))

    def speak(self, text: str) -> None:
        """Say a whole non-streamed text (e.g. proactive messages)."""
        if not self.enabled:
            return
        cleaned = spoken_text(text.strip())
        if cleaned:
            self._queue.put(cleaned)

    def stop(self) -> None:
        """Interrupt immediately: drop the queue and kill current audio."""
        with self._splicing_lock:
            self._splicer = SentenceSplicer()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(None)
        self._stop_speech.set()

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    # ---- worker loop ---------------------------------------------------------
    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:          # stop sentinel
                while True:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        break
                self._stop_speech.clear()
                continue
            if not item or self._stop_speech.is_set():
                self._stop_speech.clear()
                continue
            try:
                audio = self._synthesize(item)
                if audio and not self._stop_speech.is_set():
                    self._speaking.set()
                    self._player.play(audio)
            except Exception as exc:  # noqa: BLE001
                print(f"[capri] tts error: {exc}", file=sys.stderr)
            finally:
                if self._queue.empty():
                    self._speaking.clear()

    def _synthesize(self, sentence: str) -> bytes:
        """One sentence -> mp3 bytes. Isolated for easy engine swaps/tests."""
        import edge_tts

        async def _go() -> bytes:
            com = edge_tts.Communicate(sentence, self.voice_name)
            buf = bytearray()
            async for chunk in com.stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
            return bytes(buf)

        return asyncio.run(_go())

    def close(self) -> None:
        self.stop()
        self.enabled = False
