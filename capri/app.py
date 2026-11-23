"""Application bootstrap: wires engine, GUI, tray, and autonomy together."""
from __future__ import annotations

import sys
import time
from typing import Iterator

from PySide6.QtCore import QObject, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .autonomy import AutonomyScheduler
from .chat import ChatController
from .config import Config, PROVIDER_PRESETS
from .gui.chat_window import ChatWindow, ResponseWorker
from .gui.companion import CompanionFace
from .gui.settings_dialog import SettingsDialog
from .gui.widgets import app_icon
from .memory import MemoryManager  # noqa: F401  (re-export convenience)
from .voice_in import MicGate, PushToTalk, Transcriber, VadListener
from .voice_out import Speaker
from .vision import ScreenWatcher, capture_jpeg


class EngineBridge(QObject):
    """Bridges the background autonomy thread into the Qt event loop."""

    proactive = Signal(str, str)   # kind, instruction


class VoiceBridge(QObject):
    """Bridges background mic threads into the Qt event loop."""

    heard = Signal(str)           # transcribed user speech
    speech_started = Signal()     # user began talking (used to interrupt TTS)


class CapriApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setQuitOnLastWindowClosed(False)
        self.cfg = Config.load()
        self.engine: ChatController | None = None
        self._ensure_engine()

        self.bridge = EngineBridge()
        self.bridge.proactive.connect(self._run_proactive)

        self.companion = CompanionFace(self.accent)
        self.chat_window = ChatWindow(self.accent)
        self.chat_window.send_message.connect(self.handle_user_message)
        self.companion.chat_requested.connect(self._toggle_chat)
        self.companion.settings_requested.connect(self.open_settings)
        self.companion.quit_requested.connect(self.quit)

        self._build_tray()
        self._sync_appearance()
        self._start_autonomy()
        self._last_activity = time.time()

        self._setup_voice_and_vision()

        self.mood_timer = QTimer(self)
        self.mood_timer.timeout.connect(self._refresh_mood)
        self.mood_timer.start(5000)

        self.companion.show()

    # ---- engine ------------------------------------------------------------
    def _ensure_engine(self) -> None:
        try:
            self.engine = ChatController(self.cfg)
        except Exception as exc:
            print(f"[capri] engine init failed: {exc}", file=sys.stderr)
            self.engine = None

    @property
    def accent(self) -> str:
        return self.cfg.section("appearance").get("accent", "#ff4f8b")

    def _sync_appearance(self) -> None:
        ap = self.cfg.section("appearance")
        self.companion.setWindowFlag(Qt.WindowStaysOnTopHint, bool(ap.get("always_on_top", True)))
        self.chat_window.setWindowFlag(Qt.WindowStaysOnTopHint, bool(ap.get("always_on_top", True)))
        self.companion.setWindowOpacity(float(ap.get("opacity", 0.95)))
        self.companion.set_accent(self.accent)
        self.setWindowIcon(app_icon(self.accent))

    # ---- tray ---------------------------------------------------------------
    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(app_icon(self.accent))
        menu = QMenu()
        act_chat = QAction("Open chat", self)
        act_chat.triggered.connect(self.show_chat)
        act_settings = QAction("Settings", self)
        act_settings.triggered.connect(self.open_settings)
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.quit)
        menu.addAction(act_chat)
        menu.addAction(act_settings)
        menu.addSeparator()
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("Capri")
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

    def _tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.Trigger:
            self._toggle_chat()

    # ---- voice / vision ----------------------------------------------------------
    def _setup_voice_and_vision(self) -> None:
        self.speaker: Speaker | None = None
        self.ptt: PushToTalk | None = None
        self.vad: VadListener | None = None
        self.watcher: ScreenWatcher | None = None
        self.voice_bridge = VoiceBridge()
        self.voice_bridge.heard.connect(self._on_voice_text)
        self.voice_bridge.speech_started.connect(self._on_user_speech_started)

        self.chat_window.voice_toggle.connect(self._toggle_voice)
        self.chat_window.vision_toggle.connect(self._toggle_vision)
        self._apply_voice_settings()
        self._apply_vision_settings()
        self.chat_window.set_voice_active(bool(self.cfg.section("voice").get("enabled", False)))
        self.chat_window.set_vision_active(bool(self.cfg.section("vision").get("enabled", False)))

    def _apply_voice_settings(self) -> None:
        v = self.cfg.section("voice")
        # tear down previous components (cheap; they are lazy about hardware)
        if self.ptt is not None:
            self.ptt.close()
            self.ptt = None
        if self.vad is not None:
            self.vad.close()
            self.vad = None
        if self.speaker is None and str(v.get("tts_engine", "edge_tts")) != "off":
            try:
                self.speaker = Speaker(self.cfg)
            except Exception as exc:  # noqa: BLE001
                print(f"[capri] tts unavailable: {exc}", file=sys.stderr)
        if not bool(v.get("enabled", False)):
            return
        transcriber = Transcriber(self.cfg)
        gate = MicGate(self.speaker)
        ptt = PushToTalk(
            self.cfg, transcriber,
            on_text=lambda t: self.voice_bridge.heard.emit(t),
            on_interrupt=lambda: self.voice_bridge.speech_started.emit(),
            gate=gate)
        if ptt.start():
            self.ptt = ptt
        if bool(v.get("vad_enabled", False)):
            vad = VadListener(
                self.cfg, transcriber,
                on_text=lambda t: self.voice_bridge.heard.emit(t),
                on_speech_start=lambda: self.voice_bridge.speech_started.emit(),
                gate=gate)
            if vad.start():
                self.vad = vad

    def _apply_vision_settings(self) -> None:
        if self.watcher is not None:
            self.watcher.close()
            self.watcher = None
        if not bool(self.cfg.section("vision").get("enabled", False)):
            return
        self.watcher = ScreenWatcher(self.cfg)
        self.watcher.start()

    def _toggle_voice(self, on: bool) -> None:
        self.cfg.section("voice")["enabled"] = bool(on)
        self.cfg.save()
        self._apply_voice_settings()

    def _toggle_vision(self, on: bool) -> None:
        self.cfg.section("vision")["enabled"] = bool(on)
        self.cfg.save()
        self._apply_vision_settings()

    def _on_user_speech_started(self) -> None:
        if self.speaker is not None:
            self.speaker.stop()

    def _on_voice_text(self, text: str) -> None:
        if not text:
            return
        self.chat_window.show()
        self.chat_window.raise_()
        self.chat_window.add_bubble(text, from_capri=False)
        self.handle_user_message(text)

    def _feed_speaker(self, delta: str) -> None:
        if self.speaker is not None:
            self.speaker.feed(delta)

    def _capture_screen_if_needed(self) -> bytes | None:
        if not bool(self.cfg.section("vision").get("enabled", False)):
            return None
        try:
            return capture_jpeg(
                monitor=int(self.cfg.section("vision").get("monitor", 1)),
                max_width=int(self.cfg.section("vision").get("max_width", 768)),
                quality=int(self.cfg.section("vision").get("jpeg_quality", 60)))
        except Exception as exc:  # noqa: BLE001
            print(f"[capri] screen capture failed: {exc}", file=sys.stderr)
            return None

    # ---- chat flow ------------------------------------------------------------
    def handle_user_message(self, text: str) -> None:
        if self.engine is None:
            self.chat_window.show_error("engine unavailable — check console")
            return
        if not self.cfg.api_key():
            preset = PROVIDER_PRESETS[self.cfg.provider]
            self.chat_window.show_error(
                f"No {preset['label']} API key set. Open Settings → AI model, "
                f"or set ${preset['key_env']}.")
            return
        if self.chat_window.is_busy():
            self.chat_window.show_error("one sec — still typing!")
            return
        self._last_activity = time.time()
        if self.speaker is not None:
            self.speaker.stop()   # she yields the floor to you
        image = self._capture_screen_if_needed()
        worker = ResponseWorker(self._respond_generator(text, image))
        self.chat_window.attach_worker(worker)
        if self.speaker is not None:
            worker.chunk.connect(self._feed_speaker)
            worker.finished_ok.connect(lambda _: self.speaker.finish())
            worker.failed.connect(lambda _: self.speaker.stop())
        self.chat_window.stream_start()
        worker.start()

    def _respond_generator(self, text: str, image: bytes | None = None) -> Iterator[str]:
        def _gen():
            engine = self.engine
            if engine is not None:
                yield from engine.respond(text, image=image)
        return _gen()

    def show_chat(self) -> None:
        self.chat_window.show()
        self.chat_window.raise_()
        self.chat_window.activateWindow()
        self.chat_window.input.setFocus()

    def _toggle_chat(self) -> None:
        if self.chat_window.isVisible():
            self.chat_window.hide()
        else:
            self.show_chat()

    # ---- autonomy ---------------------------------------------------------------
    def _start_autonomy(self) -> None:
        self.scheduler = AutonomyScheduler(
            self.cfg,
            on_proactive=self.bridge.proactive.emit,
            minutes_since_last_activity=self._minutes_idle,
        )
        self.scheduler.start()

    def _minutes_idle(self) -> float:
        return (time.time() - self._last_activity) / 60.0

    def _run_proactive(self, kind: str, instruction: str) -> None:
        if self.engine is None or not self.cfg.api_key():
            return
        if not self.cfg.section("autonomy").get("enabled", True):
            return
        if self.engine.client.in_cooldown():
            return  # provider just rate-limited us; don't ping it again for fun
        if self.chat_window.is_busy():
            return
        self._last_activity = time.time()
        worker = ResponseWorker(self.engine.respond_autonomous(kind, instruction))
        worker.chunk.connect(lambda _d: None)
        worker.finished_ok.connect(self._on_proactive_done)
        worker.failed.connect(lambda m: None)
        self._proactive_worker = worker
        worker.start()

    def _on_proactive_done(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.chat_window.add_bubble(text, from_capri=True)
        if self.speaker is not None:
            self.speaker.speak(text)
        if not self.chat_window.isVisible():
            self.tray.showMessage("Capri", text, QSystemTrayIcon.Information, 6000)

    # ---- mood sync -------------------------------------------------------------
    def _refresh_mood(self) -> None:
        if self.engine is None:
            return
        state, intensity = self.engine.emotion.current()
        self.companion.set_mood(state, intensity)

    # ---- settings ----------------------------------------------------------------
    def open_settings(self) -> None:
        if self.engine is None:
            self._ensure_engine()
        dlg = SettingsDialog(self.cfg, self.engine.personality if self.engine else None)
        if dlg.exec():
            self._sync_appearance()
            if self.engine:
                self._ensure_engine()  # rebuild client in case provider/model changed
            self._apply_voice_settings()   # provider may have changed for STT
            self._apply_vision_settings()
            self.chat_window.set_voice_active(bool(self.cfg.section("voice").get("enabled", False)))
            self.chat_window.set_vision_active(bool(self.cfg.section("vision").get("enabled", False)))
            self.scheduler.nudge()

    # ---- shutdown -------------------------------------------------------------------
    def quit(self) -> None:  # noqa: D102
        if getattr(self, "engine", None) and self.engine is not None:
            try:
                self.engine.emotion.save()
            except Exception:
                pass
        for closer in (getattr(self, "ptt", None), getattr(self, "vad", None),
                       getattr(self, "watcher", None)):
            if closer is not None:
                try:
                    closer.close()
                except Exception:
                    pass
        if getattr(self, "speaker", None) is not None:
            try:
                self.speaker.close()
            except Exception:
                pass
        tray = getattr(self, "tray", None)
        if tray is not None:
            tray.hide()
        super().quit()


def main() -> int:
    app = CapriApp(sys.argv)
    return app.exec()
