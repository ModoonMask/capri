"""Capri's chat window: frameless, themed, streaming display."""
from __future__ import annotations

import html as html_mod

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QScrollArea,
    QFrame, QPushButton, QSizePolicy,
)

from .widgets import FramelessMixin, app_icon


class ResponseWorker(QThread):
    """Streams a generator of text deltas into Qt signals."""

    chunk = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, generator):
        super().__init__()
        self._generator = generator

    def run(self) -> None:
        parts: list[str] = []
        try:
            for delta in self._generator:
                parts.append(delta)
                self.chunk.emit(delta)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit("".join(parts))


class ChatBubble(QLabel):
    def __init__(self, text: str, from_capri: bool, max_width: int = 340):
        super().__init__()
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)
        self.setText(f"<div style='line-height:135%'>{html_mod.escape(text)}</div>")
        self.setObjectName("bubbleCapri" if from_capri else "bubbleUser")
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter if from_capri else Qt.AlignRight | Qt.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setMaximumWidth(max_width)


class ChatWindow(QWidget, FramelessMixin):
    send_message = Signal(str)

    def __init__(self, accent: str = "#ff4f8b"):
        super().__init__()
        self.init_drag()
        self.setWindowTitle("Capri")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.setWindowIcon(app_icon(accent))
        self.resize(460, 560)
        self._worker: ResponseWorker | None = None
        self._current_capri_bubble: ChatBubble | None = None
        self._build_ui()

    # ---- UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(1, 1, 1, 1)
        self.setStyleSheet(self._qss())

        titlebar = self._titlebar()
        root.addWidget(titlebar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.messages_host = QWidget()
        self.messages_host.setObjectName("messagesHost")
        self.messages = QVBoxLayout(self.messages_host)
        self.messages.setAlignment(Qt.AlignTop)
        self.messages.setSpacing(8)
        self.messages.setContentsMargins(10, 10, 10, 10)
        self.scroll.setWidget(self.messages_host)
        root.addWidget(self.scroll, 1)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(8, 6, 8, 8)
        self.input = QLineEdit()
        self.input.setPlaceholderText("talk to her…")
        self.input.returnPressed.connect(self._on_send)
        send_btn = QPushButton("send ⚡")
        send_btn.setCursor(Qt.PointingHandCursor)
        send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.input, 1)
        input_row.addWidget(send_btn)
        root.addLayout(input_row)

    def _qss(self) -> str:
        return """
        QWidget#titlebar { background: #221a29; border-bottom: 1px solid rgba(255,79,139,90); }
        QWidget#messagesHost { background: rgba(16, 12, 20, 235); }
        """

    def _titlebar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("titlebar")
        bar.setFixedHeight(34)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 4, 8, 4)
        title = QLabel("💜 Capri")
        title.setStyleSheet("background: transparent; font-weight: 600; letter-spacing: 1px;")
        layout.addWidget(title)
        layout.addStretch(1)
        hide_btn = QPushButton("—")
        hide_btn.setFixedSize(26, 24)
        hide_btn.setCursor(Qt.PointingHandCursor)
        hide_btn.clicked.connect(self.hide)
        layout.addWidget(hide_btn)
        return bar

    # ---- chat flow ----------------------------------------------------------
    def _on_send(self) -> None:
        text = self.input.text().strip()
        if not text or self._worker is not None:
            return
        self.input.clear()
        self.add_bubble(text, from_capri=False)
        self.send_message.emit(text)

    def is_busy(self) -> bool:
        return self._worker is not None

    def add_bubble(self, text: str, from_capri: bool) -> None:
        bubble = ChatBubble(text, from_capri)
        if from_capri:
            row = QHBoxLayout()
        else:
            row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if not from_capri:
            row.addStretch(1)
        row.addWidget(bubble)
        if from_capri:
            row.addStretch(1)
        host = QWidget()
        host.setStyleSheet("background: transparent;")
        host.setLayout(row)
        self.messages.addWidget(host)
        QTimer.singleShot(10, self._scroll_to_end)

    def stream_start(self) -> None:
        self._current_capri_bubble = ChatBubble("", from_capri=True)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._current_capri_bubble)
        row.addStretch(1)
        host = QWidget()
        host.setStyleSheet("background: transparent;")
        host.setLayout(row)
        self.messages.addWidget(host)

    def stream_delta(self, delta: str) -> None:
        if self._current_capri_bubble is None:
            self.stream_start()
        bubble = self._current_capri_bubble
        raw = getattr(bubble, "_raw", "") + delta
        bubble._raw = raw
        accentless = html_mod.escape(raw).replace("\n", "<br>")
        bubble.setText(f"<div style='line-height:135%'>{accentless}</div>")
        QTimer.singleShot(5, self._scroll_to_end)

    def stream_finish(self) -> None:
        self._worker = None
        self._current_capri_bubble = None

    def show_error(self, message: str) -> None:
        self.add_bubble(f"⚠ {message}", from_capri=True)

    def attach_worker(self, worker: ResponseWorker) -> None:
        self._worker = worker
        worker.chunk.connect(self.stream_delta)
        worker.finished_ok.connect(lambda _: self.stream_finish())
        worker.failed.connect(self._on_failed)

    def _on_failed(self, message: str) -> None:
        self.stream_finish()
        self.show_error(message)

    def _scroll_to_end(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
