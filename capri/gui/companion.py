"""The floating companion: a small always-on-top character window.

Drawn entirely with QPainter (no assets): a rounded "core" with two eyes whose
shape/color reflect Capri's current emotional state. Clicking it toggles the
chat window. Right-click opens a small menu (chat, settings, quit).
"""
from __future__ import annotations

import math
import random

from PySide6.QtCore import Qt, QTimer, QPoint, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QBrush, QAction
from PySide6.QtWidgets import QWidget, QMenu

from .widgets import FramelessMixin

MOOD_COLORS = {
    "happy": "#ffd166", "playful": "#ff8fa3", "affectionate": "#ff4f8b",
    "jealous": "#b24feb", "curious": "#4fd1ff", "bored": "#9aa0b5",
    "irritated": "#ff6b4f", "sad": "#6f9bd1", "vulnerable": "#c9b6ff",
    "excited": "#ffe066",
}

EYE_STYLE = {
    "happy": "smile", "playful": "wink", "affectionate": "smile",
    "jealous": "glare", "curious": "round", "bored": "half",
    "irritated": "glare", "sad": "droop", "vulnerable": "droop", "excited": "round",
}


class CompanionFace(QWidget, FramelessMixin):
    clicked = Signal()
    settings_requested = Signal()
    quit_requested = Signal()
    chat_requested = Signal()

    def __init__(self, accent: str = "#ff4f8b"):
        super().__init__()
        self.init_drag()
        self._press_global = QPoint()
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(150, 150)
        self.mood = "affectionate"
        self.intensity = 0.5
        self.accent = accent
        self.thinking = False
        self._blink = 1.0
        self._bob = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)

    def _tick(self) -> None:
        # blink occasionally
        if random.random() < 0.012:
            self._blink = 0.05
        else:
            self._blink = min(1.0, self._blink + 0.08)
        if self.thinking:
            self._bob += 0.12
        self.update()

    def set_mood(self, mood: str, intensity: float = 0.5) -> None:
        self.mood = mood if mood in MOOD_COLORS else "affectionate"
        self.intensity = max(0.0, min(1.0, intensity))
        self.update()

    def set_accent(self, accent: str) -> None:
        self.accent = accent
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._press_global = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            moved = (event.globalPosition().toPoint() - self._press_global).manhattanLength() > 8
            if not moved:
                self.clicked.emit()
                self.chat_requested.emit()
        elif event.button() == Qt.RightButton:
            menu = QMenu(self)
            menu.addAction("open chat", self.chat_requested.emit)
            menu.addAction("settings", self.settings_requested.emit)
            menu.addSeparator()
            menu.addAction("quit Capri", self.quit_requested.emit)
            menu.exec(event.globalPosition().toPoint())
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        bob = math.sin(self._bob) * 2.0 if self.thinking else 0.0
        base = QColor(MOOD_COLORS.get(self.mood, self.accent))
        glow = QColor(base)
        glow.setAlpha(60 + int(40 * self.intensity))

        cx, cy = w / 2, h / 2 + bob
        r = min(w, h) * 0.30

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(cx - r * 1.35, cy - r * 1.35, r * 2.7, r * 2.7)

        painter.setPen(QPen(QColor(base).darker(140), 2))
        painter.setBrush(QBrush(QColor("#191220")))
        painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        eye_color = QColor(base).lighter(135)
        eye_style = EYE_STYLE.get(self.mood, "round")
        eye_dx = r * 0.38
        eye_dy = -r * 0.25
        eye_r = r * 0.16
        painter.setPen(QPen(eye_color, 2.2))

        wink = eye_style == "wink" and (math.sin(self._bob * 0.7) > 0.8)
        for side in (-1, 1):
            ex = cx + side * eye_dx
            ey = cy + eye_dy
            if self._blink < 0.4 or (wink and side == 1):
                painter.drawLine(QPoint(int(ex - eye_r), int(ey)), QPoint(int(ex + eye_r), int(ey)))
                continue
            if eye_style == "smile":
                path = QPainterPath()
                path.moveTo(ex - eye_r, ey + eye_r * 0.4)
                path.quadTo(ex, ey - eye_r * 1.1, ex + eye_r, ey + eye_r * 0.4)
                painter.drawPath(path)
            elif eye_style in ("glare",):
                painter.drawLine(QPoint(int(ex - eye_r), int(ey)), QPoint(int(ex + eye_r), int(ey)))
                painter.drawLine(QPoint(int(ex + side * eye_r * 0.25), int(ey)),
                                 QPoint(int(ex + side * eye_r * 0.25), int(ey - eye_r)))
            elif eye_style == "half":
                painter.drawLine(QPoint(int(ex - eye_r), int(ey)), QPoint(int(ex + eye_r), int(ey)))
                painter.drawArc(int(ex - eye_r), int(ey - eye_r), int(eye_r * 2), int(eye_r * 2), 0, 180 * 16)
            elif eye_style == "droop":
                painter.drawEllipse(ex - eye_r / 2, ey - eye_r * 0.25, eye_r, eye_r * 0.75)
            else:
                painter.drawEllipse(ex - eye_r / 2, ey - eye_r, eye_r * 2, eye_r * 2)
        painter.end()
