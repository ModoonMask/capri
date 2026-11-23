"""Small Qt widgets shared across Capri's UI (frameless dragging, theming)."""
from __future__ import annotations

import ctypes
import sys

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QIcon, QPixmap, QLinearGradient
from PySide6.QtWidgets import QWidget, QApplication


class FramelessMixin:
    """Adds drag-by-anywhere behavior to a frameless widget."""

    def init_drag(self) -> None:
        self._drag_active = False
        self._drag_pos = QPoint()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_active and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_active = False
        event.accept()


def app_icon(accent: str = "#ff4f8b") -> QIcon:
    """Programmatic heart icon — no binary assets needed."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor("#3a0d1e"), 2))
    painter.setBrush(QBrush(QColor(accent)))
    painter.drawEllipse(28, 6, 10, 10)
    painter.drawEllipse(14, 6, 10, 10)
    painter.drawPolygon([QPoint(16, 14), QPoint(36, 14), QPoint(26, 28)])
    painter.end()
    return QIcon(pixmap)


def apply_accent(widget: QWidget, accent: str) -> None:
    widget.setStyleSheet(widget.styleSheet() + COMMON_QSS.format(accent=accent))


COMMON_QSS = """
QWidget {{ background: rgba(20, 16, 24, 235); color: #f2ecf4; font-size: 13px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {accent}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox {{
    background: rgba(38, 30, 44, 235); border: 1px solid rgba(255, 79, 139, 90);
    border-radius: 6px; padding: 6px; selection-background-color: {accent}; }}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{ border: 1px solid {accent}; }}
QComboBox QAbstractItemView {{ background: #2a2130; color: #f2ecf4; selection-background-color: {accent}; }}
QPushButton {{
    background: rgba(48, 38, 56, 235); border: 1px solid rgba(255, 79, 139, 120);
    border-radius: 6px; padding: 6px 14px; color: #f2ecf4; }}
QPushButton:hover {{ background: {accent}; color: #1c0912; }}
QPushButton:pressed {{ background: #c93a70; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {accent}; border-radius: 3px; background: rgba(38, 30, 44, 235); }}
QCheckBox::indicator:checked {{ background: {accent}; }}
QLabel#bubbleUser {{ background: {accent}; color: #1c0912; border-radius: 10px; padding: 8px 10px; }}
QLabel#bubbleCapri {{ background: #332a3c; color: #f2ecf4; border-radius: 10px; padding: 8px 10px; }}
"""
