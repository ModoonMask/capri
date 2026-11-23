"""Settings panel: provider/model, keys, behavior, autonomy, appearance."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QLineEdit,
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QLabel, QDialogButtonBox,
    QTabWidget, QWidget, QColorDialog,
)

from ..config import PROVIDER_PRESETS
from .. import autostart


def _ambient_key() -> QLineEdit:
    edit = QLineEdit()
    edit.setEchoMode(QLineEdit.Password)
    edit.setPlaceholderText("leave empty to use environment variable")
    return edit


class SettingsDialog(QDialog):
    def __init__(self, cfg, personality, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.personality = personality
        self.setWindowTitle("Capri — settings")
        self.resize(520, 480)
        self._build()

    # ---- construction ------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._model_tab(), "AI model")
        tabs.addTab(self._autonomy_tab(), "Initiative")
        tabs.addTab(self._appearance_tab(), "Appearance")
        root.addWidget(tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _model_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.provider_combo = QComboBox()
        for pid, preset in PROVIDER_PRESETS.items():
            self.provider_combo.addItem(preset["label"], pid)
        self.provider_combo.setCurrentIndex(max(0, list(PROVIDER_PRESETS).index(self.cfg.provider)))
        form.addRow("Provider", self.provider_combo)

        self.key_edit = _ambient_key()
        self.key_edit.setText(self.cfg.data.get("api_keys", {}).get(self.cfg.provider, ""))
        env_name = PROVIDER_PRESETS[self.cfg.provider]["key_env"]
        form.addRow("API key", self.key_edit)
        form.addRow("", QLabel(f"Env var alternative: <code>{env_name}</code>"))

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        for model in PROVIDER_PRESETS[self.cfg.provider]["models"]:
            self.model_combo.addItem(model)
        self.model_combo.setCurrentText(self.cfg.model)
        form.addRow("Model", self.model_combo)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(float(self.cfg.get("temperature", 0.9)))
        form.addRow("Temperature", self.temp_spin)

        self.maxtok_spin = QSpinBox()
        self.maxtok_spin.setRange(64, 4096)
        self.maxtok_spin.setSingleStep(64)
        self.maxtok_spin.setValue(int(self.cfg.get("max_tokens", 512)))
        form.addRow("Max reply tokens", self.maxtok_spin)

        self.stream_check = QCheckBox("stream responses (faster feel)")
        self.stream_check.setChecked(bool(self.cfg.get("stream", True)))
        form.addRow("", self.stream_check)

        self.autostart_check = QCheckBox("start with the operating system")
        self.autostart_check.setChecked(autostart.is_enabled())
        self.autostart_check.setEnabled(self._autostart_supported())
        form.addRow("", self.autostart_check)

        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        return w

    @staticmethod
    def _autostart_supported() -> bool:
        import sys
        return sys.platform in ("win32",) or sys.platform.startswith("linux")

    def _autonomy_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        a = self.cfg.section("autonomy")
        self.auto_check = QCheckBox("let Capri start conversations on her own")
        self.auto_check.setChecked(bool(a.get("enabled", True)))
        form.addRow("", self.auto_check)

        self.min_interval = QSpinBox()
        self.min_interval.setRange(5, 720)
        self.min_interval.setSuffix(" min")
        self.min_interval.setValue(int(a.get("min_interval_minutes", 45)))
        self.max_interval = QSpinBox()
        self.max_interval.setRange(10, 1440)
        self.max_interval.setSuffix(" min")
        self.max_interval.setValue(int(a.get("max_interval_minutes", 150)))
        row = QHBoxLayout()
        row.addWidget(self.min_interval)
        row.addWidget(QLabel("to"))
        row.addWidget(self.max_interval)
        row.addStretch(1)
        form.addRow("Proactive every", row)

        self.idle_spin = QSpinBox()
        self.idle_spin.setRange(0, 240)
        self.idle_spin.setSuffix(" min of silence")
        self.idle_spin.setValue(int(a.get("idle_minutes_before_proactive", 20)))
        form.addRow("Only after", self.idle_spin)

        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(0, 720)
        self.cooldown_spin.setSuffix(" min between proactive messages")
        self.cooldown_spin.setValue(int(a.get("cooldown_minutes", 30)))
        form.addRow("Plus", self.cooldown_spin)

        quiet = a.get("quiet_hours", [23, 8])
        self.quiet_start = QSpinBox()
        self.quiet_start.setRange(0, 23)
        self.quiet_start.setValue(int(quiet[0]))
        self.quiet_end = QSpinBox()
        self.quiet_end.setRange(0, 23)
        self.quiet_end.setValue(int(quiet[1]))
        qrow = QHBoxLayout()
        qrow.addWidget(QLabel("quiet hours"))
        qrow.addWidget(self.quiet_start)
        qrow.addWidget(QLabel("to"))
        qrow.addWidget(self.quiet_end)
        qrow.addStretch(1)
        form.addRow("", qrow)

        note = QLabel(
            "Capri never spawns messages while you are typing or chatting, and never\n"
            "during quiet hours. She also never invents what you were doing while away.")
        note.setStyleSheet("color: #b9a9c4;")
        form.addRow("", note)
        return w

    def _appearance_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        ap = self.cfg.section("appearance")
        self.accent = ap.get("accent", "#ff4f8b")
        self.accent_btn = QPushButton(self.accent)
        self.accent_btn.setFixedWidth(110)
        self.accent_btn.clicked.connect(self._pick_accent)
        form.addRow("Accent color", self.accent_btn)

        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.5, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(float(ap.get("opacity", 0.95)))
        form.addRow("Window opacity", self.opacity_spin)

        self.top_check = QCheckBox("keep Capri above other windows")
        self.top_check.setChecked(bool(ap.get("always_on_top", True)))
        form.addRow("", self.top_check)

        from ..config import data_dir
        live_dir = data_dir()
        files_label = QLabel(
            f"Personality files live in <code>{live_dir / 'personality'}</code> and memory in "
            f"<code>{live_dir / 'memory'}</code>. Edit them in any editor — changes apply "
            "on the next message, no restart needed.")
        files_label.setWordWrap(True)
        form.addRow("", files_label)
        return w

    def _pick_accent(self) -> None:
        color = QColorDialog.getColor(self.accent, self, "pick her accent")
        if color.isValid():
            self.accent = color.name()
            self.accent_btn.setText(self.accent)

    def _provider_changed(self, index: int) -> None:
        pid = self.provider_combo.itemData(index)
        self.model_combo.clear()
        for model in PROVIDER_PRESETS[pid]["models"]:
            self.model_combo.addItem(model)
        env = PROVIDER_PRESETS[pid]["key_env"]
        stored = self.cfg.data.get("api_keys", {}).get(pid, "")
        self.key_edit.setText(stored)
        self.key_edit.setPlaceholderText(
            f"leave empty to use environment variable ({env})"
            if os.environ.get(env) else "paste your API key here")

    # ---- saving --------------------------------------------------------------
    def _save(self) -> None:
        pid = self.provider_combo.currentData()
        if pid != self.cfg.provider:
            self.cfg.set("provider", pid)
            self.cfg.on_provider_changed()
        keys = self.cfg.data.setdefault("api_keys", {})
        new_key = self.key_edit.text().strip()
        if new_key:
            keys[pid] = new_key
        self.cfg.data["model"] = self.model_combo.currentText().strip() or PROVIDER_PRESETS[pid]["models"][0]
        self.cfg.data["temperature"] = float(self.temp_spin.value())
        self.cfg.data["max_tokens"] = int(self.maxtok_spin.value())
        self.cfg.data["stream"] = bool(self.stream_check.isChecked())

        a = self.cfg.section("autonomy")
        a["enabled"] = bool(self.auto_check.isChecked())
        a["min_interval_minutes"] = int(self.min_interval.value())
        a["max_interval_minutes"] = int(self.max_interval.value())
        a["idle_minutes_before_proactive"] = int(self.idle_spin.value())
        a["cooldown_minutes"] = int(self.cooldown_spin.value())
        a["quiet_hours"] = [int(self.quiet_start.value()), int(self.quiet_end.value())]

        ap = self.cfg.section("appearance")
        ap["accent"] = self.accent
        ap["opacity"] = float(self.opacity_spin.value())
        ap["always_on_top"] = bool(self.top_check.isChecked())

        self.cfg.save()
        autostart.set_enabled(self.autostart_check.isChecked())
        self.accept()
