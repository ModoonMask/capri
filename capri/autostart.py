"""Start-with-OS support: Windows registry Run key, .desktop autostart on Linux."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def _command() -> str:
    if getattr(sys, "frozen", False):  # PyInstaller bundle
        return f'"{sys.executable}"'
    script = Path(__file__).resolve().parent.parent / "run_capri.py"
    return f'"{Path(sys.executable)}" "{script}"'


def set_enabled(enabled: bool) -> bool:
    if sys.platform == "win32":
        return _windows_set(enabled)
    if sys.platform.startswith("linux"):
        return _linux_set(enabled)
    return False


def is_enabled() -> bool:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
                winreg.QueryValueEx(key, "CapriCompanion")
                return True
        except (OSError, FileNotFoundError):
            return False
    if sys.platform.startswith("linux"):
        return _linux_path().exists()
    return False


def _windows_set(enabled: bool) -> bool:
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                             winreg.KEY_SET_VALUE)
        try:
            if enabled:
                winreg.SetValueEx(key, "CapriCompanion", 0, winreg.REG_SZ, _command())
            else:
                try:
                    winreg.DeleteValue(key, "CapriCompanion")
                except FileNotFoundError:
                    pass
        finally:
            key.Close()
        return True
    except OSError:
        return False


def _linux_path() -> Path:
    config_home = Path(__import__("os").environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "autostart" / "capri.desktop"


def _linux_set(enabled: bool) -> bool:
    path = _linux_path()
    try:
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "[Desktop Entry]\nType=Application\nName=Capri\nExec=" + _command() + "\n"
                "X-GNOME-Autostart-enabled=true\n", encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def windows_use_dark_titlebar(window) -> None:
    """Best-effort dark titlebar on Windows 10/11; harmless no-op elsewhere."""
    if sys.platform != "win32":
        return
    try:
        hwnd = int(window.winId())
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), 4)
    except Exception:
        pass
