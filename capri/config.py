"""Configuration management for Capri.

Config lives in ~/.capri/config.json. API keys are only stored there if the
user enters them in the settings panel; environment variables always take
precedence and are never written to disk by Capri.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

APP_NAME = "capri"

# Provider presets: every provider speaks the OpenAI chat-completions protocol
# (Gemini through Google's official OpenAI-compatible endpoint).
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "mistral": {
        "label": "Mistral",
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
        "models": ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest", "open-mistral-nemo"],
    },
    "gemini": {
        "label": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "models": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"],
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "models": [
            "mistralai/mistral-nemo",
            "meta-llama/llama-3.3-70b-instruct",
            "google/gemini-2.0-flash-001",
            "anthropic/claude-3.5-haiku",
        ],
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "provider": "groq",
    "api_keys": {},          # provider -> key (leave empty to use env vars)
    "model": "",             # empty = first preset model of the provider
    "language": "pt-BR",     # chat language, e.g. "pt-BR", "English"
    "temperature": 0.9,
    "max_tokens": 320,
    "stream": True,
    "appearance": {
        "accent": "#ff4f8b",
        "companion_scale": 1.0,
        "opacity": 0.95,
        "always_on_top": True,
    },
    "autonomy": {
        "enabled": True,
        "min_interval_minutes": 45,
        "max_interval_minutes": 150,
        "quiet_hours": [23, 8],      # [start_hour, end_hour), wrapped at midnight
        "idle_minutes_before_proactive": 20,
        "cooldown_minutes": 30,
    },
    "context": {
        "max_history_messages": 30,
        "history_char_budget": 3000,
        "memory_char_budget": 2500,
        "memory_extract_min_interval_seconds": 90,
    },
}


@dataclass
class Config:
    """Runtime configuration with nested-dict access kept simple and typed."""

    data: dict[str, Any] = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_CONFIG)))
    path: Path = field(default_factory=lambda: default_config_path())

    # ---- IO -------------------------------------------------------------
    @staticmethod
    def load() -> "Config":
        path = default_config_path()
        cfg = Config(path=path)
        if path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                cfg.data = _deep_merge(cfg.data, stored)
            except (json.JSONDecodeError, OSError):
                pass  # fall back to defaults, never crash the app over config
        return cfg

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Accessors ------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, DEFAULT_CONFIG.get(key, default))

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def section(self, name: str) -> dict[str, Any]:
        return self.data.setdefault(name, {})

    # ---- Provider helpers ----------------------------------------------
    @property
    def provider(self) -> str:
        p = self.data.get("provider", "groq")
        return p if p in PROVIDER_PRESETS else "groq"

    @property
    def base_url(self) -> str:
        return PROVIDER_PRESETS[self.provider]["base_url"]

    def api_key(self) -> str:
        """Env var wins over the settings panel so keys never *have* to hit disk."""
        stored = self.data.get("api_keys", {}).get(self.provider, "")
        env_name = PROVIDER_PRESETS[self.provider]["key_env"]
        return os.environ.get(env_name, "") or stored

    @property
    def model(self) -> str:
        stored = self.data.get("model", "")
        if stored:
            return stored
        return PROVIDER_PRESETS[self.provider]["models"][0]

    def on_provider_changed(self) -> None:
        """Reset stored model when switching providers so it always matches presets."""
        self.data["model"] = ""


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def default_config_path() -> Path:
    home = Path(os.environ.get("CAPRI_HOME", Path.home() / f".{APP_NAME}"))
    return home / "config.json"


def bundled_data_dir() -> Path:
    """The defaults shipped inside the project/repo (source templates)."""
    package_dir = Path(__file__).resolve().parent
    return package_dir.parent / "data"


def data_dir() -> Path:
    """Directory holding personality + memory files (override with CAPRI_DATA)."""
    env = os.environ.get("CAPRI_DATA")
    if env:
        return Path(env)
    bundled = bundled_data_dir()
    if (bundled / "personality").is_dir():
        return bundled
    return default_config_path().parent / "data"


def ensure_data_seeded() -> Path:
    """First run: copy template personality/memory files into the live data dir.

    Only copies files that do not exist yet, so user edits and accumulated
    memories are never overwritten.
    """
    import shutil

    dest = data_dir()
    src = bundled_data_dir()
    if not (src / "personality").is_dir():
        return dest
    for sub in ("personality", "memory"):
        src_sub, dest_sub = src / sub, dest / sub
        dest_sub.mkdir(parents=True, exist_ok=True)
        for item in src_sub.iterdir():
            if item.is_file():
                target = dest_sub / item.name
                if not target.exists():
                    shutil.copy2(item, target)
    return dest
