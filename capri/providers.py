"""Unified LLM provider layer.

All supported providers (Mistral, Gemini, Groq, OpenRouter) expose an
OpenAI-compatible /chat/completions endpoint, so a single httpx client with a
provider-specific base URL covers everything — including SSE streaming.

Error handling: a `ProviderError` chains the HTTP status; 429 and 5xx get
retried with exponential backoff (Respect-RateLimit headers honored when the
provider sends them).
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Iterable, Iterator

import httpx

from .config import PROVIDER_PRESETS

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
DEFAULT_429_COOLDOWN_SEC = 20.0
MAX_COOLDOWN_WAIT_SEC = 30.0

# Providers/models that accept inline images (OpenAI-style image_url parts).
# Gemini is the primary supported path; a few capable models on other
# providers are recognized by name so the door stays open.
VISION_MODEL_HINTS = ("gemini", "vision", "llava", "llama-4", "gpt-4o")


def supports_vision(provider: str, model: str) -> bool:
    if provider == "gemini":
        return True
    model_l = (model or "").lower()
    return any(hint in model_l for hint in VISION_MODEL_HINTS)


def strip_images(messages: list[Message], provider: str, model: str) -> list[Message]:
    """Replace multimodal content with its text part when the target model
    cannot see images. Never raises; logs so the user knows why she 'isn't
    looking' on non-vision providers."""
    if supports_vision(provider, model):
        return messages
    dropped = 0
    safe: list[Message] = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            dropped += 1
            text = " ".join(part.get("text", "") for part in content
                            if part.get("type") == "text")
            msg = dict(msg, content=text)
        safe.append(msg)
    if dropped:
        print(f"[capri] {provider}/{model} does not support vision; "
              f"{dropped} screenshot(s) not sent (text-only turn)", file=sys.stderr)
    return safe


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status in RETRYABLE_STATUS


class Message(dict):
    """Tiny typed-ish dict wrapper: {"role": ..., "content": ...}."""

    def __init__(self, role: str, content: str):
        super().__init__(role=role, content=content)


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


def assistant(content: str) -> Message:
    return Message("assistant", content)


class LLMClient:
    """Synchronous streaming client; UI layers call it from worker threads."""

    def __init__(self, provider: str, api_key: str, model: str, base_url: str,
                 temperature: float = 0.9, max_tokens: int = 512,
                 extra_headers: dict[str, str] | None = None,
                 transport: httpx.BaseTransport | None = None):
        if provider not in PROVIDER_PRESETS:
            raise ProviderError(f"Unknown provider '{provider}'")
        if not api_key:
            raise ProviderError(
                f"No API key for '{provider}'. Set it in Settings or via "
                f"${PROVIDER_PRESETS[provider]['key_env']}.", status=401)
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.extra_headers = extra_headers or {}
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT, transport=transport)
        self._cooldown_until = 0.0

    def close(self) -> None:
        self._client.close()

    # ---- TPM safety: reactive cooldown after any 429 ----------------------
    def in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    def _throttle(self) -> None:
        wait = self._cooldown_until - time.time()
        if wait > 0:
            time.sleep(min(wait, MAX_COOLDOWN_WAIT_SEC))

    def _penalize_429(self, retry_after: float | None = None) -> None:
        self._cooldown_until = time.time() + (retry_after or DEFAULT_429_COOLDOWN_SEC)

    # ---- core ------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _payload(self, messages: list[Message], stream: bool,
                 temperature: float | None = None,
                 max_tokens: int | None = None) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": strip_images(list(messages), self.provider, self.model),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _request_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self._throttle()
            try:
                resp = self._client.post(self._url(), json=payload, headers=self._headers())
            except httpx.HTTPError as exc:
                last_error = ProviderError(f"Network error talking to {self.provider}: {exc}")
                if attempt == MAX_RETRIES - 1:
                    raise last_error from exc
                time.sleep(1.5 ** attempt)
                continue
            if resp.status_code == 200:
                return resp
            retry_after = _retry_after_seconds(resp)
            err = ProviderError(
                f"{self.provider} returned {resp.status_code}: {_error_text(resp)}",
                status=resp.status_code,
            )
            if resp.status_code == 429:
                self._penalize_429(retry_after)
            if err.retryable and attempt < MAX_RETRIES - 1:
                time.sleep(retry_after if retry_after else 1.5 ** attempt)
                last_error = err
                continue
            raise err
        raise last_error or ProviderError("Request failed after retries")

    # ---- high-level --------------------------------------------------------
    def generate(self, messages: list[Message], temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        """Non-streaming completion; used for memory extraction."""
        resp = self._request_with_retries(self._payload(messages, False, temperature, max_tokens))
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"Unexpected response shape from {self.provider}") from exc
        return (content or "").strip()

    def stream(self, messages: list[Message], temperature: float | None = None,
               max_tokens: int | None = None) -> Iterator[str]:
        """Yield incremental text deltas via SSE."""
        payload = self._payload(messages, True, temperature, max_tokens)
        attempt = 0
        while True:
            self._throttle()
            try:
                with self._client.stream("POST", self._url(), json=payload,
                                         headers=self._headers()) as resp:
                    if resp.status_code != 200:
                        retry_after = _retry_after_seconds(resp)
                        body = resp.read().decode("utf-8", "replace")
                        err = ProviderError(
                            f"{self.provider} returned {resp.status_code}: "
                            f"{_error_text_from(body)}", status=resp.status_code)
                        if resp.status_code == 429:
                            self._penalize_429(retry_after)
                        if err.retryable and attempt < MAX_RETRIES - 1:
                            attempt += 1
                            time.sleep(retry_after or (1.5 ** attempt))
                            continue
                        raise err
                    yield from _iter_sse_deltas(resp)
                    return
            except httpx.HTTPError as exc:
                if attempt < MAX_RETRIES - 1:
                    attempt += 1
                    time.sleep(1.5 ** attempt)
                    continue
                raise ProviderError(f"Network error during streaming: {exc}") from exc


def _iter_sse_deltas(resp: httpx.Response) -> Iterator[str]:
    for line in resp.iter_lines():
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        try:
            delta = obj["choices"][0].get("delta", {})
            piece = delta.get("content")
        except (KeyError, IndexError):
            continue
        if piece:
            yield piece


def _error_text(resp: httpx.Response) -> str:
    return _error_text_from(resp.text[:400])


def _error_text_from(body: str) -> str:
    try:
        obj = json.loads(body)
        return str(obj.get("error", {}).get("message", body))[:300]
    except json.JSONDecodeError:
        return body[:300]


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    for header in ("retry-after", "x-ratelimit-reset", "ratelimit-reset"):
        raw = resp.headers.get(header)
        if not raw:
            continue
        try:
            return max(1.0, float(raw))
        except ValueError:
            continue
    return None


def make_client(cfg) -> LLMClient:
    """Build an LLMClient from a Config object."""
    return LLMClient(
        provider=cfg.provider,
        api_key=cfg.api_key(),
        model=cfg.model,
        base_url=cfg.base_url,
        temperature=float(cfg.get("temperature", 0.9)),
        max_tokens=int(cfg.get("max_tokens", 512)),
        extra_headers={"HTTP-Referer": "https://capri.local", "X-Title": "Capri"} 
        if cfg.provider == "openrouter" else None,
    )
