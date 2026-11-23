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
import time
from typing import Any, Iterable, Iterator

import httpx

from .config import PROVIDER_PRESETS

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


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

    def close(self) -> None:
        self._client.close()

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
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _request_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
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
            if err.retryable and attempt < MAX_RETRIES - 1:
                time.sleep(retry_after or 1.5 ** attempt)
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
            try:
                with self._client.stream("POST", self._url(), json=payload,
                                         headers=self._headers()) as resp:
                    if resp.status_code != 200:
                        body = resp.read().decode("utf-8", "replace")
                        err = ProviderError(
                            f"{self.provider} returned {resp.status_code}: "
                            f"{_error_text_from(body)}", status=resp.status_code)
                        if err.retryable and attempt < MAX_RETRIES - 1:
                            attempt += 1
                            time.sleep(_retry_after_from(body) or (1.5 ** attempt))
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


def _retry_after_from(body: str) -> float | None:
    return None  # headers were consumed by the stream context; fall back to backoff


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
