"""Screen vision for Capri ("streamer mode"): she sees your game, not your files.

- capture(): grabs the configured monitor with mss, downscales so the width is
  ~max_width px, and encodes JPEG (quality ~60) — small enough to be worth
  attaching, cheap enough to refresh.
- image_parts(): wraps the JPEG into the OpenAI-compatible inline image format
  ("image_url" content part with a base64 data URI).
- ScreenWatcher: optional background loop that refreshes a cached frame at
  cfg.vision.interval_s, keeping it only when the screen changed "enough"
  (mean pixel diff >= change_threshold). The cached frame is attached to the
  NEXT user turn; it never triggers an API call by itself.

Cost & privacy discipline:
- Images are captured only when you speak (voice turn) or, if the watcher is
  on, refreshed into RAM at a slow cadence.
- Images are NEVER persisted: not in conversation-history JSONL, not in memory
  files, not on disk anywhere. Only her text reply is stored.
"""
from __future__ import annotations

import base64
import sys
import threading
import time
from io import BytesIO
from typing import Callable


class Frame:
    __slots__ = ("jpeg", "ts")

    def __init__(self, jpeg: bytes, ts: float):
        self.jpeg = jpeg
        self.ts = ts


def capture_jpeg(monitor: int = 1, max_width: int = 768, quality: int = 60) -> bytes:
    """One screenshot -> JPEG bytes of at most max_width px wide."""
    import mss
    from PIL import Image

    with mss.mss() as sct:
        shots = sct.monitors
        index = monitor if 0 < monitor < len(shots) else 1
        raw = sct.grab(shots[index])
    img = Image.frombytes("RGB", raw.size, raw.rgb)
    if img.width > max_width:
        height = round(img.height * max_width / img.width)
        img = img.resize((max_width, height), Image.BILINEAR)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def data_uri(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def image_parts(text: str, jpeg: bytes) -> list[dict]:
    """OpenAI-compatible multimodal user content (text + inline image)."""
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_uri(jpeg)}},
    ]


def frame_signature(jpeg: bytes) -> float:
    """Grayscale mean of a 32x18 downscale — cheap change fingerprint."""
    from PIL import Image

    img = Image.open(BytesIO(jpeg)).convert("L").resize((32, 18))
    pixels = img.tobytes()
    return sum(pixels) / len(pixels)


def frames_differ(a: bytes, b: bytes, threshold: float = 8.0) -> bool:
    """True when the mean pixel diff of downscaled frames >= threshold (0-255)."""
    from PIL import Image

    if not b:
        return True
    ia = Image.open(BytesIO(a)).convert("L").resize((32, 18))
    ib = Image.open(BytesIO(b)).convert("L").resize((32, 18))
    pa, pb = ia.tobytes(), ib.tobytes()
    diff = sum(abs(x - y) for x, y in zip(pa, pb))
    return diff / len(pa) >= threshold


class ScreenWatcher:
    """Refreshes the cached 'latest interesting frame' at a slow cadence."""

    def __init__(self, cfg, on_fresh: Callable[[bytes], None] | None = None):
        v = cfg.section("vision")
        self.monitor = int(v.get("monitor", 1))
        self.interval_s = float(v.get("interval_s", 12))
        self.max_width = int(v.get("max_width", 768))
        self.quality = int(v.get("jpeg_quality", 60))
        self.change_threshold = float(v.get("change_threshold", 8.0))
        self.on_fresh = on_fresh
        self.latest: Frame | None = None
        self._stop = threading.Event()

    def start(self) -> bool:
        try:
            import mss  # noqa: F401  (fails on headless boxes)
        except Exception as exc:  # noqa: BLE001
            print(f"[capri] screen watching unavailable ({exc})", file=sys.stderr)
            return False
        threading.Thread(target=self._loop, daemon=True, name="capri-vision").start()
        return True

    def close(self) -> None:
        self._stop.set()

    def latest_frame(self) -> Frame | None:
        return self.latest

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                jpeg = capture_jpeg(monitor=self.monitor, max_width=self.max_width,
                                    quality=self.quality)
                prev = self.latest.jpeg if self.latest else b""
                if frames_differ(jpeg, prev, self.change_threshold):
                    self.latest = Frame(jpeg, time.time())
                    if self.on_fresh:
                        self.on_fresh(jpeg)
            except Exception as exc:  # noqa: BLE001
                print(f"[capri] vision loop: {exc}", file=sys.stderr)
                self._stop.wait(5.0)   # back off quietly and retry
            self._stop.wait(self.interval_s)
