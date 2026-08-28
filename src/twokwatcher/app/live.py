"""Shared live state between the watcher thread and the UI.

The frame loop runs on its own thread and the HTTP server answers on others,
so everything crossing that boundary lives here behind one lock. Keeping it in
a single small class means there is exactly one place to reason about
thread safety, rather than it being smeared across the server handlers.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from ..pipeline import Event, EventBus

# Ring buffer of recent activity, sized for "what happened in this game".
MAX_EVENTS = 400


class LiveState:
    """What the watcher is seeing right now, readable by the UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self.state = "unknown"
        self.state_since = time.monotonic()
        self.source = "-"
        self.running = False
        self.error: str | None = None
        self.scoreboard: dict[str, Any] = {}
        self.frames_seen = 0
        self.frames_sampled = 0
        self._events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self._previews: dict[str, str] = {}
        self._preview_size: tuple[int, int] = (0, 0)

    # --- writes, from the watcher thread ---------------------------------

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe("state_change", self._on_state)
        bus.subscribe("scoreboard", self._on_scoreboard)
        bus.subscribe("preview", self._on_preview)

    def _on_state(self, event: Event) -> None:
        with self._lock:
            self.state = event.data["current"]
            self.state_since = time.monotonic()
            self._push(event, f"{event.data['previous']} → {event.data['current']}")

    def _on_scoreboard(self, event: Event) -> None:
        with self._lock:
            self.scoreboard = dict(event.data)

    def _on_preview(self, event: Event) -> None:
        """Store the HUD crops the parser is currently looking at.

        Showing these beside the parsed values is the whole point of the app:
        a bad crop or a misread is obvious at a glance, where a wrong number on
        its own tells you nothing about which stage went wrong.
        """
        crops: dict[str, np.ndarray] = event.data.get("crops", {})
        encoded = {}
        for name, crop in crops.items():
            if crop is None or crop.size == 0:
                continue
            # Upscale small HUD crops so they are actually legible in the UI.
            if crop.shape[0] < 60:
                crop = cv2.resize(crop, None, fx=3, fy=3,
                                  interpolation=cv2.INTER_NEAREST)
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                import base64
                encoded[name] = base64.b64encode(buf.tobytes()).decode("ascii")
        with self._lock:
            self._previews = encoded
            self._preview_size = event.data.get("frame_size", (0, 0))

    def note(self, kind: str, message: str) -> None:
        """Record something worth showing that did not come off the bus."""
        with self._lock:
            self._events.appendleft({
                "kind": kind, "message": message,
                "at": time.strftime("%H:%M:%S"), "frame": None,
            })

    def set_progress(self, seen: int, sampled: int) -> None:
        with self._lock:
            self.frames_seen = seen
            self.frames_sampled = sampled

    def set_running(self, running: bool, *, source: str = "-",
                    error: str | None = None) -> None:
        with self._lock:
            self.running = running
            self.source = source
            self.error = error

    def _push(self, event: Event, message: str) -> None:
        """Append to the activity log. Caller must hold the lock."""
        self._events.appendleft({
            "kind": event.kind,
            "message": message,
            "at": time.strftime("%H:%M:%S"),
            "frame": event.frame_index,
        })

    # --- reads, from the server threads -----------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "error": self.error,
                "source": self.source,
                "state": self.state,
                "state_seconds": round(time.monotonic() - self.state_since, 1),
                "uptime": round(time.monotonic() - self._started, 1),
                "frames_seen": self.frames_seen,
                "frames_sampled": self.frames_sampled,
                "scoreboard": dict(self.scoreboard),
                "events": list(self._events)[:60],
                "previews": dict(self._previews),
                "frame_size": list(self._preview_size),
            }
