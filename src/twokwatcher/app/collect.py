"""Automatic collection of representative frames.

The first live session's most valuable output is not stats — it is footage.
Every parser still unwritten is blocked on nobody having seen this particular
HUD, and a folder of representative frames unblocks all of them offline.

So rather than asking someone to remember to take screenshots while they are
playing, this rides the state machine: it saves frames keyed by the screen they
came from, keeping a spread of each rather than a burst from one moment. Play a
normal session and you end up with a labelled calibration set.

PNG, not JPEG. Compression artifacts around small HUD glyphs are exactly the
detail the glyph atlas needs intact.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..pipeline import Event, EventBus

log = logging.getLogger(__name__)

# Per-state caps. Post-game screens earn a bigger budget: they are the densest
# structured data in the game and vary most between modes.
DEFAULT_CAPS = {
    "live": 25,
    "post_game": 30,
    "dead_ball": 10,
    "menu": 8,
    "replay": 6,
    "loading": 3,
    "unknown": 3,
}

# Minimum seconds between saves within one state, so a session yields a spread
# of situations rather than thirty near-identical frames from one possession.
MIN_INTERVAL = 4.0


class FrameCollector:
    """Saves a bounded, spread-out sample of frames per game state."""

    def __init__(
        self,
        out_dir: Path | str = "data/collect",
        *,
        caps: dict[str, int] | None = None,
        min_interval: float = MIN_INTERVAL,
        enabled: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.caps = dict(caps or DEFAULT_CAPS)
        self.min_interval = min_interval
        self.enabled = enabled
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._last_saved: dict[str, float] = {}
        self._state = "unknown"
        self._manifest: list[dict] = []

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe("state_change", self._on_state)
        bus.subscribe("preview", self._on_frame)

    @property
    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    @property
    def total(self) -> int:
        with self._lock:
            return sum(self._counts.values())

    def _on_state(self, event: Event) -> None:
        with self._lock:
            self._state = event.data["current"]
            # A transition is the most informative moment to capture — it is
            # where the screen just became something new — so let the next
            # frame through regardless of how recently we saved.
            self._last_saved.pop(self._state, None)

    def _on_frame(self, event: Event) -> None:
        frame = event.data.get("frame")
        if not self.enabled or frame is None:
            return

        with self._lock:
            state = self._state
            cap = self.caps.get(state, 5)
            count = self._counts.get(state, 0)
            if count >= cap:
                return
            now = time.monotonic()
            if now - self._last_saved.get(state, -1e9) < self.min_interval:
                return
            self._last_saved[state] = now
            self._counts[state] = count + 1
            index = count

        self._write(frame, state, index, event)

    def _write(self, frame: np.ndarray, state: str, index: int,
               event: Event) -> None:
        directory = self.out_dir / state
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{state}-{index:03d}.png"
        try:
            cv2.imwrite(str(path), frame)
        except Exception:                                  # noqa: BLE001
            # Collection is a bonus, never a reason to take the watcher down.
            log.exception("Could not write %s", path)
            return

        with self._lock:
            self._manifest.append({
                "path": str(path.relative_to(self.out_dir)),
                "state": state,
                "frame_index": event.frame_index,
                "video_ts": round(event.video_ts, 3),
                "size": [int(frame.shape[1]), int(frame.shape[0])],
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

    def flush_manifest(self) -> Path | None:
        """Write the index of what was collected."""
        with self._lock:
            entries = list(self._manifest)
        if not entries:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "manifest.json"
        path.write_text(json.dumps(
            {"frames": entries, "counts": self.counts}, indent=2))
        return path
