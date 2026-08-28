"""Loading and resolving the on-disk configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/regions.yaml")
EXAMPLE_CONFIG_PATH = Path("config/regions.example.yaml")


@dataclass(frozen=True)
class Region:
    """A named rectangle, stored in normalized (0..1) coordinates.

    Normalized so the same config works whether the capture is 720p, 1080p or
    1440p. Pixel coordinates only appear at the moment we slice a frame.
    """

    name: str
    x: float
    y: float
    w: float
    h: float

    def to_pixels(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) in pixels for a frame of the given size."""
        return (
            int(round(self.x * frame_w)),
            int(round(self.y * frame_h)),
            int(round(self.w * frame_w)),
            int(round(self.h * frame_h)),
        )

    def crop(self, frame):
        """Slice this region out of a frame (an HxWxC ndarray)."""
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = self.to_pixels(frame_w, frame_h)
        # Clamp so a slightly out-of-bounds region degrades instead of throwing.
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame_w, x + w), min(frame_h, y + h)
        return frame[y0:y1, x0:x1]


@dataclass
class Config:
    """Everything 2kWatcher reads from disk at startup."""

    regions: dict[str, Region] = field(default_factory=dict)
    capture: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def region(self, name: str) -> Region:
        try:
            return self.regions[name]
        except KeyError:
            raise KeyError(
                f"No region named {name!r} in the config. "
                f"Run `2kw calibrate` to define it."
            ) from None

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            # Fall back to the checked-in example so a fresh clone still runs.
            if EXAMPLE_CONFIG_PATH.exists():
                path = EXAMPLE_CONFIG_PATH
            else:
                raise FileNotFoundError(
                    f"No config at {path}. Copy config/regions.example.yaml "
                    f"to config/regions.yaml and calibrate it."
                )

        raw = yaml.safe_load(path.read_text()) or {}
        regions = {
            name: Region(name=name, **spec)
            for name, spec in (raw.get("regions") or {}).items()
        }
        return cls(regions=regions, capture=raw.get("capture") or {}, raw=raw)

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.raw)
        payload["capture"] = self.capture
        payload["regions"] = {
            r.name: {"x": round(r.x, 5), "y": round(r.y, 5),
                     "w": round(r.w, 5), "h": round(r.h, 5)}
            for r in self.regions.values()
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=True))
        return path
