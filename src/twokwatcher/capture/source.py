"""The frame-source abstraction.

The whole point of this module is that nothing downstream knows or cares where
frames come from. Today that is the OBS virtual camera; later it may be an
NVDEC-backed GPU path that never touches host memory. Swapping that out should
not require touching the state machine, the HUD parsers, or the tracker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class Frame:
    """One captured frame plus the metadata downstream stages need."""

    image: np.ndarray          # BGR, HxWx3, as OpenCV hands it to us
    index: int                 # monotonic frame counter for this source
    timestamp: float           # seconds since the source was opened

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


class FrameSource(ABC):
    """A stream of frames, iterable and closeable."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or None once the source is exhausted."""

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def fps(self) -> float:
        """Native frame rate of the source, best-effort."""

    def __iter__(self) -> Iterator[Frame]:
        try:
            while (frame := self.read()) is not None:
                yield frame
        finally:
            self.close()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_source(
    kind: str = "virtualcam",
    *,
    device_index: int = 0,
    video_path: str | Path | None = None,
) -> FrameSource:
    """Build a frame source by name.

    'virtualcam' reads live from the capture path; 'file' replays a recording,
    which is how most development happens — you cannot iterate on a parser
    against a live game you are also trying to play.
    """
    from .videofile import VideoFileSource
    from .virtualcam import VirtualCamSource

    if kind == "file":
        if video_path is None:
            raise ValueError("kind='file' needs a video_path")
        return VideoFileSource(video_path)
    if kind == "virtualcam":
        return VirtualCamSource(device_index)
    raise ValueError(f"Unknown source kind {kind!r} (expected 'virtualcam' or 'file')")
