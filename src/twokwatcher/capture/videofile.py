"""Replay a recorded file as if it were live.

This is the source you will actually use most. Parsers get developed against
saved footage, deterministically and repeatably, rather than against a live
game you are simultaneously trying to play.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from .source import Frame, FrameSource


class VideoFileSource(FrameSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No video at {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not decode {self.path}")
        self._index = 0

    @property
    def fps(self) -> float:
        reported = self._cap.get(cv2.CAP_PROP_FPS)
        return reported if 1.0 < reported < 1000.0 else 60.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def seek(self, frame_index: int) -> None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._index = frame_index

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        frame = Frame(
            image=image,
            index=self._index,
            # Wall-clock is meaningless for a file; use position in the video.
            timestamp=self._index / self.fps,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
