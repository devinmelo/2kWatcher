"""Live capture, by way of the OBS virtual camera.

We deliberately do not open the Elgato device directly. Its driver generally
allows a single consumer, and OBS is going to want it — for the replay buffer
that powers highlight clipping, and for NVENC recording that costs us no CUDA.
So OBS owns the card and we read its virtual camera output.
"""

from __future__ import annotations

import sys
import time

import cv2

from .source import Frame, FrameSource


class VirtualCamSource(FrameSource):
    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        # DirectShow is markedly better behaved than MSMF for virtual cameras.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(device_index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open video device {device_index}. "
                f"Is OBS running with 'Start Virtual Camera' enabled? "
                f"Run `2kw devices` to list what is available."
            )
        self._index = 0
        self._t0 = time.monotonic()

    @property
    def fps(self) -> float:
        reported = self._cap.get(cv2.CAP_PROP_FPS)
        # Virtual cameras frequently report 0 or something nonsensical.
        return reported if 1.0 < reported < 1000.0 else 60.0

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        frame = Frame(
            image=image,
            index=self._index,
            timestamp=time.monotonic() - self._t0,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def list_devices(max_index: int = 8) -> list[dict]:
    """Probe device indices and report which ones yield a frame.

    Crude, but there is no portable way to enumerate capture devices through
    OpenCV, and this is enough to find which index is the OBS virtual camera.
    """
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    found = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, backend)
        try:
            if not cap.isOpened():
                continue
            ok, image = cap.read()
            if not ok or image is None:
                continue
            h, w = image.shape[:2]
            found.append({"index": index, "width": w, "height": h,
                          "fps": cap.get(cv2.CAP_PROP_FPS)})
        finally:
            cap.release()
    return found
