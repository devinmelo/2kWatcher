import numpy as np

from twokwatcher.capture.source import Frame, FrameSource
from twokwatcher.config import Config
from twokwatcher.pipeline import Event, EventBus, Runner


class FakeSource(FrameSource):
    """A synthetic source, so the loop is testable without a capture card."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    @property
    def fps(self):
        return 60.0

    def read(self):
        if self._i >= len(self._frames):
            return None
        image = self._frames[self._i]
        frame = Frame(image=image, index=self._i, timestamp=self._i / 60.0)
        self._i += 1
        return frame

    def close(self):
        pass


def test_event_bus_wildcard_and_kind():
    bus = EventBus()
    seen = []
    bus.subscribe("state_change", lambda e: seen.append(("kind", e.kind)))
    bus.subscribe("*", lambda e: seen.append(("all", e.kind)))
    bus.publish(Event(kind="state_change", frame_index=0, video_ts=0.0))
    assert seen == [("kind", "state_change"), ("all", "state_change")]


def test_runner_samples_below_capture_rate():
    dark = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(60)]
    runner = Runner(FakeSource(dark), Config.load(), sample_fps=10)
    stats = runner.run()
    assert stats.frames_seen == 60
    # 60fps source at 10fps target -> every 6th frame.
    assert stats.frames_sampled == 10


def test_dark_frames_classify_as_loading():
    dark = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(30)]
    stats = Runner(FakeSource(dark), Config.load(), sample_fps=30).run()
    assert stats.state_frames.get("loading", 0) > 0
