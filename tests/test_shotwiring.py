"""The shot banner reaching the pipeline as an event.

Covers the wiring rather than the parser: that a banner appearing and clearing
produces exactly one event, that a banner still on screen produces none yet,
and that the buffer cannot grow without bound.
"""

import numpy as np
import pytest

from twokwatcher.config import Config
from twokwatcher.hud.shotfeedback import ShotFeedback
from twokwatcher.pipeline import EventBus
from twokwatcher.pipeline.runner import MAX_BANNER_FRAMES, Runner


class FakeSource:
    fps = 60.0

    def __init__(self, frames):
        self._frames = frames

    def __iter__(self):
        from twokwatcher.capture import Frame
        for i, image in enumerate(self._frames):
            yield Frame(image=image, index=i, timestamp=i / self.fps)

    def read(self):
        raise NotImplementedError

    def close(self):
        pass


class StubReader:
    """Stands in for ShotFeedbackReader: scripted presence, canned reading."""

    def __init__(self, present_flags, reading):
        self._present = list(present_flags)
        self._reading = reading
        self.read_calls = []

    def available(self):
        return True

    def present(self, image):
        return bool(self._present[int(image[0, 0, 0])])

    def read_event(self, frames):
        self.read_calls.append(len(frames))
        return self._reading


def frames_for(flags):
    """One tiny frame per flag, tagged with its index so the stub can see it."""
    out = []
    for i, _ in enumerate(flags):
        f = np.zeros((8, 8, 3), np.uint8)
        f[:, :, 0] = i
        out.append(f)
    return out


def run_with(flags, reading):
    bus = EventBus()
    seen = []
    bus.subscribe("shot_feedback", seen.append)

    runner = Runner(FakeSource(frames_for(flags)), Config.load(),
                    bus=bus, sample_fps=60)
    runner.shot_feedback = StubReader(flags, reading)
    runner._shot_reader_ready = True
    # The banner stage is driven directly here; the active-state gate that
    # guards it lives in _process and is covered by the pipeline tests.
    for frame in FakeSource(frames_for(flags)):
        runner._collect_shot_feedback(frame)
    return runner, seen


GOOD = ShotFeedback(timing="EXCELLENT", coverage="OPEN", distance_feet=22.3)
UNREAD = ShotFeedback()


def test_a_banner_that_clears_publishes_one_event():
    runner, seen = run_with([False, True, True, True, False, False], GOOD)
    assert len(seen) == 1
    assert seen[0].data["timing"] == "EXCELLENT"
    assert seen[0].data["coverage"] == "OPEN"
    assert seen[0].data["frames"] == 3
    assert runner.stats.shots == 1


def test_a_banner_still_on_screen_publishes_nothing_yet():
    """The event is the whole banner, so it cannot be emitted mid-shot."""
    runner, seen = run_with([False, True, True, True], GOOD)
    assert seen == []
    assert len(runner._banner) == 3


def test_two_banners_produce_two_events():
    _, seen = run_with([True, False, True, True, False], GOOD)
    assert len(seen) == 2


def test_an_unread_banner_is_not_logged():
    """A row of nulls is worse than no row."""
    runner, seen = run_with([False, True, True, False], UNREAD)
    assert seen == []
    assert runner.stats.shots == 0


def test_the_buffer_is_capped():
    """A presence gate stuck on must not grow the buffer without bound."""
    flags = [True] * (MAX_BANNER_FRAMES + 25)
    runner, seen = run_with(flags, GOOD)
    assert seen == []
    assert len(runner._banner) == MAX_BANNER_FRAMES


def test_a_read_that_raises_does_not_kill_the_loop():
    class Exploding(StubReader):
        def read_event(self, frames):
            raise RuntimeError("tesseract fell over")

    bus = EventBus()
    seen = []
    bus.subscribe("shot_feedback", seen.append)
    flags = [False, True, True, False]
    runner = Runner(FakeSource(frames_for(flags)), Config.load(),
                    bus=bus, sample_fps=60)
    runner.shot_feedback = Exploding(flags, GOOD)
    runner._shot_reader_ready = True
    for frame in FakeSource(frames_for(flags)):
        runner._collect_shot_feedback(frame)      # must not raise
    assert seen == []


def test_nothing_runs_when_tesseract_is_missing():
    class Unavailable(StubReader):
        def available(self):
            return False

    flags = [True, True, False]
    runner = Runner(FakeSource(frames_for(flags)), Config.load(), sample_fps=60)
    runner.shot_feedback = Unavailable(flags, GOOD)
    runner._shot_reader_ready = None
    for frame in FakeSource(frames_for(flags)):
        runner._collect_shot_feedback(frame)
    assert runner._banner == []
    assert runner.stats.shots == 0
