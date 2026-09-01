"""Saving a clip of each shot.

The lookback the shooter search already keeps is, for free, a recording of the
shot that produced it. Writing it out is what makes a logged shot checkable by
a human rather than merely plausible — and it is the only way to tell whether
an attributed shooter is the right one.
"""

import time

import cv2
import numpy as np
import pytest

from twokwatcher.config import Config
from twokwatcher.pipeline import EventBus
from twokwatcher.pipeline.runner import CLIP_FPS, Runner
from twokwatcher.hud.shotfeedback import ShotFeedback


class FakeSource:
    fps = 60.0
    def __iter__(self): return iter(())
    def read(self): return None
    def close(self): pass


class FakeFrame:
    def __init__(self, index):
        # Distinct content per frame, so a clip's frames can be told apart.
        self.image = np.full((240, 320, 3), index % 250, np.uint8)
        self.index = index
        self.timestamp = index * 0.1


class StubBanner:
    def __init__(self, flags):
        self.flags = flags

    def available(self): return True

    def present(self, image):
        return bool(self.flags[int(image[0, 0, 0]) % len(self.flags)])

    def read_event(self, frames):
        return ShotFeedback(timing="EXCELLENT", coverage="OPEN",
                            distance_feet=22.3)


def build(tmp_path, flags, clips=True):
    bus = EventBus()
    seen = []
    bus.subscribe("shot_feedback", seen.append)
    runner = Runner(FakeSource(), Config.load(), bus=bus, sample_fps=60,
                    clip_dir=(tmp_path / "clips") if clips else None)
    runner.shot_feedback = StubBanner(flags)
    runner._shot_reader_ready = True
    runner._plate_reader_ready = False        # isolate clips from plate reading
    return runner, seen


def feed(runner, flags):
    for i, _ in enumerate(flags):
        runner._collect_shot_feedback(FakeFrame(i))


def settle():
    """Clips encode on a worker thread."""
    time.sleep(1.2)


def test_a_shot_writes_a_clip(tmp_path):
    flags = [0, 0, 0, 1, 1, 0]
    runner, seen = build(tmp_path, flags)
    feed(runner, flags)
    settle()

    assert len(seen) == 1
    path = seen[0].data["clip"]
    assert path is not None
    from pathlib import Path
    assert Path(path).exists(), "clip file was never written"


def test_the_clip_holds_the_run_up_and_the_banner(tmp_path):
    """A clip that stopped at the release would not show the verdict."""
    flags = [0, 0, 0, 1, 1, 0]
    runner, seen = build(tmp_path, flags)
    feed(runner, flags)
    settle()

    cap = cv2.VideoCapture(seen[0].data["clip"])
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    # three frames of run-up plus two of banner
    assert count == 5
    assert abs(fps - CLIP_FPS) < 0.51


def test_clips_can_be_turned_off(tmp_path):
    flags = [0, 0, 1, 0]
    runner, seen = build(tmp_path, flags, clips=False)
    feed(runner, flags)
    settle()
    assert seen[0].data["clip"] is None
    assert not (tmp_path / "clips").exists()


def test_writing_does_not_block_the_frame_loop(tmp_path):
    flags = [0] * 12 + [1, 0]
    runner, _ = build(tmp_path, flags)
    started = time.perf_counter()
    feed(runner, flags)
    assert time.perf_counter() - started < 1.0


def test_two_shots_do_not_share_a_filename(tmp_path):
    flags = [0, 0, 1, 0, 0, 1, 0]
    runner, seen = build(tmp_path, flags)
    feed(runner, flags)
    settle()
    assert len(seen) == 2
    paths = {e.data["clip"] for e in seen}
    assert len(paths) == 2, "one shot overwrote another"


def test_a_clip_directory_is_created_on_demand(tmp_path):
    target = tmp_path / "nested" / "clips"
    bus = EventBus()
    seen = []
    bus.subscribe("shot_feedback", seen.append)
    runner = Runner(FakeSource(), Config.load(), bus=bus, sample_fps=60,
                    clip_dir=target)
    runner.shot_feedback = StubBanner([0, 0, 1, 0])
    runner._shot_reader_ready = True
    runner._plate_reader_ready = False
    feed(runner, [0, 0, 1, 0])
    settle()
    assert target.exists()
