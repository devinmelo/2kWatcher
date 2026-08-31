"""Naming the player who took a shot.

2K names two players at any moment: your own, wherever they are, and whoever
has the ball. So the shooter names himself — right up until he releases, at
which point his plate goes and the banner arrives in its place. Measured on a
real session, 22 of 60 banner frames carried no plate at all, which is why the
shooter is looked for in the frames leading up to a banner rather than in the
banner itself.

These cover the buffering and the lookback. Reading a plate is covered by
test_nameplate.
"""

import numpy as np
import pytest

from twokwatcher.config import Config
from twokwatcher.hud.nameplate import Nameplate
from twokwatcher.pipeline import EventBus
from twokwatcher.pipeline.runner import (
    SHOOTER_BAND,
    SHOOTER_LOOKBACK_FRAMES,
    SHOOTER_SEARCH_DEPTH,
    Runner,
)
from twokwatcher.hud.shotfeedback import ShotFeedback

ROSTER = ["Lil Knotty", "colby2818", "IIMarrll"]
ME = "Lil Knotty"


class FakeSource:
    fps = 60.0
    def __iter__(self): return iter(())
    def read(self): return None
    def close(self): pass


class FakeFrame:
    def __init__(self, index, tag=0):
        self.image = np.full((1080, 1920, 3), tag, np.uint8)
        self.index = index
        self.timestamp = index * 0.1


class StubBanner:
    """Scripted presence, so a shot can be made to start and stop on cue."""

    def __init__(self, flags, reading):
        self.flags = flags
        self.reading = reading

    def available(self): return True

    def present(self, image):
        return bool(self.flags[int(image[0, 0, 0])])

    def read_event(self, frames):
        return self.reading


class StubPlates:
    """Returns a scripted handler per frame tag."""

    def __init__(self, by_tag):
        self.by_tag = by_tag
        self.calls = []

    def available(self): return True

    def ball_handler(self, band, roster, me):
        tag = int(band[0, 0, 0])
        self.calls.append(tag)
        name = self.by_tag.get(tag)
        if name is None:
            return None
        return Nameplate(name=name, raw=name, x=0, y=0, confidence=0.9)


GOOD = ShotFeedback(timing="EXCELLENT", coverage="OPEN", distance_feet=22.3)


def build(flags, by_tag, roster=ROSTER, me=ME):
    bus = EventBus()
    seen = []
    bus.subscribe("shot_feedback", seen.append)
    runner = Runner(FakeSource(), Config.load(), bus=bus, sample_fps=60)
    runner.shot_feedback = StubBanner(flags, GOOD)
    runner.nameplates = StubPlates(by_tag)
    runner._shot_reader_ready = True
    runner._plate_reader_ready = True
    runner.set_roster(roster, me)
    return runner, seen


def feed(runner, flags):
    for i, _ in enumerate(flags):
        runner._collect_shot_feedback(FakeFrame(i, tag=i))


# --- the buffer ---------------------------------------------------------

def test_only_non_banner_frames_are_remembered():
    """Banner frames are the shot; the lookback is what came before it."""
    flags = [0, 0, 1, 1, 0]
    runner, _ = build(flags, {})
    feed(runner, flags)
    remembered = [index for index, _ in runner._recent]
    assert 2 not in remembered and 3 not in remembered


def test_the_buffer_does_not_grow_without_bound():
    flags = [0] * (SHOOTER_LOOKBACK_FRAMES + 40)
    runner, _ = build(flags, {})
    feed(runner, flags)
    assert len(runner._recent) == SHOOTER_LOOKBACK_FRAMES


def test_only_a_band_of_each_frame_is_kept():
    """Plates live over players; the crowd and scoreboard are not worth holding."""
    flags = [0]
    runner, _ = build(flags, {})
    feed(runner, flags)
    _, band = runner._recent[0]
    expected = int(SHOOTER_BAND[1] * 1080) - int(SHOOTER_BAND[0] * 1080)
    assert band.shape[0] == expected
    assert band.shape[0] < 1080


# --- finding the shooter ------------------------------------------------

def test_the_shooter_comes_from_before_the_banner():
    flags = [0, 0, 1, 0]
    runner, seen = build(flags, {0: ME, 1: "colby2818"})
    feed(runner, flags)
    assert len(seen) == 1
    assert seen[0].data["shooter"] == "colby2818"


def test_the_most_recent_holder_wins():
    """Whoever had it last is the shooter, not whoever had it a possession ago."""
    flags = [0, 0, 0, 1, 0]
    runner, seen = build(flags, {0: "IIMarrll", 1: "IIMarrll", 2: "colby2818"})
    feed(runner, flags)
    assert seen[0].data["shooter"] == "colby2818"


def test_only_your_plate_showing_means_the_shot_was_yours():
    """Your plate is always drawn; a second appears only for the ball handler."""
    flags = [0, 0, 1, 0]
    runner, seen = build(flags, {0: ME, 1: ME})
    feed(runner, flags)
    assert seen[0].data["shooter"] == ME


def test_the_search_does_not_run_past_its_depth():
    """Looking too far back names whoever had it on the previous possession."""
    flags = [0] * 12 + [1, 0]
    by_tag = {0: "IIMarrll"}          # only the oldest frame has a handler
    runner, seen = build(flags, by_tag)
    feed(runner, flags)
    assert len(runner.nameplates.calls) <= SHOOTER_SEARCH_DEPTH
    assert seen[0].data["shooter"] == ME


def test_no_roster_means_no_shooter():
    """Naming is only reliable because the candidates are known."""
    flags = [0, 0, 1, 0]
    runner, seen = build(flags, {0: "colby2818"}, roster=[], me=None)
    feed(runner, flags)
    assert seen[0].data["shooter"] is None
    assert runner.stats.shots == 1
    assert runner.stats.shots_attributed == 0


def test_an_attributed_shot_is_counted():
    flags = [0, 0, 1, 0]
    runner, seen = build(flags, {0: "colby2818"})
    feed(runner, flags)
    assert (runner.stats.shots, runner.stats.shots_attributed) == (1, 1)


def test_the_roster_arrives_from_the_box_score():
    runner, _ = build([0], {})
    runner.set_roster(["a", "b"], "a")
    assert runner.roster == ["a", "b"]
    assert runner.me == "a"
