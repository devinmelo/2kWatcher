"""Tests against real Rec gameplay frames.

The fixtures come from a 1180x664 screen recording — smaller and more heavily
compressed than a real 1080p capture — so they exercise the parser closer to
its worst case than to its best.
"""

import cv2
import pytest

from twokwatcher.hud import ShotFeedbackReader
from twokwatcher.hud.shotfeedback import (
    _find_value,
    _parse_distance,
    COVERAGE_VALUES,
    TIMING_VALUES,
)

# Sampled uniformly across each event, as the pipeline would - not
# cherry-picked, so these reflect real performance rather than a best case.
SHOT1 = [f"tests/fixtures/rec_shot1_{k}.jpg" for k in range(8)]
SHOT2 = [f"tests/fixtures/rec_shot2_{k}.jpg" for k in range(8)]
NO_SHOT = [f"tests/fixtures/rec_noshot_{s}.jpg" for s in "ab"]


def load(paths):
    images = [cv2.imread(p) for p in paths]
    assert all(i is not None for i in images), f"missing fixtures: {paths}"
    return images


@pytest.fixture(scope="module")
def reader():
    r = ShotFeedbackReader()
    if not r.available():
        pytest.skip("tesseract is not installed")
    return r


def test_banner_is_detected_across_a_shot(reader):
    """Detection is of the event, not of every individual frame.

    The banner animates in, so the first frame or two of an event sit just
    under the threshold. That costs nothing: an event lasts dozens of frames
    and only has to be noticed once. The threshold is set where it is because
    it produced no false positives at all over the source recording, and a
    false positive is far more expensive than a late one.
    """
    for shot in (SHOT1, SHOT2):
        detected = [reader.present(i) for i in load(shot)]
        assert sum(detected) >= len(detected) - 1
        # Once it is up, it stays detected.
        assert all(detected[2:])


def test_no_banner_during_ordinary_play(reader):
    """Presence must not fire on the crowd and court alone."""
    for image in load(NO_SHOT):
        assert not reader.present(image)


def test_open_shot_reads_by_consensus(reader):
    feedback = reader.read_event(load(SHOT1))
    assert feedback.timing == "EXCELLENT"
    assert feedback.coverage == "OPEN"


def test_contested_shot_reads_by_consensus(reader):
    feedback = reader.read_event(load(SHOT2))
    assert feedback.timing == "EXCELLENT"
    assert feedback.coverage == "LIGHT CONTEST"


def test_most_frames_abstain_but_none_lie(reader):
    """The safety property: a frame that cannot read returns None, not a guess.

    On this low-resolution footage most frames abstain. That is fine, and much
    better than the alternative - a wrong verdict entering the log is far worse
    than a missing one, and consensus only works because the errors are
    omissions rather than mistakes.
    """
    for shot, expected in ((SHOT1, "OPEN"), (SHOT2, "LIGHT CONTEST")):
        readings = [reader.read(i).coverage for i in load(shot)]
        assert any(r is None for r in readings), "fixtures are too clean"
        assert {r for r in readings if r} == {expected}


def test_consensus_beats_the_single_frames_it_is_built_from(reader):
    """Most individual frames read nothing; the event still resolves."""
    singles = [reader.read(i).timing for i in load(SHOT2)]
    assert singles.count(None) >= len(singles) // 2
    assert reader.read_event(load(SHOT2)).timing == "EXCELLENT"


def test_a_short_plain_value_is_not_widened(reader):
    """'OPEN' must not be read as 'WIDE OPEN' just because it fits inside it."""
    assert reader.read_event(load(SHOT1)).coverage == "OPEN"


@pytest.mark.parametrize("text,expected", [
    ("TIMING EXCELLENT COVERAGE OPEN", "EXCELLENT"),
    ("GXCELLENT OOEN", "EXCELLENT"),          # observed OCR mangling
    ("TIMING SLIGHTLY LATE", "SLIGHTLY LATE"),
    ("", None),
    ("qwerty zxcvbn", None),                  # noise must not match anything
])
def test_timing_vocabulary_matching(text, expected):
    assert _find_value(text, TIMING_VALUES) == expected


@pytest.mark.parametrize("text,expected", [
    ("COVERAGE LIGHT CONTEST", "LIGHT CONTEST"),
    ("UGHT CONTEST", "LIGHT CONTEST"),        # observed OCR mangling
    ("COVERAGE WIDE OPEN", "WIDE OPEN"),
    ("COVERAGE OPEN", "OPEN"),
    ("", None),
])
def test_coverage_vocabulary_matching(text, expected):
    assert _find_value(text, COVERAGE_VALUES) == expected


@pytest.mark.parametrize("text,expected", [
    ("DISTANCE 24'6\"", 24.5),
    ("DISTANCE 3'0\"", 3.0),
    ("DISTANCE 12'", 12.0),
    ("no distance here", None),
    ("99'99\"", None),                        # out of range, so rejected
])
def test_distance_parsing(text, expected):
    assert _parse_distance(text) == expected


def test_unknown_values_are_kept_rather_than_forced(reader):
    """An unseen verdict must surface as unknown, not as the nearest known one."""
    from twokwatcher.hud.shotfeedback import ShotFeedback
    assert _find_value("TIMING PERFECTLY SUBLIME", TIMING_VALUES) is None
    assert ShotFeedback().any_read is False
