"""Detecting that a box score is on screen at all.

This is what lets the state machine reach POST_GAME. GameState has had the
member since the beginning but nothing ever returned it, so the box score
parser — the densest data 2K gives you — could never be triggered by the
pipeline.

Density heuristics do not work here and it is worth recording why, so nobody
re-tries them: measured over 157 frames from three capture sessions, a box
score and a crowded gameplay frame have indistinguishable text-line counts
(16 against a gameplay maximum of 19) and indistinguishable edge densities.
What does separate them is the column-header row, which both the MyCareer
post-game recap and the Rec/Crews mid-game overlay draw identically.
"""

import cv2
import numpy as np
import pytest

from twokwatcher.config import Config
from twokwatcher.hud.boxscore import (
    HEADER_MATCH_THRESHOLD,
    header_score,
    is_box_score,
)
from twokwatcher.state import GameState
from twokwatcher.state.detectors import ScreenClassifier

BOX_SCREENS = [
    "tests/fixtures/postgame_mycareer_1080p.png",
    "tests/fixtures/gamestats_rec_1080p.png",
]
GAMEPLAY = [
    "tests/fixtures/rec_shot1_0.jpg",
    "tests/fixtures/rec_noshot_a.jpg",
]


def load(path):
    image = cv2.imread(path)
    assert image is not None, f"missing fixture: {path}"
    return image


@pytest.mark.parametrize("path", BOX_SCREENS)
def test_a_box_score_screen_is_detected(path):
    assert is_box_score(load(path))


@pytest.mark.parametrize("path", GAMEPLAY)
def test_gameplay_is_not_a_box_score(path):
    assert not is_box_score(load(path))


def test_a_blank_frame_is_not_a_box_score():
    assert not is_box_score(np.zeros((1080, 1920, 3), np.uint8))


def test_the_threshold_sits_in_the_gap_not_on_an_edge():
    """A threshold hard against the worst positive is one bad frame from failing."""
    worst_positive = min(header_score(load(p)) for p in BOX_SCREENS)
    best_negative = max(header_score(load(p)) for p in GAMEPLAY)
    assert best_negative < HEADER_MATCH_THRESHOLD < worst_positive
    # Comfortably inside, not scraping either side.
    assert worst_positive - HEADER_MATCH_THRESHOLD > 0.1
    assert HEADER_MATCH_THRESHOLD - best_negative > 0.1


def test_the_detector_survives_a_resolution_change():
    """Regions are normalized so a different capture size costs nothing."""
    image = load(BOX_SCREENS[0])
    smaller = cv2.resize(image, (1280, 720))
    assert is_box_score(smaller)


@pytest.mark.parametrize("path", BOX_SCREENS)
def test_the_classifier_reaches_post_game(path):
    state, _ = ScreenClassifier(Config.load()).classify(load(path))
    assert state is GameState.POST_GAME


def test_the_box_score_is_found_even_over_a_live_scoreboard():
    """2K can open the box score mid-game, with the scoreboard still beneath it.

    Six of the box screens on disk are that overlay. They read as
    scoreboard_present, so a check placed in the menu branch never sees them,
    which is why the box-score test runs before the scoreboard signal.
    """
    image = load("tests/fixtures/gamestats_over_live_1080p.png")
    classifier = ScreenClassifier(Config.load())
    signals = classifier.signals(image)
    assert signals.scoreboard_present, "fixture no longer shows the plate"
    state, _ = classifier.classify(image)
    assert state is GameState.POST_GAME
