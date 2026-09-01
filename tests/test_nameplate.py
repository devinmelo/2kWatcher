"""Reading the gamertag plates drawn over players.

The fixture is a real 1080p Rec frame showing both plates 2K draws: your own
player, and separately the player holding the ball. That pair is the whole
basis for attributing a shot to a shooter, so it is worth pinning to a frame
rather than a synthetic construction.
"""

import cv2
import numpy as np
import pytest

from twokwatcher.hud import NameplateReader
from twokwatcher.hud.nameplate import _resolve, _strongest_apart

FIXTURE = "tests/fixtures/nameplates_two_1080p.png"
ROSTER = ["Lil Knotty", "njsimas456", "colby2818", "Hurricane091798",
          "AnRaDe728", "MxreDiff", "who Wetz", "whyygdk", "Raqbully",
          "stophesii"]
ME = "Lil Knotty"


def load():
    image = cv2.imread(FIXTURE)
    assert image is not None, f"missing fixture: {FIXTURE}"
    return image


@pytest.fixture(scope="module")
def reader():
    r = NameplateReader()
    if not r.available():
        pytest.skip("tesseract is not installed")
    return r


# --- finding plates -----------------------------------------------------

def test_both_plates_are_found(reader):
    plates = [p for p in reader.read(load(), ROSTER) if p.resolved]
    assert {p.name for p in plates} == {"Lil Knotty", "colby2818"}


def test_a_distant_plate_is_found_at_a_smaller_scale(reader):
    """Plates shrink with distance; one template size finds only the near one."""
    plates = {p.name: p for p in reader.read(load(), ROSTER) if p.resolved}
    assert plates["Lil Knotty"].scale == 1.0
    assert plates["colby2818"].scale < 1.0


def test_the_ball_handler_is_the_plate_that_is_not_yours(reader):
    handler = reader.ball_handler(load(), ROSTER, ME)
    assert handler is not None
    assert handler.name == "colby2818"


def test_a_blank_frame_has_no_plates(reader):
    assert reader.read(np.zeros((1080, 1920, 3), np.uint8), ROSTER) == []
    assert reader.ball_handler(np.zeros((1080, 1920, 3), np.uint8),
                               ROSTER, ME) is None


# --- resolving against the roster ---------------------------------------

def test_a_mangled_reading_still_resolves(reader):
    """OCR returns '-Il Knotty' for this plate; ten candidates make that enough."""
    plates = {p.name: p for p in reader.read(load(), ROSTER) if p.resolved}
    assert plates["Lil Knotty"].raw != "Lil Knotty"


def test_a_name_not_in_the_game_is_refused():
    """A plate from another lobby must not be snapped onto this roster."""
    assert _resolve("DidntDoTheMath", ROSTER) is None
    assert _resolve("Applause Needed", ROSTER) is None


def test_resolution_is_case_insensitive_and_exact_wins():
    assert _resolve("colby2818", ROSTER) == "colby2818"
    assert _resolve("COLBY2818", ROSTER) == "colby2818"


def test_nothing_resolves_without_a_roster():
    """The roster is what makes a reading trustworthy; without it, abstain."""
    assert _resolve("colby2818", None) is None
    assert _resolve("colby2818", []) is None


def test_empty_readings_resolve_to_nothing():
    assert _resolve("", ROSTER) is None


# --- badge de-duplication -----------------------------------------------

def test_one_badge_is_reported_once():
    """Template matching hits the same badge at several offsets and scales."""
    cluster = [(100, 100, 0.9, 1.0), (103, 101, 0.8, 0.85),
               (99, 104, 0.7, 0.7), (400, 400, 0.65, 1.0)]
    kept = _strongest_apart(cluster, 24)
    assert len(kept) == 2
    assert kept[0][:3] == (100, 100, 0.9)


def test_the_strongest_match_in_a_cluster_wins():
    cluster = [(100, 100, 0.6, 1.0), (104, 100, 0.95, 0.7)]
    kept = _strongest_apart(cluster, 24)
    assert len(kept) == 1
    assert kept[0][2] == 0.95
