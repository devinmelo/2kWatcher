"""Tests against real panel crops from a 1080p Rec capture.

The fixtures are the panel region exactly as `Config.region("player_panel")`
crops it, so what these exercise is what the pipeline sees.

The parts that need neither a glyph atlas nor Tesseract — finding the columns,
identifying the layout, refusing a cross-fade — are tested unconditionally.
The parts that need them skip when they are absent, as elsewhere.
"""

import cv2
import numpy as np
import pytest

from twokwatcher.config import Config
from twokwatcher.hud import PlayerPanelReader
from twokwatcher.hud.playerpanel import (
    GRADES,
    STAT_LABELS,
    _closest,
    _consensus,
)

GAME_STATS = "tests/fixtures/panel_game_stats.png"
TOP_PERFORMERS = "tests/fixtures/panel_top_performers.png"
MIDFADE = "tests/fixtures/panel_midfade.png"
ATTRIBUTES = "tests/fixtures/panel_attributes.png"


def load(path):
    image = cv2.imread(path)
    assert image is not None, f"missing fixture: {path}"
    return image


@pytest.fixture(scope="module")
def reader():
    return PlayerPanelReader(Config.load())


# --- column arrangement -------------------------------------------------

def test_single_player_arrangement_is_found(reader):
    assert reader.read_panel(load(GAME_STATS)).columns == 5


def test_two_player_arrangement_is_found(reader):
    assert reader.read_panel(load(TOP_PERFORMERS)).columns == 8


def test_a_cross_fade_is_refused_rather_than_guessed(reader):
    """Half-faded frames are exactly where a confident wrong read comes from."""
    panel = reader.read_panel(load(MIDFADE))
    assert panel.layout is None
    assert panel.players == []


def test_arrangement_alone_never_claims_a_layout(reader):
    """Five columns is a shape, not a meaning — the attributes panel has five too."""
    for path in (GAME_STATS, ATTRIBUTES):
        panel = reader.read_panel(load(path))
        assert panel.columns == 5
        if not reader._ocr_available():
            # Nothing has confirmed what these columns hold.
            assert panel.layout is None


def test_attributes_panel_is_never_read_as_a_stat_line(reader):
    """DUNK 94 must never become 94 points."""
    if not reader._ocr_available():
        pytest.skip("tesseract is not installed")
    panel = reader.read_panel(load(ATTRIBUTES))
    assert panel.layout is None
    assert panel.players == []


def test_game_stats_panel_is_confirmed_by_its_labels(reader):
    if not reader._ocr_available():
        pytest.skip("tesseract is not installed")
    single = reader.read_panel(load(GAME_STATS))
    assert single.layout == "game_stats"
    assert len(single.players) == 1

    dual = reader.read_panel(load(TOP_PERFORMERS))
    assert dual.layout == "top_performers"
    assert len(dual.players) == 2


def test_columns_are_found_where_they_are_drawn(reader):
    """The layout is discovered, so the discovered columns must be the real ones."""
    image = load(GAME_STATS)
    columns = reader._label_columns(image, 1.0, 1.0)
    assert len(columns) == 5
    # Measured centres, panel-relative, from the 1080p capture.
    for found, expected in zip(columns, (102, 174, 247, 318, 388)):
        assert abs(found - expected) <= 8, f"{found} vs {expected}"


def test_two_player_columns_split_evenly_between_players(reader):
    columns = reader._label_columns(load(TOP_PERFORMERS), 1.0, 1.0)
    assert len(columns) == 8
    first, second = columns[:4], columns[4:]
    # The second player's block is the first one shifted by a fixed offset.
    offsets = [b - a for a, b in zip(first, second)]
    assert max(offsets) - min(offsets) <= 6


def test_an_empty_panel_reads_nothing(reader):
    blank = reader.read_panel(np.zeros((92, 458, 3), np.uint8))
    assert (blank.columns, blank.layout) == (0, None)
    assert reader.read_panel(np.zeros((0, 0, 3), np.uint8)).layout is None


# --- the settle gate ----------------------------------------------------

def test_panel_is_not_settled_on_the_first_frame(reader):
    reader.reset()
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert reader.settled(frame) is False


def test_an_unchanged_panel_settles_and_a_changed_one_does_not(reader):
    reader.reset()
    still = np.zeros((1080, 1920, 3), np.uint8)
    assert reader.settled(still) is False       # nothing to compare against
    assert reader.settled(still) is True        # unchanged since

    moved = still.copy()
    region = Config.load().region("player_panel")
    x, y, w, h = region.to_pixels(1920, 1080)
    moved[y:y + h, x:x + w] = 255
    assert reader.settled(moved) is False
    reader.reset()


# --- vocabulary matching ------------------------------------------------

def test_stat_labels_snap_to_known_values():
    assert _closest("PTS", STAT_LABELS) == "PTS"
    assert _closest("3P%", STAT_LABELS) == "3P%"


def test_a_confused_glyph_does_not_shorten_the_label():
    """"0REB" is nearer OREB than REB, but plain difflib prefers the shorter."""
    assert _closest("0REB", STAT_LABELS) == "OREB"
    assert _closest("REB", STAT_LABELS) == "REB"


def test_digits_in_a_real_label_survive_correction():
    """The confusion table must not eat labels that legitimately start with a digit."""
    assert _closest("3PM", STAT_LABELS) == "3PM"
    assert _closest("3PA", STAT_LABELS) == "3PA"


def test_an_unknown_label_is_dropped_not_snapped():
    """Filing a stat under the wrong label silently corrupts the numbers."""
    assert _closest("XYZZY", STAT_LABELS) is None
    assert _closest("", STAT_LABELS) is None
    assert _closest(None, STAT_LABELS) is None


def test_grades_snap_to_known_values():
    assert _closest("A-", GRADES) == "A-"
    assert _closest("b+", GRADES) == "B+"


def test_consensus_needs_a_majority():
    assert _consensus(["A", "A", "B"]) == "A"
    assert _consensus([None, None]) is None
    # A genuine three-way split is not a reading.
    assert _consensus(["A", "B", "C", "D"]) is None


# --- reading, where the machine can -------------------------------------

def test_nothing_is_invented_without_an_atlas_or_ocr(reader):
    """A scaffold that invents stats would be worse than one that admits it."""
    if reader.ready and reader._ocr_available():
        pytest.skip("atlas and OCR both present, so real values are readable")
    panel = reader.read_panel(load(GAME_STATS))
    for line in panel.players:
        assert (line.points, line.rebounds, line.assists) == (None, None, None)


def test_known_line_reads_correctly(reader):
    if not reader.ready:
        pytest.skip("no glyph atlas; run `2kw atlas` first")
    if not reader._ocr_available():
        pytest.skip("tesseract is not installed")
    # Hand-read from the fixture: Lil Knotty, PG, grade A, 13 PTS 0 REB 8 AST,
    # with 3P% as the varying fifth column.
    line = reader.read_panel(load(GAME_STATS)).players[0]
    assert line.points == 13
    assert line.rebounds == 0
    assert line.assists == 8
    assert line.grade == "A"
    assert line.extra_label == "3P%"


def test_rotation_consensus_survives_one_bad_frame(reader):
    """One mangled frame in a slot must not decide the slot."""
    if not reader._ocr_available():
        pytest.skip("tesseract is not installed")
    good = load(GAME_STATS)
    fade = load(MIDFADE)
    merged = reader.read_rotation([good, good, fade, good])
    assert merged.layout == "game_stats"
    assert len(merged.players) == 1


def test_rotation_reports_the_arrangement_even_when_unconfirmed(reader):
    """Without OCR there is still a shape to report, just no layout."""
    good = load(GAME_STATS)
    merged = reader.read_rotation([good, good, good])
    assert merged.columns == 5
