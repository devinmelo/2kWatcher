"""Regression tests against a real post-game screen.

The fixture is an actual 1920x1080 capture, so these pin the parser to real
pixels rather than to synthetic ones that would drift from what 2K draws.
"""

import cv2
import pytest

from twokwatcher.hud import BoxScoreParser, resolve_names
from twokwatcher.hud.boxscore import CHECKSUM_COLUMNS, _split_fraction

FIXTURE = "tests/fixtures/postgame_mycareer_1080p.png"

# Hand-read from the fixture: name, team, grade, pts, reb, ast, stl, blk,
# fouls, tov, fg, tp, ft.
EXPECTED = [
    ("MikeSmith2324",   "them", "B",  19,  0,  3, 1, 0, 2, 3, (8, 23), (3, 11), (0, 0)),
    ("AI Player",       "them", "B+",  6,  2,  2, 0, 0, 0, 0, (2, 3),  (2, 2),  (0, 0)),
    ("Shady 3pt0",      "them", "C+",  6,  0,  3, 0, 0, 4, 2, (3, 6),  (0, 1),  (0, 0)),
    ("swaglilfire",     "them", "B-",  5,  4,  4, 1, 0, 0, 1, (2, 6),  (1, 2),  (0, 2)),
    ("bigjimslim207",   "them", "B",   9,  7,  4, 1, 1, 1, 1, (4, 10), (1, 5),  (0, 0)),
    ("Lil Knotty",      "us",   "A+", 27,  0,  9, 0, 0, 1, 0, (12, 17), (3, 6), (0, 0)),
    ("Miranda57095",    "us",   "B+",  8,  0,  2, 0, 0, 0, 0, (3, 5),  (2, 3),  (0, 0)),
    ("colby2818",       "us",   "A",  25,  3,  3, 1, 0, 0, 0, (12, 17), (0, 3), (1, 2)),
    ("Hurricane091798", "us",   "A",  13, 20,  5, 1, 1, 1, 3, (6, 9),  (1, 2),  (0, 0)),
    ("AnRaDe728",       "us",   "A",  10, 13,  6, 0, 1, 0, 1, (5, 9),  (0, 1),  (0, 0)),
]
ROSTER = [name for name, *_ in EXPECTED if name != "AI Player"]


@pytest.fixture(scope="module")
def parsed():
    parser = BoxScoreParser()
    if not parser.available():
        pytest.skip("tesseract is not installed")
    image = cv2.imread(FIXTURE)
    assert image is not None, f"missing fixture {FIXTURE}"
    box = parser.parse(image)
    resolve_names(box, ROSTER)
    return box


def test_all_ten_rows_are_found(parsed):
    assert len(parsed.players) == 10
    assert sum(p.team == "us" for p in parsed.players) == 5


def test_identity_is_read_off_the_screen(parsed):
    """The green and red triangles beat any configured gamertag."""
    assert parsed.you is not None
    assert parsed.you.name == "Lil Knotty"
    assert parsed.you.team == "us"
    matchup = [p for p in parsed.players if p.is_matchup]
    assert [p.name for p in matchup] == ["MikeSmith2324"]


def test_ai_slots_are_detected_by_missing_platform_icon(parsed):
    """Not by the row label: OCR renders 'AI Player' as 'Al Player'."""
    ai = [p for p in parsed.players if p.is_ai]
    assert len(ai) == 1
    assert ai[0].pts == 6


def test_gamertags_resolve_against_the_roster(parsed):
    names = [p.name for p in parsed.players if not p.is_ai]
    assert set(names) == set(ROSTER)


@pytest.mark.parametrize("index", range(len(EXPECTED)))
def test_every_stat_line_matches(parsed, index):
    expected = EXPECTED[index]
    row = parsed.players[index]
    name, team, grade, pts, reb, ast, stl, blk, fouls, tov, fg, tp, ft = expected

    assert row.team == team
    assert row.grade == grade, f"{name} grade"
    for field, want in (("pts", pts), ("reb", reb), ("ast", ast),
                        ("stl", stl), ("blk", blk), ("fouls", fouls),
                        ("tov", tov)):
        assert getattr(row, field) == want, f"{name} {field}"
    for (made, att), want in (((row.fgm, row.fga), fg),
                              ((row.tpm, row.tpa), tp),
                              ((row.ftm, row.fta), ft)):
        assert (made, att) == want, f"{name} fraction"


def test_totals_agree_with_the_rows(parsed):
    """The TOTAL row is a free integrity check on the whole parse."""
    assert parsed.checksum_failures == []
    assert parsed.totals["them"].pts == 45
    assert parsed.totals["us"].pts == 83


def test_a_clean_parse_is_marked_trustworthy(parsed):
    assert parsed.unread_cells == []
    assert parsed.trustworthy


def test_fouls_and_turnovers_are_excluded_from_the_checksum():
    """2K reports team fouls and team turnovers there, not a sum of rows."""
    assert "fouls" not in CHECKSUM_COLUMNS
    assert "tov" not in CHECKSUM_COLUMNS
    assert "pts" in CHECKSUM_COLUMNS


@pytest.mark.parametrize("text,expected", [
    ("8/23", (8, 23)),
    ("0/0", (0, 0)),
    ("12/17", (12, 17)),
    # Makes cannot exceed attempts, so this read must be rejected.
    ("9/2", (None, None)),
    ("", (None, None)),
    (None, (None, None)),
    ("823", (None, None)),
])
def test_fractions_reject_impossible_readings(text, expected):
    assert _split_fraction(text) == expected


def test_resolve_names_is_a_no_op_without_a_roster(parsed):
    assert resolve_names(parsed, []) == {}
