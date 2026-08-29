"""Regression tests against the Rec game-stats screen.

2K shows a full box score mid-game under GAME STATS, and it is the same table
the post-game screen draws. This fixture is that screen at the end of a Rec
game, captured on the machine that owns the capture card.

It is worth having alongside the MyCareer post-game fixture because the two
differ in ways that have already caused bugs: the Rec screen puts YOUR team in
the top block where MyCareer puts the opposition, and its FOULS and TO columns
do sum to their TOTAL where MyCareer's do not.

The truth below is hand-read from the fixture. The final score it adds up to,
81-53, is corroborated by the game's own "YOU WON 81 - 53" screen.
"""

import cv2
import pytest

from twokwatcher.hud import BoxScoreParser, resolve_names

FIXTURE = "tests/fixtures/gamestats_rec_1080p.png"

# name, team, grade, pts, reb, ast, stl, blk, fouls, tov, fg, tp, ft
EXPECTED = [
    ("Lil Knotty",      "us",   "A",  15,  0, 13, 0, 0, 0, 2, (7, 9),   (1, 2),  (0, 0)),
    ("njsimas456",      "us",   "A-", 26,  0,  1, 0, 0, 0, 1, (9, 15),  (8, 14), (0, 0)),
    ("colby2818",       "us",   "B+", 16,  1,  0, 0, 1, 0, 2, (7, 13),  (2, 5),  (0, 0)),
    ("Hurricane091798", "us",   "A-", 12, 10,  5, 0, 1, 0, 3, (5, 10),  (2, 2),  (0, 0)),
    ("AnRaDe728",       "us",   "A",  12, 28, 10, 1, 2, 0, 1, (6, 12),  (0, 1),  (0, 0)),
    ("MxreDiff",        "them", "D+",  2,  2,  8, 0, 0, 0, 0, (1, 9),   (0, 8),  (0, 0)),
    ("who Wetz",        "them", "C+",  8,  2,  1, 3, 0, 0, 0, (3, 10),  (2, 9),  (0, 0)),
    ("whyygdk",         "them", "C+", 43,  0,  1, 2, 0, 1, 2, (20, 36), (3, 5),  (0, 0)),
    ("Raqbully",        "them", "B+",  0,  1,  4, 4, 0, 0, 0, (0, 4),   (0, 4),  (0, 0)),
    ("stophesii",       "them", "A",   0, 13,  2, 0, 3, 0, 0, (0, 2),   (0, 1),  (0, 0)),
]
ROSTER = [name for name, *_ in EXPECTED]


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


def test_your_team_is_the_top_block_on_this_screen(parsed):
    """The opposite of MyCareer, which is why team cannot come from row order."""
    assert parsed.you is not None
    assert parsed.you.name == "Lil Knotty"
    assert parsed.you.team == "us"
    assert [p.name for p in parsed.players if p.team == "us"][:2] == [
        "Lil Knotty", "njsimas456"]


def test_the_totals_agree_with_the_rows(parsed):
    assert parsed.checksum_failures == []


def test_the_score_matches_the_result_screen(parsed):
    """81-53, corroborated by the game's own YOU WON screen."""
    ours = sum(p.pts for p in parsed.players if p.team == "us" and p.pts is not None)
    theirs = sum(p.pts for p in parsed.players if p.team == "them" and p.pts is not None)
    assert (ours, theirs) == (81, 53)


ATTRS = ("grade", "pts", "reb", "ast", "stl", "blk", "fouls", "tov",
         "fgm", "fga", "tpm", "tpa", "ftm", "fta")


def _wants(expected):
    return (*expected[2:10], expected[10][0], expected[10][1],
            expected[11][0], expected[11][1], expected[12][0], expected[12][1])


@pytest.mark.parametrize("index", range(len(EXPECTED)))
def test_every_row_is_identified(parsed, index):
    name, team = EXPECTED[index][0], EXPECTED[index][1]
    row = parsed.players[index]
    assert row.name == name
    assert row.team == team


# One cell on this screen is still read wrong: Hurricane091798's field goals
# are 5/10 on screen and come back as 3/10. It is pinned rather than tolerated
# — a second wrong cell, anywhere, fails this test. The checksum does not catch
# it, because colby2818's field-goal cell abstains and a column with a missing
# value cannot be summed against its TOTAL.
KNOWN_WRONG = {"Hurricane091798/fgm"}

# Cells the parser declines. It may read more; reading fewer should fail.
MAX_ABSTENTIONS = 3


def test_no_new_cell_is_read_wrong(parsed):
    """Cells may go unread; none may go newly wrong. That is the contract."""
    wrong = set()
    for row, expected in zip(parsed.players, EXPECTED):
        for attr, want in zip(ATTRS, _wants(expected)):
            got = getattr(row, attr)
            if got is not None and got != want:
                wrong.add(f"{row.name}/{attr}")
    assert wrong == KNOWN_WRONG


def test_abstentions_stay_rare(parsed):
    missing = [f"{row.name}/{attr}"
               for row, expected in zip(parsed.players, EXPECTED)
               for attr in ATTRS if getattr(row, attr) is None]
    assert len(missing) <= MAX_ABSTENTIONS, missing
