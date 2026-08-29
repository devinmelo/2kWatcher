"""Attributing shots to the player who took them.

Shot feedback grades your own release — 2K does not show it for a teammate's
shot or an opponent's, because you did not release anything. So every logged
shot is yours, and the work is not detecting a shooter but recording which
gamertag "yours" refers to.

That name comes from one place: the green triangle beside your row on the box
score. Until a box score has been read, shots are written unattributed rather
than guessed at, and claimed afterwards.
"""

import pytest

from twokwatcher.config import Config
from twokwatcher.app.live import LiveState
from twokwatcher.app.watcher import WatcherThread
from twokwatcher.hud.boxscore import BoxScore, PlayerRow
from twokwatcher.storage import Database


class FakeEvent:
    def __init__(self, data, frame_index=1, video_ts=0.5):
        self.data = data
        self.frame_index = frame_index
        self.video_ts = video_ts


def a_shot(timing="EXCELLENT", coverage="OPEN", distance=22.3):
    return FakeEvent({"timing": timing, "coverage": coverage,
                      "distance_feet": distance})


def a_box_score(you="Lil Knotty"):
    box = BoxScore()
    for i in range(5):
        box.players.append(PlayerRow(name=f"mate{i}", team="us", pts=10))
    for i in range(5):
        box.players.append(PlayerRow(name=f"opp{i}", team="them", pts=8))
    box.players[0].name = you
    box.players[0].is_you = True
    box.totals["us"] = PlayerRow(name="TOTAL", team="us", pts=50)
    box.totals["them"] = PlayerRow(name="TOTAL", team="them", pts=40)
    from twokwatcher.pipeline.runner import _stat_line
    return FakeEvent({
        "players": [_stat_line(p) for p in box.players],
        "totals": {t: _stat_line(r) for t, r in box.totals.items()},
        "checksum_failures": [], "unread_cells": [], "trustworthy": True,
    })


@pytest.fixture
def watcher(tmp_path):
    db_path = tmp_path / "w.db"
    w = WatcherThread(Config.load(), LiveState(), db_path, collect=False)
    with Database(db_path) as db:
        w.session_id = db.start_session(source="test")
    return w


def test_the_you_marker_survives_into_the_event():
    """It was being stripped, which left nothing to attribute shots with."""
    data = a_box_score().data
    assert data["players"][0]["is_you"] is True
    assert all(p["is_you"] is False for p in data["players"][1:])


def test_a_box_score_records_who_you_are(watcher):
    watcher._log_box_score(a_box_score())
    with Database(watcher.db_path) as db:
        me = db.me()
        assert me is not None
        assert me["gamertag"] == "Lil Knotty"


def test_only_your_row_is_marked_as_you(watcher):
    watcher._log_box_score(a_box_score())
    with Database(watcher.db_path) as db:
        flagged = [r["gamertag"] for r in db.roster() if r["is_me"]]
        assert flagged == ["Lil Knotty"]


def test_shots_after_a_box_score_carry_the_player(watcher):
    watcher._log_box_score(a_box_score())
    with Database(watcher.db_path) as db:
        watcher._log_shot(db, a_shot())
        shots = db.shots()
        assert len(shots) == 1
        assert shots[0]["gamertag"] == "Lil Knotty"


def test_shots_before_a_box_score_are_claimed_afterwards(watcher):
    """A session usually logs shots long before the first box score."""
    with Database(watcher.db_path) as db:
        watcher._log_shot(db, a_shot())
        watcher._log_shot(db, a_shot(timing="LATE"))
        assert [s["player_id"] for s in db.shots()] == [None, None]

    watcher._log_box_score(a_box_score())

    with Database(watcher.db_path) as db:
        shots = db.shots()
        assert len(shots) == 2
        assert all(s["gamertag"] == "Lil Knotty" for s in shots)


def test_shots_can_be_selected_for_one_player(watcher):
    watcher._log_box_score(a_box_score())
    with Database(watcher.db_path) as db:
        watcher._log_shot(db, a_shot())
        me = db.me()
        assert len(db.shots(player_id=me["id"])) == 1
        assert db.shots(player_id=me["id"] + 999) == []


def test_a_known_player_is_remembered_across_sessions(tmp_path):
    """Restarting the app should not lose who you are."""
    db_path = tmp_path / "w.db"
    first = WatcherThread(Config.load(), LiveState(), db_path, collect=False)
    with Database(db_path) as db:
        first.session_id = db.start_session(source="test")
    first._log_box_score(a_box_score())

    second = WatcherThread(Config.load(), LiveState(), db_path, collect=False)
    with Database(db_path) as db:
        known = db.me()
        assert known is not None
        second.me_player_id = known["id"]
        second.session_id = db.start_session(source="test")
        second._log_shot(db, a_shot())
        assert db.shots()[0]["gamertag"] == "Lil Knotty"


def test_teammates_are_not_marked_as_you(watcher):
    """Only the row behind the green triangle is yours."""
    watcher._log_box_score(a_box_score())
    with Database(watcher.db_path) as db:
        roster = {r["gamertag"]: r["is_me"] for r in db.roster()}
        assert roster["mate1"] == 0
        assert roster["opp0"] == 0
