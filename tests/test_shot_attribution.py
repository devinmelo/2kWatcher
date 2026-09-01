"""Who a logged shot belongs to.

Shots are logged without a player, and that is deliberate rather than
unfinished. 2K draws the feedback banner for every shot in the game, not only
the one you took, and nothing on or near the banner names the shooter — it
carries TIMING, COVERAGE and DISTANCE and nothing else. The only thing that
identifies a shooter is who had the ball when it went up, which needs the
tracker that does not exist yet.

So the identity that IS readable gets recorded — the box score's green triangle
says which roster row is yours, and that is stored on players.is_me — but it is
not used to claim shots. Attributing every banner to the owner of the capture
would not be recording data, it would be inventing it.
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
    """The parser reads it; the payload must not throw it away."""
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
        roster = {r["gamertag"]: r["is_me"] for r in db.roster()}
        assert roster["mate1"] == 0
        assert roster["opp0"] == 0


def test_shots_are_logged_without_a_shooter(watcher):
    """The banner shows for everyone's shots, and never says whose it was."""
    watcher._log_box_score(a_box_score())
    with Database(watcher.db_path) as db:
        watcher._log_shot(db, a_shot())
        shots = db.shots()
        assert len(shots) == 1
        assert shots[0]["player_id"] is None


def test_knowing_who_you_are_does_not_claim_the_shots(watcher):
    """Even once identity is known, a banner is not evidence you took it."""
    with Database(watcher.db_path) as db:
        watcher._log_shot(db, a_shot())
        watcher._log_shot(db, a_shot(timing="LATE"))

    watcher._log_box_score(a_box_score())

    with Database(watcher.db_path) as db:
        assert watcher.me_player_id is not None
        assert [s["player_id"] for s in db.shots()] == [None, None]


def test_shots_can_be_queried_by_player_once_one_is_known(watcher):
    """The query path is ready for when a shooter can actually be identified."""
    watcher._log_box_score(a_box_score())
    with Database(watcher.db_path) as db:
        watcher._log_shot(db, a_shot())
        me = db.me()
        assert db.shots(player_id=me["id"]) == []
        assert len(db.shots()) == 1


def test_a_known_player_is_remembered_across_sessions(tmp_path):
    """Restarting the app should not lose who you are."""
    db_path = tmp_path / "w.db"
    first = WatcherThread(Config.load(), LiveState(), db_path, collect=False)
    with Database(db_path) as db:
        first.session_id = db.start_session(source="test")
    first._log_box_score(a_box_score())

    with Database(db_path) as db:
        known = db.me()
        assert known is not None and known["gamertag"] == "Lil Knotty"


def test_ocr_spellings_of_one_gamertag_become_one_player(watcher):
    """A live session turned one opponent into three roster entries.

    The box score is read several times a game and OCR spells a name slightly
    differently each time; without resolution each spelling became its own
    player and its own stat line.
    """
    from twokwatcher.app.watcher import _closest_known

    first = a_box_score()
    first.data["players"][5]["name"] = "Juju Watkin5"
    watcher._log_box_score(first)

    second = a_box_score()
    second.data["players"][5]["name"] = "Juju Watkin5S"
    watcher._log_box_score(second)

    third = a_box_score()
    third.data["players"][5]["name"] = "Juju WatkinS"
    watcher._log_box_score(third)

    with Database(watcher.db_path) as db:
        tags = [r["gamertag"] for r in db.roster()]
        assert sorted(t for t in tags if "Juju" in t) == ["Juju Watkin5"]
        assert len(db.stat_lines(watcher.game_id)) == 10


def test_a_genuinely_different_name_is_not_merged():
    from twokwatcher.app.watcher import _closest_known
    assert _closest_known("TotallyDifferent", ["Juju Watkin5"]) == "TotallyDifferent"
    assert _closest_known("njsimas456", ["Juju Watkin5", "Lil Knotty"]) == "njsimas456"


def test_unreadable_name_cells_do_not_become_players(watcher):
    """A live session put "a", "q", "nn" and "i a" on the roster."""
    from twokwatcher.app.watcher import _is_plausible_gamertag

    event = a_box_score()
    for i, junk in enumerate(["a", "q", "nn", "i a"], start=6):
        event.data["players"][i]["name"] = junk
    watcher._log_box_score(event)

    with Database(watcher.db_path) as db:
        tags = [r["gamertag"] for r in db.roster()]
        assert not ({"a", "q", "nn", "i a"} & set(tags))
        assert "Lil Knotty" in tags
