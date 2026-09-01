"""The box score reaching the pipeline and the database.

Covers the wiring, not the parsing: that entering POST_GAME triggers a read,
that it happens off the frame loop, that re-entering while one is still running
does not pile up a second, and that what lands in the database is a game, a
roster and ten stat lines.
"""

import threading
import time

import numpy as np
import pytest

from twokwatcher.config import Config
from twokwatcher.hud.boxscore import BoxScore, PlayerRow
from twokwatcher.pipeline import EventBus
from twokwatcher.pipeline.runner import Runner
from twokwatcher.storage import Database


class FakeSource:
    fps = 60.0

    def __iter__(self):
        return iter(())

    def read(self):
        return None

    def close(self):
        pass


class FakeFrame:
    def __init__(self):
        self.image = np.zeros((16, 16, 3), np.uint8)
        self.index = 7
        self.timestamp = 1.25


def sample_box():
    box = BoxScore()
    for i in range(5):
        box.players.append(PlayerRow(name=f"us{i}", team="us", grade="A",
                                     pts=10 + i, reb=1, ast=2, stl=0, blk=0,
                                     tov=1, fgm=4, fga=8, tpm=1, tpa=2,
                                     ftm=0, fta=0))
    for i in range(5):
        box.players.append(PlayerRow(name=f"them{i}", team="them", grade="B",
                                     pts=5 + i, reb=2, ast=1, stl=1, blk=0,
                                     tov=0, fgm=2, fga=6, tpm=0, tpa=1,
                                     ftm=0, fta=0))
    box.totals["us"] = PlayerRow(name="TOTAL", team="us", pts=60)
    box.totals["them"] = PlayerRow(name="TOTAL", team="them", pts=35)
    return box


class StubParser:
    def __init__(self, box=None, delay=0.0, blow_up=False):
        self.box = box if box is not None else sample_box()
        self.delay = delay
        self.blow_up = blow_up
        self.calls = 0
        self.started = threading.Event()

    def available(self):
        return True

    def parse(self, image):
        self.calls += 1
        self.started.set()
        if self.delay:
            time.sleep(self.delay)
        if self.blow_up:
            raise RuntimeError("tesseract fell over")
        return self.box


def runner_with(parser):
    bus = EventBus()
    seen = []
    bus.subscribe("box_score", seen.append)
    runner = Runner(FakeSource(), Config.load(), bus=bus, sample_fps=60)
    runner.box_score = parser
    runner._box_reader_ready = True
    return runner, seen


def finish(runner, timeout=10.0):
    """Wait for the parse thread AND its subscribers to be done.

    Subscribers run in registration order inside publish(), so seeing the event
    land in a list says nothing about whether the one that writes to the
    database has run yet. Joining the worker is what actually settles it.
    """
    if runner._box_thread is not None:
        runner._box_thread.join(timeout)


def wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_entering_post_game_publishes_a_box_score():
    parser = StubParser()
    runner, seen = runner_with(parser)
    runner._read_box_score(FakeFrame())
    assert wait_for(lambda: len(seen) == 1), "no box_score event published"
    data = seen[0].data
    assert len(data["players"]) == 10
    assert data["totals"]["us"]["pts"] == 60
    assert runner.stats.box_scores == 1


def test_the_frame_loop_is_not_blocked_by_the_parse():
    """A real parse takes about 90 seconds; inline it would stall capture."""
    parser = StubParser(delay=1.5)
    runner, _ = runner_with(parser)
    started = time.perf_counter()
    runner._read_box_score(FakeFrame())
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"_read_box_score blocked for {elapsed:.2f}s"
    assert parser.started.wait(5.0)


def test_a_second_entry_does_not_start_a_second_parse():
    """Re-opening the screen must not queue work behind work."""
    parser = StubParser(delay=1.0)
    runner, _ = runner_with(parser)
    runner._read_box_score(FakeFrame())
    assert parser.started.wait(5.0)
    runner._read_box_score(FakeFrame())
    assert parser.calls == 1


def test_a_parse_that_raises_publishes_nothing():
    parser = StubParser(blow_up=True)
    runner, seen = runner_with(parser)
    runner._read_box_score(FakeFrame())
    assert wait_for(lambda: not runner._box_thread.is_alive())
    assert seen == []
    assert runner.stats.box_scores == 0


def test_an_unreadable_screen_is_not_written_as_empty_rows():
    """Ten empty rows are worse than no rows."""
    empty = BoxScore()
    empty.players = [PlayerRow(team="us") for _ in range(10)]
    parser = StubParser(box=empty)
    runner, seen = runner_with(parser)
    runner._read_box_score(FakeFrame())
    assert wait_for(lambda: not runner._box_thread.is_alive())
    assert seen == []


def test_nothing_runs_without_tesseract():
    class Unavailable(StubParser):
        def available(self):
            return False

    parser = Unavailable()
    runner, seen = runner_with(parser)
    runner._box_reader_ready = None
    runner._read_box_score(FakeFrame())
    assert parser.calls == 0
    assert seen == []


# --- persistence --------------------------------------------------------

def test_a_box_score_lands_in_the_database(tmp_path):
    from twokwatcher.app.live import LiveState
    from twokwatcher.app.watcher import WatcherThread

    db_path = tmp_path / "w.db"
    watcher = WatcherThread(Config.load(), LiveState(), db_path, collect=False)
    with Database(db_path) as db:
        watcher.session_id = db.start_session(source="test")

    parser = StubParser()
    runner, seen = runner_with(parser)
    runner.bus.subscribe("box_score", watcher._log_box_score)
    runner._read_box_score(FakeFrame())
    finish(runner)
    assert len(seen) == 1
    assert watcher.game_id is not None

    with Database(db_path) as db:
        lines = db.stat_lines(watcher.game_id)
        assert len(lines) == 10
        assert {r["gamertag"] for r in lines} >= {"us0", "them4"}
        by_tag = {r["gamertag"]: r for r in lines}
        assert by_tag["us0"]["pts"] == 10
        assert by_tag["us0"]["team"] == "us"
        assert by_tag["them4"]["pts"] == 9

        game = db.recent_games(1)[0]
        assert (game["score_us"], game["score_them"]) == (60, 35)
        assert game["result"] == "W"

        events = db.conn.execute(
            "SELECT kind FROM events WHERE kind = 'box_score'").fetchall()
        assert len(events) == 1


def test_reading_the_screen_twice_updates_rather_than_duplicates(tmp_path):
    """2K lets you open the box score mid-game, so the same game is read again."""
    from twokwatcher.app.live import LiveState
    from twokwatcher.app.watcher import WatcherThread

    db_path = tmp_path / "w.db"
    watcher = WatcherThread(Config.load(), LiveState(), db_path, collect=False)
    with Database(db_path) as db:
        watcher.session_id = db.start_session(source="test")

    first = sample_box()
    runner, seen = runner_with(StubParser(box=first))
    runner.bus.subscribe("box_score", watcher._log_box_score)
    runner._read_box_score(FakeFrame())
    finish(runner)
    assert len(seen) == 1

    later = sample_box()
    later.players[0].pts = 31
    runner.box_score = StubParser(box=later)
    runner._box_thread = None
    runner._read_box_score(FakeFrame())
    finish(runner)
    assert len(seen) == 2

    with Database(db_path) as db:
        lines = db.stat_lines(watcher.game_id)
        assert len(lines) == 10, "a second read duplicated the rows"
        assert {r["gamertag"]: r["pts"] for r in lines}["us0"] == 31


def test_a_declined_cell_does_not_erase_a_value_read_earlier(tmp_path):
    from twokwatcher.app.live import LiveState
    from twokwatcher.app.watcher import WatcherThread

    db_path = tmp_path / "w.db"
    watcher = WatcherThread(Config.load(), LiveState(), db_path, collect=False)
    with Database(db_path) as db:
        watcher.session_id = db.start_session(source="test")

    runner, seen = runner_with(StubParser(box=sample_box()))
    runner.bus.subscribe("box_score", watcher._log_box_score)
    runner._read_box_score(FakeFrame())
    finish(runner)
    assert len(seen) == 1

    partial = sample_box()
    partial.players[0].reb = None
    runner.box_score = StubParser(box=partial)
    runner._box_thread = None
    runner._read_box_score(FakeFrame())
    finish(runner)
    assert len(seen) == 2

    with Database(db_path) as db:
        by_tag = {r["gamertag"]: r for r in db.stat_lines(watcher.game_id)}
        assert by_tag["us0"]["reb"] == 1, "an abstention wiped a known value"
