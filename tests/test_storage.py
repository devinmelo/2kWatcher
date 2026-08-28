from twokwatcher.storage import Database


def test_roster_flags_are_sticky(tmp_path):
    with Database(tmp_path / "t.db") as db:
        me = db.upsert_player("devinmelo", is_me=True)
        # Seeing the same tag again incidentally must not demote them.
        again = db.upsert_player("devinmelo")
        assert me == again
        assert db.roster()[0]["is_me"] == 1


def test_game_result_is_derived_from_score(tmp_path):
    with Database(tmp_path / "t.db") as db:
        session = db.start_session("file")
        game = db.start_game(session, mode="rec")
        db.end_game(game, score_us=88, score_them=74)
        row = db.conn.execute("SELECT * FROM games WHERE id = ?", (game,)).fetchone()
        assert row["result"] == "W"


def test_events_store_json_payloads(tmp_path):
    with Database(tmp_path / "t.db") as db:
        session = db.start_session("file")
        game = db.start_game(session)
        db.log_event(game_id=game, kind="shot_feedback",
                     payload={"verdict": "slightly late", "made": False})
        row = db.conn.execute("SELECT payload FROM events").fetchone()
        assert "slightly late" in row["payload"]
