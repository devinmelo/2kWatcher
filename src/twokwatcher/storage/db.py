"""SQLite access.

Deliberately thin — plain SQL against a well-shaped schema, no ORM. The value
of this project lives in the data, and the data should stay trivially
inspectable with the sqlite3 CLI or a notebook.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB_PATH = Path("data/2kwatcher.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_PATH.read_text())
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- sessions -------------------------------------------------------

    def start_session(self, source: str, notes: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (started_at, source, notes) VALUES (?, ?, ?)",
            (_now(), source, notes),
        )
        self.conn.commit()
        return cur.lastrowid

    def end_session(self, session_id: int) -> None:
        self.conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?", (_now(), session_id)
        )
        self.conn.commit()

    # --- players --------------------------------------------------------

    def upsert_player(
        self, gamertag: str, *, display_name: str | None = None,
        is_me: bool = False, is_friend: bool = False,
    ) -> int:
        """Insert or refresh a roster entry, returning its id.

        Flags are OR-ed rather than overwritten, so an incidental sighting of a
        known friend never demotes them back to a stranger.
        """
        self.conn.execute(
            """
            INSERT INTO players (gamertag, display_name, is_me, is_friend,
                                 first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(gamertag) DO UPDATE SET
                display_name = COALESCE(excluded.display_name, players.display_name),
                is_me        = MAX(players.is_me, excluded.is_me),
                is_friend    = MAX(players.is_friend, excluded.is_friend),
                last_seen    = excluded.last_seen
            """,
            (gamertag, display_name, int(is_me), int(is_friend), _now(), _now()),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM players WHERE gamertag = ?", (gamertag,)
        ).fetchone()
        return row["id"]

    def me(self) -> sqlite3.Row | None:
        """The player this capture belongs to, if one has been identified."""
        return self.conn.execute(
            "SELECT * FROM players WHERE is_me = 1 ORDER BY last_seen DESC"
        ).fetchone()

    def attribute_unassigned_shots(self, player_id: int) -> int:
        """Attach a player to shots that were logged before we knew who you were.

        Shot feedback grades your own release, so every logged shot is yours —
        but the name is only learned from a box score, which may not arrive
        until well into the session. Rather than drop those shots or guess,
        they are written unattributed and claimed once the identity is known.
        """
        cur = self.conn.execute(
            "UPDATE events SET player_id = ?"
            " WHERE kind = 'shot_feedback' AND player_id IS NULL",
            (player_id,),
        )
        self.conn.commit()
        return cur.rowcount

    def shots(self, player_id: int | None = None,
              limit: int = 200) -> list[sqlite3.Row]:
        """Logged shots, newest first, optionally for one player."""
        if player_id is None:
            return self.conn.execute(
                "SELECT e.*, p.gamertag FROM events e"
                " LEFT JOIN players p ON p.id = e.player_id"
                " WHERE e.kind = 'shot_feedback'"
                " ORDER BY e.id DESC LIMIT ?", (limit,),
            ).fetchall()
        return self.conn.execute(
            "SELECT e.*, p.gamertag FROM events e"
            " LEFT JOIN players p ON p.id = e.player_id"
            " WHERE e.kind = 'shot_feedback' AND e.player_id = ?"
            " ORDER BY e.id DESC LIMIT ?", (player_id, limit),
        ).fetchall()

    def roster(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM players ORDER BY is_me DESC, is_friend DESC, gamertag"
        ).fetchall()

    # --- games ----------------------------------------------------------

    def start_game(
        self, session_id: int, *, mode: str | None = None,
        video_path: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO games (session_id, mode, started_at, video_path)"
            " VALUES (?, ?, ?, ?)",
            (session_id, mode, _now(), video_path),
        )
        self.conn.commit()
        return cur.lastrowid

    def end_game(
        self, game_id: int, *, score_us: int | None = None,
        score_them: int | None = None,
    ) -> None:
        result = None
        if score_us is not None and score_them is not None:
            result = "W" if score_us > score_them else "L"
        self.conn.execute(
            "UPDATE games SET ended_at = ?, score_us = ?, score_them = ?,"
            " result = ? WHERE id = ?",
            (_now(), score_us, score_them, result, game_id),
        )
        self.conn.commit()

    def record_stat_line(
        self, *, game_id: int, player_id: int, team: str, grade: str | None = None,
        pts: int | None = None, reb: int | None = None, ast: int | None = None,
        stl: int | None = None, blk: int | None = None, tov: int | None = None,
        fgm: int | None = None, fga: int | None = None,
        tpm: int | None = None, tpa: int | None = None,
        ftm: int | None = None, fta: int | None = None,
    ) -> None:
        """Write one player's line for a game, replacing any earlier one.

        The box score can be read more than once — 2K lets you open it mid-game
        — so this upserts rather than inserts. A later read supersedes an
        earlier one, but only field by field: a cell the parser declined this
        time must not wipe a value it managed to read before.

        FOULS has no column here. The full parse is kept as JSON on the
        matching event, so nothing is lost, but the normalized table holds only
        what the schema was built for.
        """
        self.conn.execute(
            """
            INSERT INTO game_players (game_id, player_id, team, grade, pts, reb,
                                      ast, stl, blk, tov, fgm, fga, tpm, tpa,
                                      ftm, fta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_id, player_id) DO UPDATE SET
                team  = excluded.team,
                grade = COALESCE(excluded.grade, game_players.grade),
                pts   = COALESCE(excluded.pts,   game_players.pts),
                reb   = COALESCE(excluded.reb,   game_players.reb),
                ast   = COALESCE(excluded.ast,   game_players.ast),
                stl   = COALESCE(excluded.stl,   game_players.stl),
                blk   = COALESCE(excluded.blk,   game_players.blk),
                tov   = COALESCE(excluded.tov,   game_players.tov),
                fgm   = COALESCE(excluded.fgm,   game_players.fgm),
                fga   = COALESCE(excluded.fga,   game_players.fga),
                tpm   = COALESCE(excluded.tpm,   game_players.tpm),
                tpa   = COALESCE(excluded.tpa,   game_players.tpa),
                ftm   = COALESCE(excluded.ftm,   game_players.ftm),
                fta   = COALESCE(excluded.fta,   game_players.fta)
            """,
            (game_id, player_id, team, grade, pts, reb, ast, stl, blk, tov,
             fgm, fga, tpm, tpa, ftm, fta),
        )
        self.conn.commit()

    def stat_lines(self, game_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT gp.*, p.gamertag FROM game_players gp"
            " JOIN players p ON p.id = gp.player_id"
            " WHERE gp.game_id = ? ORDER BY gp.team DESC, gp.pts DESC",
            (game_id,),
        ).fetchall()

    # --- events and logging ---------------------------------------------

    def log_event(
        self, *, game_id: int | None, kind: str, frame_index: int | None = None,
        video_ts: float | None = None, player_id: int | None = None,
        quarter: int | None = None, game_clock: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO events (game_id, player_id, frame_index, video_ts,"
            " quarter, game_clock, kind, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (game_id, player_id, frame_index, video_ts, quarter, game_clock,
             kind, json.dumps(payload) if payload else None),
        )
        self.conn.commit()
        return cur.lastrowid

    def record_correction(
        self, *, kind: str, corrected: str, field: str | None = None,
        observed: str | None = None, game_id: int | None = None,
        frame_index: int | None = None, crop_png: bytes | None = None,
    ) -> int:
        """Store a human correction, with the pixels that caused it.

        The crop is the point: a correction without it is a one-off fix, a
        correction with it is a labelled example the parsers can be improved
        against later.
        """
        cur = self.conn.execute(
            "INSERT INTO corrections (created_at, kind, field, observed,"
            " corrected, game_id, frame_index, crop_png)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), kind, field, observed, corrected, game_id, frame_index,
             crop_png),
        )
        self.conn.commit()
        return cur.lastrowid

    def corrections(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, created_at, kind, field, observed, corrected"
            " FROM corrections ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def recent_games(self, limit: int = 25) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM games ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def log_state(
        self, session_id: int, previous: str, current: str,
        frame_index: int, video_ts: float,
    ) -> None:
        self.conn.execute(
            "INSERT INTO state_log (session_id, frame_index, video_ts, previous,"
            " current) VALUES (?, ?, ?, ?, ?)",
            (session_id, frame_index, video_ts, previous, current),
        )
        self.conn.commit()
