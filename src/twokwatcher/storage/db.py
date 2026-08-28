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
