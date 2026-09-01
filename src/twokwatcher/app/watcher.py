"""Runs the frame loop on a background thread, under the UI's control.

The app needs to start and stop watching without exiting, so the loop cannot
own the process the way it does under `2kw run`. This wraps it in a thread with
a stop flag, and funnels failures into LiveState rather than letting them kill
the thread silently — a watcher that has quietly died must look different in
the UI from one that is idle.
"""

from __future__ import annotations

import difflib
import logging
import threading
from pathlib import Path

from ..capture import open_source
from ..config import Config
from ..pipeline import EventBus, Runner
from ..storage import Database
from .collect import FrameCollector
from .live import LiveState

log = logging.getLogger(__name__)

# How close a freshly read gamertag must be to one already on record
# to be treated as the same player. Gamertags are long, so a strict
# bar still absorbs the glyph confusions OCR makes on them.
NAME_MATCH_RATIO = 0.82


def _closest_known(name, known):
    """The registry spelling of a gamertag, if it is already on record."""
    if not name or not known or name in known:
        return name
    matches = difflib.get_close_matches(name, known, n=1,
                                        cutoff=NAME_MATCH_RATIO)
    return matches[0] if matches else name


class StoppableRunner(Runner):
    """A Runner that checks a flag between frames."""

    def __init__(self, *args, stop: threading.Event, live: LiveState, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop = stop
        self._live = live

    def _process(self, frame) -> None:
        super()._process(frame)
        self._live.set_progress(self.stats.frames_seen, self.stats.frames_sampled)
        if self._stop.is_set():
            raise _Stopped()


class _Stopped(Exception):
    """Internal: unwinds the frame loop when the UI asks it to stop."""


class WatcherThread:
    """Owns the watcher's lifecycle for the app."""

    def __init__(self, config: Config, live: LiveState, db_path, *,
                 device_index: int | None = None, video_path=None,
                 collect_dir="data/collect", collect: bool = True,
                 clip_dir="data/clips", clips: bool = True) -> None:
        self.config = config
        self.live = live
        self.db_path = db_path
        self.device_index = device_index
        self.video_path = video_path
        # On by default: the first sessions are worth far more as a labelled
        # frame set than as a database, because every parser still unwritten
        # is blocked on nobody having seen this HUD.
        self.collector = FrameCollector(collect_dir, enabled=collect)
        # A second of video either side of every shot. Namespaced per session
        # once the session opens, so one run can never overwrite another's.
        self.clip_root = Path(clip_dir) if clips else None
        # Set once games are opened and closed off the post-game screen; until
        # then shots are logged against the session with a null game.
        self.game_id: int | None = None
        # Which roster row is yours, from the box score's YOU marker. Not
        # currently used to attribute shots — see _log_shot.
        self.me_player_id: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.session_id: int | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.is_running:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="watcher")
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.live.set_running(False)

    def _player_id_for(self, db, gamertag):
        """The roster id for a named shooter, if the plate resolved to one."""
        if not gamertag:
            return None
        row = db.conn.execute("SELECT id FROM players WHERE gamertag = ?",
                              (gamertag,)).fetchone()
        return row["id"] if row else None

    def _log_shot(self, db, event) -> None:
        """Persist one shot, and surface it in the UI as it happens."""
        data = event.data
        try:
            db.log_event(
                # The shooter comes from the gamertag plate that was up in
                # the frames before the banner, not from the banner itself,
                # which never names anyone.
                game_id=self.game_id, kind="shot_feedback",
                player_id=self._player_id_for(db, data.get("shooter")),
                frame_index=event.frame_index, video_ts=event.video_ts,
                payload=data, clip_path=data.get("clip"),
            )
        except Exception:                                # noqa: BLE001
            # A failed write must not take the watcher down mid-game.
            log.exception("Could not log a shot")
            return
        shooter = data.get("shooter")
        parts = ([shooter] if shooter else []) + [
            p for p in (data.get("timing"), data.get("coverage")) if p]
        distance = data.get("distance_feet")
        if distance is not None:
            parts.append(f"{distance:.0f}ft")
        self.live.note("shot", " / ".join(parts) or "unread")

    def _log_box_score(self, event) -> None:
        """Persist a parsed box score: the game, its roster, and every line.

        Opens its own connection rather than borrowing the watcher's. The box
        score is parsed on a worker thread — it takes about ninety seconds and
        cannot hold up the frame loop — so this arrives off-thread, and SQLite
        connections belong to the thread that made them. Doing it here is cheap
        because a box score lands once per game screen, not once per frame.

        The whole parse also goes onto the event as JSON. The normalized table
        has no column for every stat 2K shows — fouls in particular — and this
        screen is too expensive to read for anything it yielded to be discarded.
        """
        data = event.data
        players = [p for p in data.get("players", []) if p.get("name")]
        if not players:
            return
        try:
            with Database(self.db_path) as db:
                if self.game_id is None:
                    self.game_id = db.start_game(
                        self.session_id, video_path=str(self.video_path or ""))
                db.log_event(game_id=self.game_id, kind="box_score",
                             frame_index=event.frame_index,
                             video_ts=event.video_ts, payload=data)

                # Snap gamertags onto ones already on record before they are
                # written. The box score is read several times a game and OCR
                # spells a name slightly differently each time — one session
                # turned a single opponent into "Juju Watkin5", "Juju Watkin5S"
                # and "Juju WatkinS", three roster entries and three stat lines
                # for one player. The registry is what settles it, which is the
                # job resolve_names was written for.
                known = [row["gamertag"] for row in db.roster()]
                for line in players:
                    line["name"] = _closest_known(line["name"], known)

                for line in players:
                    # The green triangle on the screen is what says which row
                    # is yours, and it is the only thing that does. Recording
                    # it here is what lets shots be attributed to a player.
                    player_id = db.upsert_player(
                        line["name"], is_me=bool(line.get("is_you")))
                    known.append(line["name"])
                    if line.get("is_you"):
                        self.me_player_id = player_id
                    db.record_stat_line(
                        game_id=self.game_id, player_id=player_id,
                        team=line.get("team", "them"),
                        **{k: line.get(k) for k in
                           ("grade", "pts", "reb", "ast", "stl", "blk", "tov",
                            "fgm", "fga", "tpm", "tpa", "ftm", "fta")},
                    )

                totals = data.get("totals") or {}
                ours = (totals.get("us") or {}).get("pts")
                theirs = (totals.get("them") or {}).get("pts")
                if ours is not None and theirs is not None:
                    db.end_game(self.game_id, score_us=ours, score_them=theirs)
        except Exception:                                # noqa: BLE001
            log.exception("Could not log a box score")
            return

        note = f"{len(players)} stat lines"
        if not data.get("trustworthy", True):
            flags = len(data.get("checksum_failures") or []) + \
                len(data.get("unread_cells") or [])
            note += f" ({flags} to review)"
        self.live.note("box score", note)

    def _run(self) -> None:
        source_name = "file" if self.video_path else "virtualcam"
        bus = EventBus()
        self.live.subscribe(bus)
        self.collector.subscribe(bus)
        bus.subscribe("preview",
                      lambda _e: self.live.set_collected(self.collector.counts))

        try:
            # SQLite connections are per-thread, so the watcher opens its own.
            with Database(self.db_path) as db:
                self.session_id = db.start_session(source=source_name)
                known = db.me()
                if known is not None:
                    self.me_player_id = known["id"]
                # Shots are the point of the whole exercise, so they are
                # persisted even before games are being opened and closed —
                # a shot with a null game_id is still a shot, and can be
                # attributed later from the session's state log.
                bus.subscribe("shot_feedback", lambda e: self._log_shot(db, e))
                bus.subscribe("box_score", self._log_box_score)
                bus.subscribe("state_change", lambda e: db.log_state(
                    self.session_id, e.data["previous"], e.data["current"],
                    e.frame_index, e.video_ts))

                if self.video_path:
                    source = open_source("file", video_path=self.video_path)
                else:
                    source = open_source("virtualcam",
                                         device_index=self.device_index or 0)

                self.live.set_running(True, source=str(
                    self.video_path or f"device {self.device_index or 0}"))
                self.live.note("watcher", "Started watching")

                try:
                    with source:
                        clip_dir = (self.clip_root / str(self.session_id)
                                    if self.clip_root else None)
                        StoppableRunner(source, self.config, bus=bus,
                                        stop=self._stop, live=self.live,
                                        clip_dir=clip_dir).run()
                    self.live.note("watcher", "Source ended")
                finally:
                    db.end_session(self.session_id)
                    manifest = self.collector.flush_manifest()
                    if manifest is not None:
                        self.live.note(
                            "watcher",
                            f"Collected {self.collector.total} frames "
                            f"→ {manifest.parent}")
        except _Stopped:
            self.live.note("watcher", "Stopped")
        except Exception as exc:                     # noqa: BLE001
            # Surfaced in the UI. A dead watcher must not look like an idle one.
            log.exception("Watcher failed")
            self.live.set_running(False, error=str(exc))
            self.live.note("error", str(exc))
            return
        self.live.set_running(False)
