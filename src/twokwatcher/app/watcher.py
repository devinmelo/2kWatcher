"""Runs the frame loop on a background thread, under the UI's control.

The app needs to start and stop watching without exiting, so the loop cannot
own the process the way it does under `2kw run`. This wraps it in a thread with
a stop flag, and funnels failures into LiveState rather than letting them kill
the thread silently — a watcher that has quietly died must look different in
the UI from one that is idle.
"""

from __future__ import annotations

import logging
import threading

from ..capture import open_source
from ..config import Config
from ..pipeline import EventBus, Runner
from ..storage import Database
from .live import LiveState

log = logging.getLogger(__name__)


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
                 device_index: int | None = None, video_path=None) -> None:
        self.config = config
        self.live = live
        self.db_path = db_path
        self.device_index = device_index
        self.video_path = video_path
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

    def _run(self) -> None:
        source_name = "file" if self.video_path else "virtualcam"
        bus = EventBus()
        self.live.subscribe(bus)

        try:
            # SQLite connections are per-thread, so the watcher opens its own.
            with Database(self.db_path) as db:
                self.session_id = db.start_session(source=source_name)
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
                        StoppableRunner(source, self.config, bus=bus,
                                        stop=self._stop, live=self.live).run()
                    self.live.note("watcher", "Source ended")
                finally:
                    db.end_session(self.session_id)
        except _Stopped:
            self.live.note("watcher", "Stopped")
        except Exception as exc:                     # noqa: BLE001
            # Surfaced in the UI. A dead watcher must not look like an idle one.
            log.exception("Watcher failed")
            self.live.set_running(False, error=str(exc))
            self.live.note("error", str(exc))
            return
        self.live.set_running(False)
