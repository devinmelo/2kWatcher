"""The main frame loop.

Structure worth preserving as this grows: the loop samples frames, classifies
screen state, and publishes events. It does not itself know about the database,
the dashboard, or the tracker — those attach as subscribers. Adding the tracker
later should mean registering another stage here, not restructuring this file.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field

import numpy as np

from ..capture import FrameSource
from ..config import Config
from ..hud import BoxScoreParser, ScoreboardReader, ShotFeedbackReader
from ..state import GameState, StateMachine
from ..state.detectors import ScreenClassifier
from .events import Event, EventBus

log = logging.getLogger(__name__)

# A banner is up for about a second. This is a generous ceiling on how many
# frames of one are worth keeping, so a gate that sticks on cannot grow the
# buffer without bound.
MAX_BANNER_FRAMES = 40


@dataclass
class RunnerStats:
    frames_seen: int = 0
    frames_sampled: int = 0
    transitions: int = 0
    shots: int = 0
    box_scores: int = 0
    state_frames: dict[str, int] = field(default_factory=dict)


class Runner:
    """Samples a frame source and drives the state machine over it."""

    def __init__(
        self,
        source: FrameSource,
        config: Config,
        *,
        bus: EventBus | None = None,
        sample_fps: float | None = None,
        preview_every: int = 5,
    ) -> None:
        self.source = source
        self.config = config
        self.bus = bus or EventBus()
        self.classifier = ScreenClassifier(config)
        self.scoreboard = ScoreboardReader(config)
        self.shot_feedback = ShotFeedbackReader()
        self.box_score = BoxScoreParser()
        self.machine = StateMachine()
        self.stats = RunnerStats()

        # Frames of the banner currently on screen, read together once it
        # clears. None of them is trustworthy alone.
        self._banner: list[np.ndarray] = []
        self._shot_reader_ready: bool | None = None
        self._box_thread: threading.Thread | None = None
        self._box_reader_ready: bool | None = None

        # How often to publish HUD crops for the UI. Encoding costs real time,
        # so this runs well below the sample rate — the crops are for a human
        # to eyeball, and a human cannot read ten updates a second anyway.
        self.preview_every = max(1, preview_every)
        self._preview_regions = (
            "scoreboard", "game_clock", "score_home", "score_away",
            "shot_clock", "shot_feedback",
        )

        # Sample well below the capture rate. The HUD does not change fast
        # enough to justify 60fps, and the headroom belongs to the tracker.
        target = sample_fps or config.capture.get("sample_fps", 10)
        self._stride = max(1, int(round(self.source.fps / max(target, 1))))

    def run(self, max_frames: int | None = None) -> RunnerStats:
        log.info("Sampling every %d frame(s) from a %.0ffps source",
                 self._stride, self.source.fps)
        if not self.scoreboard.ready:
            log.warning(
                "No glyph atlas found — scoreboard fields will read as None. "
                "Run `2kw atlas` once you have footage."
            )

        for frame in self.source:
            self.stats.frames_seen += 1
            if frame.index % self._stride:
                continue
            self.stats.frames_sampled += 1

            self._process(frame)

            if max_frames and self.stats.frames_sampled >= max_frames:
                break
        return self.stats

    def _process(self, frame) -> None:
        if self.stats.frames_sampled % self.preview_every == 0:
            self._publish_preview(frame)

        observed, signals = self.classifier.classify(frame.image)
        key = observed.value
        self.stats.state_frames[key] = self.stats.state_frames.get(key, 0) + 1

        self.bus.publish(Event(
            kind="signals", frame_index=frame.index, video_ts=frame.timestamp,
            data={"observed": observed.value,
                  "edges": round(signals.scoreboard_edge_density, 4),
                  "dark": round(signals.scoreboard_dark_fraction, 3),
                  "luma": round(signals.mean_luma, 1),
                  "clock_moved": signals.clock_changed},
        ))

        transition = self.machine.update(
            observed, timestamp=frame.timestamp, frame_index=frame.index
        )
        if transition is not None:
            self.stats.transitions += 1
            self.bus.publish(Event(
                kind="state_change",
                frame_index=frame.index,
                video_ts=frame.timestamp,
                data={"previous": transition.previous.value,
                      "current": transition.current.value},
            ))
            if transition.current is GameState.POST_GAME:
                self._read_box_score(frame)

        # Everything below here is gated on being in a live game, which is the
        # whole reason the state machine is the first thing built.
        if not self.machine.is_active:
            return

        reading = self.scoreboard.read(frame.image)
        if reading.complete:
            self.bus.publish(Event(
                kind="scoreboard",
                frame_index=frame.index,
                video_ts=frame.timestamp,
                data={"score_home": reading.score_home,
                      "score_away": reading.score_away,
                      "game_clock": reading.game_clock,
                      "shot_clock": reading.shot_clock},
            ))

        self._collect_shot_feedback(frame)
        _ = signals

    def _read_box_score(self, frame) -> None:
        """Parse the box score on a worker thread, once per entry to the screen.

        This cannot run inline. A full parse is 130 cells against six OCR
        configurations each, and each of those is a Tesseract subprocess:
        measured at 98 seconds on this machine. In the frame loop that would
        stall capture for a minute and a half and drop thousands of frames.

        So the frame is copied and handed off, and the result arrives as an
        event whenever it is ready. One parse runs at a time — the screen can
        be reopened, and queueing them up would only pile work behind work.
        """
        if self._box_thread is not None and self._box_thread.is_alive():
            return
        if not self._box_score_ready():
            return

        image = frame.image.copy()
        index, timestamp = frame.index, frame.timestamp

        def work() -> None:
            try:
                box = self.box_score.parse(image)
            except Exception:                               # noqa: BLE001
                log.exception("Could not read the box score")
                return
            if not any(p.complete for p in box.players):
                # The screen was detected but nothing came off it. Better to
                # say nothing than to write ten empty rows.
                log.info("Box score parsed but no rows were readable")
                return
            self.stats.box_scores += 1
            self.bus.publish(Event(
                kind="box_score", frame_index=index, video_ts=timestamp,
                data={
                    "players": [_stat_line(p) for p in box.players],
                    "totals": {team: _stat_line(row)
                               for team, row in box.totals.items()},
                    "checksum_failures": list(box.checksum_failures),
                    "unread_cells": list(box.unread_cells),
                    "trustworthy": box.trustworthy,
                },
            ))

        self._box_thread = threading.Thread(target=work, daemon=True,
                                            name="boxscore")
        self._box_thread.start()

    def _box_score_ready(self) -> bool:
        if self._box_reader_ready is None:
            self._box_reader_ready = self.box_score.available()
            if not self._box_reader_ready:
                log.warning(
                    "Tesseract not found — box scores will not be logged. "
                    "Install it and restart to capture game stats."
                )
        return self._box_reader_ready

    def _collect_shot_feedback(self, frame) -> None:
        """Buffer a shot banner while it is up, and read it once it clears.

        The banner is on screen for about a second — dozens of frames — and at
        capture resolution no single one of them is reliable. So frames are
        accumulated while the cheap presence gate holds, then read together by
        consensus when it drops. That is also what keeps the cost sane: OCR
        runs once per shot rather than ten times a second.
        """
        if not self._shot_feedback_ready():
            return

        if self.shot_feedback.present(frame.image):
            # Cap the buffer: a gate stuck on must not grow without bound.
            if len(self._banner) < MAX_BANNER_FRAMES:
                self._banner.append(frame.image)
            return

        if not self._banner:
            return
        frames, self._banner = self._banner, []
        try:
            feedback = self.shot_feedback.read_event(frames)
        except Exception:                                   # noqa: BLE001
            log.exception("Could not read a shot banner")
            return
        if not feedback.any_read:
            # Most events that reach here were a false trigger on the presence
            # gate. Silence beats a row of nulls in the database.
            return

        self.stats.shots += 1
        self.bus.publish(Event(
            kind="shot_feedback",
            frame_index=frame.index,
            video_ts=frame.timestamp,
            data={"timing": feedback.timing,
                  "coverage": feedback.coverage,
                  "distance_feet": feedback.distance_feet,
                  "frames": len(frames),
                  "unmatched": feedback.unmatched},
        ))

    def _shot_feedback_ready(self) -> bool:
        """Whether the banner reader can run at all, checked once."""
        if self._shot_reader_ready is None:
            self._shot_reader_ready = self.shot_feedback.available()
            if not self._shot_reader_ready:
                log.warning(
                    "Tesseract not found — shot feedback will not be logged. "
                    "Install it and restart to capture shot timing."
                )
        return self._shot_reader_ready

    def _publish_preview(self, frame) -> None:
        """Publish the crops the parsers are working from.

        Showing a human the actual pixels beside the parsed value is what makes
        a bad region or a misread diagnosable. A wrong number on its own does
        not say whether the crop, the threshold or the atlas is at fault.
        """
        crops = {}
        for name in self._preview_regions:
            region = self.config.regions.get(name)
            if region is not None:
                crops[name] = region.crop(frame.image)
        self.bus.publish(Event(
            kind="preview",
            frame_index=frame.index,
            video_ts=frame.timestamp,
            data={"crops": crops, "frame": frame.image,
                  "frame_size": frame.size},
        ))


def _stat_line(row) -> dict:
    """One parsed row as plain data.

    is_you is carried through deliberately. It is the only thing on the screen
    that says which of the ten rows is yours, and it is what lets a shot be
    attributed to a player rather than filed against nobody.
    """
    return asdict(row)
