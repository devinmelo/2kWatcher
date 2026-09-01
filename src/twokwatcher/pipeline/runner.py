"""The main frame loop.

Structure worth preserving as this grows: the loop samples frames, classifies
screen state, and publishes events. It does not itself know about the database,
the dashboard, or the tracker — those attach as subscribers. Adding the tracker
later should mean registering another stage here, not restructuring this file.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from ..capture import FrameSource
from ..config import Config
from ..hud import (BoxScoreParser, NameplateReader, ScoreboardReader,
                   ShotFeedbackReader)
from ..state import GameState, StateMachine
from ..state.detectors import ScreenClassifier
from .events import Event, EventBus

log = logging.getLogger(__name__)

# A banner is up for about a second. This is a generous ceiling on how many
# frames of one are worth keeping, so a gate that sticks on cannot grow the
# buffer without bound.
MAX_BANNER_FRAMES = 40

# Frames kept from before a shot, to find who was holding the ball.
#
# By the time the banner renders the ball is already in the air and the
# shooter's plate has usually gone: measured over a real session, 22 of 60
# banner frames carried no plate at all. So the shooter is looked for in the
# frames leading up to the banner rather than in the banner itself.
#
# Only a horizontal band is kept. Plates appear over players, which across 72
# sightings never rose above y=283 or fell below y=925, so the crowd and the
# scoreboard can be dropped — that is a third of every frame not held in
# memory, and a third less to search.
SHOOTER_LOOKBACK_FRAMES = 15
SHOOTER_BAND = (0.24, 0.88)          # fractions of frame height
# How far back to actually look. The gap between losing the ball and the banner
# appearing is a few frames; searching further just costs time and risks
# naming whoever had it on the previous possession.
SHOOTER_SEARCH_DEPTH = 6

# Clips play back at the rate the loop samples at, so a second of game is a
# second of clip.
CLIP_FPS = 10.0


@dataclass
class RunnerStats:
    frames_seen: int = 0
    frames_sampled: int = 0
    transitions: int = 0
    shots: int = 0
    shots_attributed: int = 0
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
        clip_dir=None,
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

        # Recent frames, for working out who took a shot after the fact.
        self.nameplates = NameplateReader()
        self._recent: deque = deque(maxlen=SHOOTER_LOOKBACK_FRAMES)
        self._plate_reader_ready: bool | None = None
        # Who is in this game, and which of them is you. Both come from the box
        # score, so shooters can only be named once one has been read — which
        # is what makes the reading reliable rather than open-ended OCR.
        self.roster: list[str] = []
        self.me: str | None = None
        # Where per-shot clips are written, or None to write none. The caller
        # namespaces this per session — filenames alone would collide across
        # runs, which is exactly how the frame collector loses its history.
        self.clip_dir = clip_dir

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
            # Hand the roster back to the loop. Naming a shooter is only
            # reliable because the candidates are known, so this is what turns
            # plate reading from open-ended OCR into a ten-way choice.
            named = [p.name for p in box.players if p.name]
            you = next((p.name for p in box.players if p.is_you), None)
            if named:
                self.set_roster(named, you)

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
            # Nothing in flight, so this frame is run-up for the next shot.
            self._remember(frame)
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

        shooter = self._find_shooter()
        clip = self._write_clip(frames, frame.index)
        self.stats.shots += 1
        if shooter is not None:
            self.stats.shots_attributed += 1
        self.bus.publish(Event(
            kind="shot_feedback",
            frame_index=frame.index,
            video_ts=frame.timestamp,
            data={"timing": feedback.timing,
                  "coverage": feedback.coverage,
                  "distance_feet": feedback.distance_feet,
                  "frames": len(frames),
                  "shooter": shooter,
                  "clip": clip,
                  "unmatched": feedback.unmatched},
        ))
        # This frame is past the banner, so it belongs to the next shot's
        # run-up — not to the one just written.
        self._recent.clear()
        self._remember(frame)

    def _write_clip(self, banner_frames, index: int) -> str | None:
        """Save the second before a shot, and the banner that ended it.

        The lookback the shooter search uses is also, for free, a recording of
        the shot that produced it — so it is worth keeping. The banner frames
        follow the run-up, which makes a clip that shows the release and then
        the verdict on it, and that is what makes a logged shot checkable by a
        human rather than merely plausible.

        Written on a worker thread. Encoding a second of 1080p costs about
        290ms, which is nothing once per shot but far too much inside a loop
        with a 100ms budget per frame.
        """
        if self.clip_dir is None:
            return None
        frames = [image for _, image in self._recent] + list(banner_frames)
        if not frames:
            return None

        directory = Path(self.clip_dir)
        path = directory / f"shot-{index:07d}.mp4"

        def work() -> None:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                height, width = frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                    CLIP_FPS, (width, height))
                try:
                    for image in frames:
                        if image.shape[:2] == (height, width):
                            writer.write(image)
                finally:
                    writer.release()
            except Exception:                               # noqa: BLE001
                log.exception("Could not write the clip for shot %s", index)

        threading.Thread(target=work, daemon=True, name="clip").start()
        return str(path)

    def _remember(self, frame) -> None:
        """Keep this frame, in case the next shot needs it.

        Whole frames, not the plate band. The band is all the shooter search
        needs, but a clip of a shot has to show the banner too, and that sits
        above the band — a clip cropped to the plates would be missing the very
        thing it is evidence for.
        """
        image = frame.image
        if image is None or image.size == 0:
            return
        self._recent.append((frame.index, image.copy()))

    def _find_shooter(self) -> str | None:
        """Who was holding the ball in the frames just before the banner.

        2K names two players at any moment: your own, wherever they are, and
        whoever has the ball. So the shooter names himself — right up until he
        releases, at which point the plate goes and the banner arrives. Reading
        backwards from the banner is what catches him.

        Deliberately not run on every frame. Finding and reading a plate costs
        far more than the whole rest of the loop put together, so it happens
        once per shot, over a handful of frames, and stops at the first answer.
        """
        if not self.roster or not self._plates_ready():
            return None

        saw_your_plate = False
        for _, image in list(self._recent)[-SHOOTER_SEARCH_DEPTH:][::-1]:
            h = image.shape[0]
            band = image[int(SHOOTER_BAND[0] * h):int(SHOOTER_BAND[1] * h)]
            handler = self.nameplates.ball_handler(band, self.roster, self.me)
            if handler is None:
                continue
            if handler.name != self.me:
                return handler.name
            saw_your_plate = True

        # Your plate up and nobody else's is what the screen looks like when
        # the ball is yours, so that is an answer. Reading no plate at all is
        # not: it means the run-up was unreadable, and claiming it anyway was
        # crediting you with shots you never took — 22 attributed against 12
        # attempts in the box score, because the two cases were conflated.
        return self.me if saw_your_plate else None

    def _plates_ready(self) -> bool:
        if self._plate_reader_ready is None:
            self._plate_reader_ready = self.nameplates.available()
            if not self._plate_reader_ready:
                log.warning("Tesseract not found — shots will not be attributed.")
        return self._plate_reader_ready

    def set_roster(self, roster: list[str], me: str | None = None) -> None:
        """Tell the runner who is in this game, so shooters can be named."""
        self.roster = list(roster)
        self.me = me

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
