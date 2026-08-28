"""The main frame loop.

Structure worth preserving as this grows: the loop samples frames, classifies
screen state, and publishes events. It does not itself know about the database,
the dashboard, or the tracker — those attach as subscribers. Adding the tracker
later should mean registering another stage here, not restructuring this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..capture import FrameSource
from ..config import Config
from ..hud import ScoreboardReader
from ..state import GameState, StateMachine
from ..state.detectors import ScreenClassifier
from .events import Event, EventBus

log = logging.getLogger(__name__)


@dataclass
class RunnerStats:
    frames_seen: int = 0
    frames_sampled: int = 0
    transitions: int = 0
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
        self.machine = StateMachine()
        self.stats = RunnerStats()

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

        # TODO (next): shot-feedback detection off the `shot_feedback` region.
        # That is the highest-value signal in the project — every release logged
        # with its timing verdict and outcome — and it needs the same glyph
        # atlas treatment as the scoreboard, plus a template per verdict string.
        _ = signals

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
