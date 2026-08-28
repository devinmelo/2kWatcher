"""Per-frame screen classification.

Intentionally heuristic and cheap. These run on every sampled frame, so they
must stay far below the cost of anything they gate.

CALIBRATION NOTE: the thresholds here are starting points, not tuned values.
They need to be fitted against real footage — `2kw probe` dumps the underlying
measurements per frame so you can see where the actual separation lies.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Config
from .machine import GameState


@dataclass
class FrameSignals:
    """Cheap measurements taken from one frame, before any interpretation."""

    scoreboard_present: bool
    scoreboard_edge_density: float
    mean_luma: float
    clock_changed: bool


class ScreenClassifier:
    """Turns a frame into a candidate GameState.

    The core signal is the scoreboard bug: it is present with high edge density
    during a game and absent in menus. Distinguishing LIVE from DEAD_BALL and
    REPLAY then hinges on whether the game clock is advancing, which is why this
    class keeps a little history rather than being a pure function.
    """

    def __init__(self, config: Config, *, edge_threshold: float = 0.045) -> None:
        self.config = config
        self.edge_threshold = edge_threshold
        self._last_clock_crop: np.ndarray | None = None
        self._clock_static_samples = 0

    def signals(self, image: np.ndarray) -> FrameSignals:
        scoreboard = self.config.region("scoreboard").crop(image)
        edge_density = _edge_density(scoreboard)

        clock_crop = self.config.region("game_clock").crop(image)
        clock_changed = self._clock_changed(clock_crop)

        return FrameSignals(
            scoreboard_present=edge_density >= self.edge_threshold,
            scoreboard_edge_density=edge_density,
            mean_luma=float(np.mean(image)) if image.size else 0.0,
            clock_changed=clock_changed,
        )

    def classify(self, image: np.ndarray) -> tuple[GameState, FrameSignals]:
        sig = self.signals(image)

        # A near-black frame is a load screen regardless of anything else.
        if sig.mean_luma < 12.0:
            return GameState.LOADING, sig

        if not sig.scoreboard_present:
            # TODO: separate MENU from POST_GAME. The box score screen has a
            # distinctive dense-text layout; a template match on the header, or
            # a simple text-row-density heuristic, should split them cleanly.
            return GameState.MENU, sig

        # Scoreboard is up, so we are in a game. A moving clock means live ball.
        if sig.clock_changed:
            self._clock_static_samples = 0
            return GameState.LIVE, sig

        self._clock_static_samples += 1
        # A clock stopped briefly is a dead ball; stopped for a long stretch
        # while the HUD persists is more likely a replay or cutscene.
        # TODO: replays in 2K are visually distinctive (letterboxing, a REPLAY
        # tag, a camera angle the gameplay cameras never use). Detect that
        # directly rather than inferring it from clock staleness.
        if self._clock_static_samples > 40:
            return GameState.REPLAY, sig
        return GameState.DEAD_BALL, sig

    def _clock_changed(self, crop: np.ndarray) -> bool:
        """Whether the game-clock crop differs from the previous sample.

        Comparing pixels rather than OCRing digits is deliberate: it is orders
        of magnitude cheaper, and "did this change" is all the state machine
        needs. Actually reading the clock is the HUD parser's job, and only
        happens once we already know we are in a game.
        """
        if crop.size == 0:
            return False
        small = cv2.cvtColor(cv2.resize(crop, (32, 16)), cv2.COLOR_BGR2GRAY)
        previous, self._last_clock_crop = self._last_clock_crop, small
        if previous is None:
            return False
        return float(np.mean(cv2.absdiff(small, previous))) > 2.0


def _edge_density(crop: np.ndarray) -> float:
    """Fraction of pixels in a crop that sit on an edge.

    HUD elements are crisp synthetic graphics with hard borders and text, so
    they score far higher than gameplay or menu backgrounds.
    """
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    return float(np.count_nonzero(edges)) / float(edges.size)
