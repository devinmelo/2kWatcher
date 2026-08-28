"""The screen/game state machine.

This is the cheapest and most important component in the project. Everything
expensive downstream — HUD parsing, and later the tracker — is gated on it, for
two reasons:

  1. Compute. There is no point running a detector over a menu.
  2. Data quality, which matters more. Frames from replays, cutscenes and
     timeouts produce plausible-looking garbage. Excluding them at the source
     is far easier than filtering them out of the database later.

The strongest available signal for "live play is happening" is the game clock
decrementing. A running clock is near-perfect ground truth, and it costs one
small crop per sample.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class GameState(str, Enum):
    UNKNOWN = "unknown"      # startup, or nothing matched
    MENU = "menu"            # menus, lobbies, the Neighborhood
    LOADING = "loading"      # load screens between states
    LIVE = "live"            # ball is live, clock running — the state that matters
    DEAD_BALL = "dead_ball"  # in-game but clock stopped: free throws, timeouts
    REPLAY = "replay"        # replay or cutscene; HUD present but not live action
    POST_GAME = "post_game"  # box score / summary screens — the stat goldmine


# States in which frames are worth handing to expensive downstream stages.
ACTIVE_STATES = frozenset({GameState.LIVE, GameState.DEAD_BALL})


@dataclass(frozen=True)
class StateTransition:
    previous: GameState
    current: GameState
    at: float           # source timestamp, seconds
    frame_index: int


class StateMachine:
    """Debounced state tracking.

    Raw per-frame classification is noisy — a scoreboard occluded by an
    animation for three frames should not read as leaving the game. Transitions
    only commit after a candidate state has held for `min_frames`.
    """

    def __init__(self, min_frames: int = 3) -> None:
        self.min_frames = min_frames
        self.state = GameState.UNKNOWN
        self.entered_at = time.monotonic()
        self._candidate: GameState | None = None
        self._candidate_count = 0
        self.history: list[StateTransition] = []

    def update(
        self, observed: GameState, *, timestamp: float, frame_index: int
    ) -> StateTransition | None:
        """Feed one classification. Returns a transition if one committed."""
        if observed == self.state:
            self._candidate = None
            self._candidate_count = 0
            return None

        if observed == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = observed
            self._candidate_count = 1

        if self._candidate_count < self.min_frames:
            return None

        transition = StateTransition(
            previous=self.state,
            current=observed,
            at=timestamp,
            frame_index=frame_index,
        )
        self.state = observed
        self.entered_at = time.monotonic()
        self._candidate = None
        self._candidate_count = 0
        self.history.append(transition)
        return transition

    @property
    def is_active(self) -> bool:
        """Whether expensive downstream work should run on current frames."""
        return self.state in ACTIVE_STATES
