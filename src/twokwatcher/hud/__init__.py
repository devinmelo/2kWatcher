"""Reading the on-screen HUD."""

from .boxscore import BoxScore, BoxScoreParser, PlayerRow, resolve_names
from .scoreboard import Scoreboard, ScoreboardReader, preprocess_digits

__all__ = [
    "BoxScore",
    "BoxScoreParser",
    "PlayerRow",
    "Scoreboard",
    "ScoreboardReader",
    "preprocess_digits",
    "resolve_names",
]
