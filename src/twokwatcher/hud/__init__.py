"""Reading the on-screen HUD."""

from .boxscore import BoxScore, BoxScoreParser, PlayerRow, resolve_names
from .playerpanel import PlayerPanel, PlayerPanelReader, PlayerStat
from .scoreboard import Scoreboard, ScoreboardReader, preprocess_digits
from .shotfeedback import ShotFeedback, ShotFeedbackReader

__all__ = [
    "BoxScore",
    "BoxScoreParser",
    "PlayerPanel",
    "PlayerPanelReader",
    "PlayerRow",
    "PlayerStat",
    "Scoreboard",
    "ScoreboardReader",
    "ShotFeedback",
    "ShotFeedbackReader",
    "preprocess_digits",
    "resolve_names",
]
