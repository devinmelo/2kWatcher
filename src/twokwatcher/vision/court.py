"""The court model.

Everything the tracker produces is expressed in this coordinate system rather
than in pixels. Pixel positions are meaningless the moment the camera pans: the
same spot on the floor lands somewhere different every frame, and two players
standing side by side can be hundreds of pixels apart depending on where they
are on screen. Court feet are stable, comparable across frames and games, and
directly interpretable — "23 feet out on the left wing" is analysis, "(847,
312)" is not.

Coordinates are in FEET, origin at centre court:

    x  runs baseline to baseline, -47 .. +47
    y  runs sideline to sideline, -25 .. +25

Dimensions are the NBA regulation court, which is what 2K models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LENGTH = 94.0          # baseline to baseline
WIDTH = 50.0           # sideline to sideline
HALF_LENGTH = LENGTH / 2
HALF_WIDTH = WIDTH / 2

PAINT_WIDTH = 16.0                 # the key, sideline-to-sideline
PAINT_HALF = PAINT_WIDTH / 2
FT_LINE_FROM_BASELINE = 19.0       # 15ft from the backboard, 4ft of overhang
FT_CIRCLE_RADIUS = 6.0
CENTRE_CIRCLE_RADIUS = 6.0
BASKET_FROM_BASELINE = 5.25        # centre of the rim
THREE_RADIUS = 23.75               # arc, measured from the rim centre
CORNER_THREE_Y = 22.0              # straight section, from the centre line
CORNER_THREE_LENGTH = 14.0         # how far the straight section runs in
RESTRICTED_RADIUS = 4.0

# x of each basket centre
BASKET_X = HALF_LENGTH - BASKET_FROM_BASELINE      # 41.75
FT_LINE_X = HALF_LENGTH - FT_LINE_FROM_BASELINE    # 28.0
CORNER_THREE_X = HALF_LENGTH - CORNER_THREE_LENGTH  # 33.0


def _landmarks() -> dict[str, tuple[float, float]]:
    """Well-defined points a detector can plausibly localise.

    Chosen for being unambiguous corners and intersections rather than points
    along a smooth curve — a detector can pin "where the free-throw line meets
    the paint" far more precisely than "somewhere on the three-point arc".

    Naming: `l`/`r` is the left/right basket (-x/+x), `t`/`b` is the top/bottom
    sideline (-y/+y).
    """
    marks: dict[str, tuple[float, float]] = {}

    for sx, side in ((-1.0, "l"), (1.0, "r")):
        for sy, edge in ((-1.0, "t"), (1.0, "b")):
            marks[f"corner_{side}{edge}"] = (sx * HALF_LENGTH, sy * HALF_WIDTH)
            # Where the paint meets the baseline.
            marks[f"paint_baseline_{side}{edge}"] = (
                sx * HALF_LENGTH, sy * PAINT_HALF)
            # Where the free-throw line meets the paint.
            marks[f"paint_ft_{side}{edge}"] = (sx * FT_LINE_X, sy * PAINT_HALF)
            # The corner-three line, at the baseline and at its inner end.
            marks[f"corner3_baseline_{side}{edge}"] = (
                sx * HALF_LENGTH, sy * CORNER_THREE_Y)
            marks[f"corner3_break_{side}{edge}"] = (
                sx * CORNER_THREE_X, sy * CORNER_THREE_Y)

        marks[f"ft_centre_{side}"] = (sx * FT_LINE_X, 0.0)
        marks[f"basket_{side}"] = (sx * BASKET_X, 0.0)

    marks["centre"] = (0.0, 0.0)
    marks["halfcourt_t"] = (0.0, -HALF_WIDTH)
    marks["halfcourt_b"] = (0.0, HALF_WIDTH)
    return marks


LANDMARKS: dict[str, tuple[float, float]] = _landmarks()


@dataclass(frozen=True)
class Zone:
    """A named region of the floor, for bucketing shots and positions."""

    name: str

    @staticmethod
    def of(x: float, y: float) -> str:
        """Classify a court position into a shot zone.

        Uses the nearer basket, so the same logic works at both ends.
        """
        basket_x = math.copysign(BASKET_X, x) if x else BASKET_X
        dx, dy = x - basket_x, y
        distance = math.hypot(dx, dy)

        if distance <= RESTRICTED_RADIUS:
            return "restricted"
        if abs(x) >= FT_LINE_X and abs(y) <= PAINT_HALF:
            return "paint"

        # Corner threes live outside the straight section, not the arc.
        if abs(y) >= CORNER_THREE_Y and abs(x) >= CORNER_THREE_X:
            return "corner_three"
        if distance >= THREE_RADIUS:
            return "above_break_three"
        if distance >= 16.0:
            return "long_mid"
        return "short_mid"


def is_inside(x: float, y: float, *, margin: float = 0.0) -> bool:
    """Whether a court position is on the floor.

    Used to reject bad detections and bad homographies: a player projected 80
    feet off the side of the court is a sign something upstream is wrong.
    """
    return abs(x) <= HALF_LENGTH + margin and abs(y) <= HALF_WIDTH + margin


def template_lines() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The court's straight lines, for drawing the validation overlay."""
    h, w, p = HALF_LENGTH, HALF_WIDTH, PAINT_HALF
    lines = [
        ((-h, -w), (h, -w)), ((-h, w), (h, w)),      # sidelines
        ((-h, -w), (-h, w)), ((h, -w), (h, w)),      # baselines
        ((0, -w), (0, w)),                            # halfcourt
    ]
    for sx in (-1.0, 1.0):
        lines += [
            ((sx * h, -p), (sx * FT_LINE_X, -p)),     # paint, near sideline
            ((sx * h, p), (sx * FT_LINE_X, p)),       # paint, far sideline
            ((sx * FT_LINE_X, -p), (sx * FT_LINE_X, p)),   # free-throw line
            ((sx * h, -CORNER_THREE_Y),
             (sx * CORNER_THREE_X, -CORNER_THREE_Y)),      # corner three
            ((sx * h, CORNER_THREE_Y),
             (sx * CORNER_THREE_X, CORNER_THREE_Y)),
        ]
    return lines


def template_arcs(segments: int = 48) -> list[np.ndarray]:
    """The court's curves, as polylines, for the validation overlay."""
    arcs = [_circle(0.0, 0.0, CENTRE_CIRCLE_RADIUS, segments)]
    for sx in (-1.0, 1.0):
        arcs.append(_circle(sx * FT_LINE_X, 0.0, FT_CIRCLE_RADIUS, segments))
        arcs.append(_three_point_arc(sx, segments))
    return arcs


def _circle(cx: float, cy: float, r: float, segments: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def _three_point_arc(sign: float, segments: int) -> np.ndarray:
    """The arc between the two corner-three breaks, around one basket."""
    basket_x = sign * BASKET_X
    # Angle at which the arc meets the straight corner section.
    limit = math.asin(min(1.0, CORNER_THREE_Y / THREE_RADIUS))
    t = np.linspace(-limit, limit, segments + 1)
    # Sweep away from the baseline, which is -x for the right basket.
    xs = basket_x - sign * THREE_RADIUS * np.cos(t)
    ys = THREE_RADIUS * np.sin(t)
    return np.stack([xs, ys], axis=1)
