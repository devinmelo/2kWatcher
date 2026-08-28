"""Mapping between screen pixels and court feet.

This is the keystone of the tracker, and the reason it gets built before any
detector. Once a frame's homography is known:

  * detections become court positions, which are comparable across frames
  * tracking runs in court space, where players move with plausible basketball
    physics (~15-20 ft/s, smoothly) instead of image space, where apparent
    motion during a camera pan is dominated by the camera rather than by the
    players — precisely when the action is fastest and association matters most
  * it is self-validating: reproject the court template onto the frame and look
    at whether the lines land on the lines. No labelled data needed to tell
    whether it works.

A homography is valid because a basketball court is planar. Players are not
points on that plane, so a detection's *feet* are what get projected, never its
centroid — a box centre floats somewhere up the player's torso and would place
them several feet behind where they stand.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import court


class HomographyError(ValueError):
    """Raised when a homography cannot be fitted from what was supplied."""


@dataclass
class CourtHomography:
    """A fitted image <-> court transform for one frame."""

    matrix: np.ndarray            # 3x3, image pixels -> court feet
    inliers: int
    total: int

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.total if self.total else 0.0

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)

    def to_court(self, points: np.ndarray) -> np.ndarray:
        """Project image points (N,2) to court feet (N,2)."""
        return _transform(np.asarray(points, dtype=np.float64), self.matrix)

    def to_image(self, points: np.ndarray) -> np.ndarray:
        """Project court points (N,2) back to image pixels (N,2)."""
        return _transform(np.asarray(points, dtype=np.float64), self.inverse)

    def feet_to_court(self, boxes: np.ndarray) -> np.ndarray:
        """Project detection boxes (N,4 as xyxy) to court positions.

        Uses the bottom-centre of each box — where the player meets the floor,
        which is the only part of them actually on the court plane.
        """
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]], axis=1)
        return self.to_court(feet)


def fit(
    correspondences: dict[str, tuple[float, float]],
    *,
    ransac_threshold: float = 1.5,
    min_points: int = 4,
) -> CourtHomography:
    """Fit a homography from named landmark detections.

    `correspondences` maps landmark names (see `court.LANDMARKS`) to the pixel
    position where that landmark was found. Four are the mathematical minimum;
    more is substantially better, because RANSAC needs surplus points to have
    anything to reject.

    The threshold is in FEET, not pixels: it is applied on the court side, where
    a fixed tolerance means the same thing everywhere. In pixels, the same error
    is far more forgiving at the near sideline than at the far one.
    """
    known = {n: p for n, p in correspondences.items() if n in court.LANDMARKS}
    unknown = set(correspondences) - set(known)
    if unknown:
        raise HomographyError(f"Unknown landmark(s): {sorted(unknown)}")
    if len(known) < min_points:
        raise HomographyError(
            f"Need at least {min_points} landmarks, got {len(known)}"
        )

    names = sorted(known)
    src = np.array([known[n] for n in names], dtype=np.float64)
    dst = np.array([court.LANDMARKS[n] for n in names], dtype=np.float64)

    matrix, mask = cv2.findHomography(
        src, dst, cv2.RANSAC, ransacReprojThreshold=ransac_threshold
    )
    if matrix is None:
        raise HomographyError(
            "findHomography failed — landmarks are probably collinear or "
            "badly mislocated"
        )
    return CourtHomography(
        matrix=matrix,
        inliers=int(mask.sum()) if mask is not None else len(names),
        total=len(names),
    )


class HomographySmoother:
    """Rejects bad per-frame fits using temporal continuity.

    2K's camera moves smoothly, so a homography that jumps between adjacent
    frames is a bad fit rather than real motion. Rather than trusting each
    frame independently, hold the last good transform and only accept a new one
    when it is both well-supported and close to what came before.

    Deliberately simple. A full Kalman filter over the matrix parameters would
    be better, but this catches the failure that actually matters — a wild fit
    from a frame where the court was mostly occluded — and is easy to reason
    about when the tracker misbehaves.
    """

    def __init__(
        self,
        *,
        min_inlier_ratio: float = 0.6,
        max_jump_feet: float = 4.0,
        alpha: float = 0.35,
    ) -> None:
        self.min_inlier_ratio = min_inlier_ratio
        self.max_jump_feet = max_jump_feet
        self.alpha = alpha
        self.current: CourtHomography | None = None
        self.rejected = 0

    def update(self, candidate: CourtHomography | None) -> CourtHomography | None:
        """Offer a new fit. Returns the transform to actually use."""
        if candidate is None or candidate.inlier_ratio < self.min_inlier_ratio:
            self.rejected += 1
            return self.current

        if self.current is None:
            self.current = candidate
            return self.current

        if self._displacement(self.current, candidate) > self.max_jump_feet:
            self.rejected += 1
            return self.current

        # Blend toward the new fit rather than snapping to it, so per-frame
        # jitter does not propagate into court positions.
        blended = (1.0 - self.alpha) * self.current.matrix + self.alpha * candidate.matrix
        self.current = CourtHomography(
            matrix=blended / blended[2, 2],
            inliers=candidate.inliers,
            total=candidate.total,
        )
        return self.current

    def reset(self) -> None:
        """Drop history. Call on a camera cut, replay, or state change."""
        self.current = None

    @staticmethod
    def _displacement(a: CourtHomography, b: CourtHomography) -> float:
        """Mean court-space disagreement between two transforms, in feet.

        Comparing matrices elementwise is meaningless — the entries are not
        commensurate. Comparing where they send the same probe points is.
        """
        probes = np.array(
            [[0.0, 0.0], [500.0, 0.0], [0.0, 500.0], [500.0, 500.0],
             [250.0, 250.0]],
            dtype=np.float64,
        )
        return float(np.linalg.norm(a.to_court(probes) - b.to_court(probes),
                                    axis=1).mean())


def draw_overlay(
    image: np.ndarray,
    homography: CourtHomography,
    *,
    colour: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Reproject the court template onto a frame.

    This is the whole validation story for the homography stage: if the drawn
    lines sit on the painted lines, the fit is right. It needs no ground truth
    and no labelling — just eyes.
    """
    canvas = image.copy()
    for start, end in court.template_lines():
        pts = homography.to_image(np.array([start, end]))
        if not np.isfinite(pts).all():
            continue
        cv2.line(canvas, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)),
                 colour, thickness, cv2.LINE_AA)

    for arc in court.template_arcs():
        pts = homography.to_image(arc)
        if not np.isfinite(pts).all():
            continue
        cv2.polylines(canvas, [pts.astype(np.int32)], False, colour,
                      thickness, cv2.LINE_AA)
    return canvas


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = points.reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
