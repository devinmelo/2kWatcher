import math

import numpy as np
import pytest

from twokwatcher.vision import court, homography


def _synthetic_camera(width=1920, height=1080):
    """A plausible broadcast-ish view of the court, as a court->image matrix.

    Built by picking where four known court points land on screen, which is how
    a real camera would be described anyway.
    """
    import cv2

    src = np.array([
        (-court.HALF_LENGTH, -court.HALF_WIDTH),
        (court.HALF_LENGTH, -court.HALF_WIDTH),
        (court.HALF_LENGTH, court.HALF_WIDTH),
        (-court.HALF_LENGTH, court.HALF_WIDTH),
    ], dtype=np.float32)
    # Far sideline compressed toward the centre: ordinary perspective.
    dst = np.array([
        (360.0, 430.0), (1560.0, 430.0), (1850.0, 980.0), (70.0, 980.0),
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _project(matrix, points):
    import cv2
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def test_court_dimensions_are_regulation():
    assert court.LENGTH == 94.0
    assert court.WIDTH == 50.0
    # Rim sits 5.25ft in from the baseline.
    assert court.BASKET_X == pytest.approx(41.75)
    # Free-throw line is 19ft from the baseline.
    assert court.FT_LINE_X == pytest.approx(28.0)


def test_landmarks_all_lie_on_the_court():
    for name, (x, y) in court.LANDMARKS.items():
        assert court.is_inside(x, y), f"{name} is off the floor"


def test_zones_classify_known_spots():
    # Directly under the rim.
    assert court.Zone.of(court.BASKET_X, 0.0) == "restricted"
    # Deep in the corner, outside the corner-three line.
    assert court.Zone.of(45.0, 24.0) == "corner_three"
    # Straightaway well beyond the arc.
    assert court.Zone.of(10.0, 0.0) == "above_break_three"
    # Elbow.
    assert court.Zone.of(28.0, 7.0) == "paint"


def test_homography_recovers_known_positions():
    """Round trip: project landmarks through a known camera, then solve."""
    camera = _synthetic_camera()
    correspondences = {
        name: tuple(_project(camera, [pos])[0])
        for name, pos in court.LANDMARKS.items()
    }

    fitted = homography.fit(correspondences)
    assert fitted.inlier_ratio > 0.95

    # Every landmark should come back within an inch of where it belongs.
    for name, expected in court.LANDMARKS.items():
        got = fitted.to_court(np.array([correspondences[name]]))[0]
        assert math.dist(got, expected) < 0.1, name


def test_homography_survives_noisy_and_mislocated_landmarks():
    """RANSAC should reject gross outliers rather than be dragged by them."""
    rng = np.random.default_rng(0)
    camera = _synthetic_camera()
    correspondences = {}
    for i, (name, pos) in enumerate(sorted(court.LANDMARKS.items())):
        px = _project(camera, [pos])[0] + rng.normal(0, 1.0, 2)
        # Every fifth landmark is badly wrong, as a detector's would be.
        if i % 5 == 0:
            px = px + rng.normal(0, 120.0, 2)
        correspondences[name] = tuple(px)

    fitted = homography.fit(correspondences, ransac_threshold=2.0)
    centre = fitted.to_court(np.array([_project(camera, [(0.0, 0.0)])[0]]))[0]
    assert math.dist(centre, (0.0, 0.0)) < 2.0


def test_fit_rejects_too_few_points():
    with pytest.raises(homography.HomographyError, match="at least 4"):
        homography.fit({"centre": (10.0, 10.0)})


def test_fit_rejects_unknown_landmark_names():
    with pytest.raises(homography.HomographyError, match="Unknown"):
        homography.fit({f"bogus_{i}": (float(i), 1.0) for i in range(5)})


def test_feet_not_centroid_is_projected():
    """A box's centre floats up the torso; its feet are on the floor."""
    camera = _synthetic_camera()
    fitted = homography.fit({
        name: tuple(_project(camera, [pos])[0])
        for name, pos in court.LANDMARKS.items()
    })
    standing_at = (20.0, 5.0)
    feet_px = _project(camera, [standing_at])[0]
    box = np.array([[feet_px[0] - 20, feet_px[1] - 90,
                     feet_px[0] + 20, feet_px[1]]])
    got = fitted.feet_to_court(box)[0]
    assert math.dist(got, standing_at) < 0.5


def test_smoother_holds_last_good_fit_through_a_bad_one():
    camera = _synthetic_camera()
    good = homography.fit({
        name: tuple(_project(camera, [pos])[0])
        for name, pos in court.LANDMARKS.items()
    })

    smoother = homography.HomographySmoother()
    assert smoother.update(good) is good

    # A fit with almost no support must not displace it.
    weak = homography.CourtHomography(matrix=good.matrix.copy(),
                                      inliers=2, total=20)
    assert smoother.update(weak) is good
    assert smoother.rejected == 1

    # Nor should a well-supported fit that teleports the court.
    wild = homography.CourtHomography(matrix=good.matrix @ np.diag([3.0, 3.0, 1.0]),
                                      inliers=20, total=20)
    smoother.update(wild)
    assert smoother.rejected == 2


def test_overlay_draws_without_blowing_up():
    camera = _synthetic_camera()
    fitted = homography.fit({
        name: tuple(_project(camera, [pos])[0])
        for name, pos in court.LANDMARKS.items()
    })
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = homography.draw_overlay(frame, fitted)
    assert out.shape == frame.shape
    # Something was actually drawn.
    assert out.any()
