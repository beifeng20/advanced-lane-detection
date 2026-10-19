import cv2
import numpy as np

from ald.lanefit import XM_PER_PIX, YM_PER_PIX, LaneFit, curvature_m, fit, offset_m, plausible, sliding_window
from ald.perspective import BirdsEye

H, W = 720, 1280


def lane_image(left_coeffs, right_coeffs) -> np.ndarray:
    """Binary bird's-eye image with two 10 px wide lines x = A y^2 + B y + C."""
    mask = np.zeros((H, W), np.uint8)
    for y in range(H):
        for c in (left_coeffs, right_coeffs):
            x = int(np.polyval(c, y))
            mask[y, max(x - 5, 0):x + 5] = 1
    return mask


def test_fit_recovers_known_parabolas():
    left, right = [2e-4, -0.2, 380.0], [2e-4, -0.2, 1000.0]
    f = fit(*sliding_window(lane_image(left, right)))
    assert f is not None
    for got, want in ((f.left, left), (f.right, right)):
        ys = np.array([0, 360, 719])
        assert np.abs(np.polyval(got, ys) - np.polyval(want, ys)).max() < 3  # pixels


def test_curvature_of_a_circle():
    # in meters, a circle of radius R near its lowest point is x ~ y^2 / (2R)
    R = 800.0
    A_m = 1 / (2 * R)
    A_px = A_m * YM_PER_PIX ** 2 / XM_PER_PIX  # convert the coefficient back to pixel units
    assert abs(curvature_m(np.array([A_px, 0.0, 640.0]), 0) - R) < 1.0


def test_straight_line_has_huge_radius():
    assert curvature_m(np.array([0.0, 0.0, 640.0]), 719) > 1e6


def test_offset_sign():
    # lane centered at x = 600 while the camera is at 640 -> car is right of center
    f = LaneFit(np.array([0, 0, 280.0]), np.array([0, 0, 920.0]), 0, 0)
    assert offset_m(f, H, W) > 0


def test_plausibility_check():
    good = LaneFit(np.array([0, 0, 320.0]), np.array([0, 0, 960.0]), 0, 0)    # 3.7 m wide, parallel
    narrow = LaneFit(np.array([0, 0, 500.0]), np.array([0, 0, 800.0]), 0, 0)  # 1.7 m wide
    crossing = LaneFit(np.array([0, 0.5, 320.0]), np.array([0, 0, 960.0]), 0, 0)
    assert plausible(good, H)
    assert not plausible(narrow, H)
    assert not plausible(crossing, H)


def test_warp_unwarp_roundtrip():
    view = BirdsEye()
    pts = np.float32([[[640, 600]]])
    back = cv2.perspectiveTransform(cv2.perspectiveTransform(pts, view.M), view.Minv)
    assert np.allclose(back, pts, atol=1e-3)
