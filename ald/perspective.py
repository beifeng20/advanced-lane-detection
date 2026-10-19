"""Bird's-eye view: a homography that maps the road plane trapezoid to a rectangle."""
import cv2
import numpy as np

# Chosen on the two undistorted straight-road images (data/test_images/straight_lines*.jpg) so that both
# lane lines become vertical after warping, at x = 320 and x = 960 (within ~13 px); 1280x720 frames.
SRC = np.float32([[589, 455], [695, 455], [1101, 720], [211, 720]])
DST = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])


class BirdsEye:
    def __init__(self, src: np.ndarray = SRC, dst: np.ndarray = DST):
        self.M = cv2.getPerspectiveTransform(src, dst)      # road -> top view
        self.Minv = cv2.getPerspectiveTransform(dst, src)   # top view -> road (to draw results back)

    def warp(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.M, (w, h), flags=cv2.INTER_LINEAR)

    def unwarp(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.Minv, (w, h), flags=cv2.INTER_LINEAR)
