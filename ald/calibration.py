"""Camera calibration from chessboard photos (pinhole model + radial/tangential distortion)."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calibration:
    K: np.ndarray          # 3x3 intrinsic matrix: focal lengths fx, fy and principal point cx, cy (pixels)
    dist: np.ndarray       # distortion coefficients k1, k2, p1, p2, k3
    rms: float             # RMS reprojection error, pixels
    used: list[str]        # images where the chessboard was found
    skipped: list[str]

    def undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.undistort(img, self.K, self.dist, None, self.K)

    def save(self, path: str) -> None:
        np.savez(path, K=self.K, dist=self.dist, rms=self.rms)

    @staticmethod
    def load(path: str) -> "Calibration":
        d = np.load(path)
        return Calibration(d["K"], d["dist"], float(d["rms"]), [], [])


def calibrate(folder: str, pattern: tuple[int, int] = (9, 6)) -> Calibration:
    """`pattern` = inner corners per row and column of the chessboard."""
    # 3-D corner coordinates on the board plane (z = 0), in units of one square
    board = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    obj_pts, img_pts, used, skipped = [], [], [], []
    size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for path in sorted(Path(folder).glob("*.jpg"), key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)):
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if not found:
            skipped.append(path.name)  # some photos crop part of the board
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # sub-pixel accuracy
        obj_pts.append(board)
        img_pts.append(corners)
        used.append(path.name)
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    return Calibration(K, dist, float(rms), used, skipped)
