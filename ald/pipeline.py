"""Frame pipeline: undistort -> threshold -> bird's-eye -> fit -> draw back on the road."""
import cv2
import numpy as np

from .calibration import Calibration
from .lanefit import LaneTracker, curvature_m, offset_m
from .perspective import BirdsEye
from .threshold import lane_mask


class LanePipeline:
    def __init__(self, calib: Calibration):
        self.calib = calib
        self.view = BirdsEye()
        self.tracker = LaneTracker()
        self.last = {}

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        und = self.calib.undistort(bgr)
        top = self.view.warp(lane_mask(und))
        f = self.tracker.update(top)
        out = und.copy()
        if f is None:
            cv2.putText(out, "lane lost", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            return out

        h, w = top.shape
        ys = np.linspace(0, h - 1, h)
        lx, rx = np.polyval(f.left, ys), np.polyval(f.right, ys)
        overlay = np.zeros_like(und)
        pts = np.hstack([np.array([np.column_stack([lx, ys])]), np.array([np.flipud(np.column_stack([rx, ys]))])])
        cv2.fillPoly(overlay, np.int32(pts), (0, 200, 0))
        cv2.polylines(overlay, np.int32([np.column_stack([lx, ys])]), False, (0, 0, 255), 25)
        cv2.polylines(overlay, np.int32([np.column_stack([rx, ys])]), False, (255, 0, 0), 25)
        out = cv2.addWeighted(out, 1.0, self.view.unwarp(overlay), 0.35, 0)

        radius = (curvature_m(f.left, h - 1) + curvature_m(f.right, h - 1)) / 2
        offset = offset_m(f, h, w)
        self.last = {"radius_m": radius, "offset_m": offset}
        label = "straight (> 3 km)" if radius > 3000 else f"{radius:,.0f} m"
        cv2.putText(out, f"Radius of curvature: {label}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        side = "right" if offset > 0 else "left"
        cv2.putText(out, f"Vehicle {abs(offset):.2f} m {side} of center", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        return out
