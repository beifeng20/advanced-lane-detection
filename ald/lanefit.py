"""Find lane pixels in the bird's-eye mask and fit x = A*y^2 + B*y + C for each line."""
from collections import deque
from dataclasses import dataclass

import numpy as np

# Pixel -> meter scale in the warped image. Lane width is 3.7 m (US highway standard) and spans
# DST x 320..960 = 640 px; the warped image covers roughly 30 m of road ahead in 720 px.
XM_PER_PIX = 3.7 / 640
YM_PER_PIX = 30.0 / 720


@dataclass
class LaneFit:
    left: np.ndarray    # polynomial coefficients [A, B, C] in pixels, x as a function of y
    right: np.ndarray
    left_px: int        # number of pixels used for each fit
    right_px: int


def sliding_window(mask: np.ndarray, n_windows: int = 9, margin: int = 100, min_pix: int = 50):
    """Start from the two peaks of the column histogram of the lower half, then follow each line upwards."""
    h, w = mask.shape
    hist = mask[h // 2:, :].sum(axis=0)
    mid = w // 2
    x_left, x_right = int(np.argmax(hist[:mid])), int(np.argmax(hist[mid:]) + mid)
    ys, xs = mask.nonzero()
    win_h = h // n_windows
    left_idx, right_idx = [], []
    for i in range(n_windows):
        y_lo, y_hi = h - (i + 1) * win_h, h - i * win_h
        in_rows = (ys >= y_lo) & (ys < y_hi)
        l = np.flatnonzero(in_rows & (np.abs(xs - x_left) < margin))
        r = np.flatnonzero(in_rows & (np.abs(xs - x_right) < margin))
        left_idx.append(l)
        right_idx.append(r)
        if len(l) > min_pix:  # re-center the next window on the found pixels
            x_left = int(xs[l].mean())
        if len(r) > min_pix:
            x_right = int(xs[r].mean())
    l, r = np.concatenate(left_idx), np.concatenate(right_idx)
    return xs[l], ys[l], xs[r], ys[r]


def around_previous(mask: np.ndarray, prev: LaneFit, margin: int = 80):
    """Faster search in the next frame: only pixels within `margin` of last frame's curves."""
    ys, xs = mask.nonzero()
    l = np.abs(xs - np.polyval(prev.left, ys)) < margin
    r = np.abs(xs - np.polyval(prev.right, ys)) < margin
    return xs[l], ys[l], xs[r], ys[r]


def fit(lx, ly, rx, ry, min_pixels: int = 200) -> LaneFit | None:
    if len(lx) < min_pixels or len(rx) < min_pixels:
        return None
    return LaneFit(np.polyfit(ly, lx, 2), np.polyfit(ry, rx, 2), len(lx), len(rx))


def curvature_m(coeffs_px: np.ndarray, y_eval_px: float) -> float:
    """Radius of curvature in meters at y_eval: R = (1 + x'^2)^1.5 / |x''|, after scaling the fit to meters."""
    A = coeffs_px[0] * XM_PER_PIX / YM_PER_PIX ** 2
    B = coeffs_px[1] * XM_PER_PIX / YM_PER_PIX
    y = y_eval_px * YM_PER_PIX
    return float((1 + (2 * A * y + B) ** 2) ** 1.5 / max(abs(2 * A), 1e-9))


def offset_m(f: LaneFit, h: int, w: int) -> float:
    """Camera position relative to lane center, meters; positive = car is right of center."""
    lx, rx = np.polyval(f.left, h - 1), np.polyval(f.right, h - 1)
    return float((w / 2 - (lx + rx) / 2) * XM_PER_PIX)


def plausible(f: LaneFit, h: int) -> bool:
    """Sanity check: lane 3-4.5 m wide at the bottom and the two lines roughly parallel."""
    ys = np.array([0, h // 2, h - 1])
    widths = (np.polyval(f.right, ys) - np.polyval(f.left, ys)) * XM_PER_PIX
    return bool(3.0 < widths[-1] < 4.5 and widths.max() - widths.min() < 0.8)


class LaneTracker:
    """Keeps a running average of the last n good fits; falls back to sliding windows when lost."""

    def __init__(self, n: int = 8, max_lost: int = 5):
        self.history: deque[LaneFit] = deque(maxlen=n)
        self.lost = 0
        self.max_lost = max_lost
        self.stats = {"frames": 0, "sliding_window": 0, "rejected": 0}

    @property
    def best(self) -> LaneFit | None:
        if not self.history:
            return None
        return LaneFit(np.mean([f.left for f in self.history], axis=0),
                       np.mean([f.right for f in self.history], axis=0), 0, 0)

    def update(self, mask: np.ndarray) -> LaneFit | None:
        self.stats["frames"] += 1
        h = mask.shape[0]
        new = None
        if self.best is not None and self.lost <= self.max_lost:
            new = fit(*around_previous(mask, self.best))
        if new is None or not plausible(new, h):
            self.stats["sliding_window"] += 1
            new = fit(*sliding_window(mask))
        if new is not None and plausible(new, h):
            self.history.append(new)
            self.lost = 0
        else:
            self.stats["rejected"] += 1
            self.lost += 1
        return self.best
