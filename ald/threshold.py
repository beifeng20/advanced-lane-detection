"""Binary lane-pixel mask: color (HLS saturation + lightness) OR horizontal gradient."""
import cv2
import numpy as np


def lane_mask(bgr: np.ndarray, s_thresh=(120, 255), l_thresh=(200, 255), sx_thresh=(25, 120)) -> np.ndarray:
    """Return a uint8 mask (1 = lane pixel candidate).

    - S channel: yellow and white lines are saturated/bright, survives shadows better than grayscale
    - L channel: catches white lines with low saturation
    - Sobel x on L: lane lines are mostly vertical, so the x-gradient is strong on their edges
    """
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    l_ch, s_ch = hls[:, :, 1], hls[:, :, 2]

    sobelx = np.absolute(cv2.Sobel(l_ch, cv2.CV_64F, 1, 0, ksize=3))
    sobelx = np.uint8(255 * sobelx / max(sobelx.max(), 1e-6))

    s_bin = (s_ch >= s_thresh[0]) & (s_ch <= s_thresh[1]) & (l_ch > 60)  # skip dark saturated shadows
    l_bin = (l_ch >= l_thresh[0]) & (l_ch <= l_thresh[1])
    g_bin = (sobelx >= sx_thresh[0]) & (sobelx <= sx_thresh[1])
    return (s_bin | l_bin | g_bin).astype(np.uint8)
