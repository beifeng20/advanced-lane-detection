"""Figures for the README: calibration, pipeline stages, curvature over time, demo GIF.

    python tools/make_figures.py      (after run.py)
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ald.calibration import Calibration  # noqa: E402
from ald.lanefit import fit, sliding_window  # noqa: E402
from ald.pipeline import LanePipeline  # noqa: E402
from ald.perspective import SRC, BirdsEye  # noqa: E402
from ald.threshold import lane_mask  # noqa: E402

DOCS = Path("docs")
DOCS.mkdir(exist_ok=True)
calib = Calibration.load("data/calibration.npz")
view = BirdsEye()


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# 1. calibration: chessboard before / after
board = cv2.imread("data/camera_cal/calibration1.jpg")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for a, im, t in zip(ax, [board, calib.undistort(board)], ["original (barrel distortion)", "undistorted"]):
    a.imshow(rgb(im))
    a.set_title(t)
    a.axis("off")
fig.tight_layout()
fig.savefig(DOCS / "calibration.jpg", dpi=90)
plt.close(fig)

# 2. pipeline stages on one test image
img = cv2.imread("data/test_images/test2.jpg")
und = calib.undistort(img)
mask = lane_mask(und)
top = view.warp(mask)
lx, ly, rx, ry = sliding_window(top)
f = fit(lx, ly, rx, ry)
vis = np.dstack([top * 255] * 3).astype(np.uint8)
vis[ly, lx] = (255, 60, 60)
vis[ry, rx] = (60, 60, 255)
ys = np.arange(top.shape[0])
for coeffs in (f.left, f.right):
    cv2.polylines(vis, np.int32([np.column_stack([np.polyval(coeffs, ys), ys])]), False, (255, 255, 0), 6)
src_vis = rgb(und).copy()
cv2.polylines(src_vis, np.int32([SRC]), True, (255, 0, 0), 3)
fig, ax = plt.subplots(2, 2, figsize=(11, 6.6))
panels = [(src_vis, "1. undistorted + perspective source area"), (mask * 255, "2. lane mask (HLS color OR Sobel x)"),
          (vis, "3. bird's-eye view: window search + 2nd-order fit"),
          (rgb(LanePipeline(calib)(img)), "4. lane drawn back on the road")]
for a, (im, t) in zip(ax.flat, panels):
    a.imshow(im, cmap="gray")
    a.set_title(t, fontsize=10)
    a.axis("off")
fig.tight_layout()
fig.savefig(DOCS / "stages.jpg", dpi=90)
plt.close(fig)

# 3. curvature and offset over time
d = np.genfromtxt("output/project_video.csv", delimiter=",", names=True)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
a1.plot(d["time_s"], np.clip(d["radius_m"], 0, 5000), lw=1)
a1.set_ylabel("radius, m (clipped at 5 km)")
a1.set_title("Measured radius of curvature and lane offset, project video")
a2.plot(d["time_s"], d["offset_m"], lw=1, color="tab:orange")
a2.axhline(0, color="gray", lw=0.8)
a2.set_ylabel("offset, m (+ = right)")
a2.set_xlabel("time, s")
fig.tight_layout()
fig.savefig(DOCS / "curvature.png", dpi=90)
plt.close(fig)

# 4. demo GIF: every 8th frame of the first 25 s, 480 px wide
cap = cv2.VideoCapture("output/project_video.mp4")
frames = []
for i in range(0, 625, 8):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    if ok:
        frames.append(Image.fromarray(rgb(cv2.resize(fr, (480, 270)))).quantize(64))
frames[0].save(DOCS / "demo.gif", save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=True)
print("figures written to docs/")
