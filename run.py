"""Process a video and save the annotated result plus per-frame measurements.

    python run.py --input data/project_video.mp4 --output output/project_video.mp4
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from ald.calibration import Calibration, calibrate
from ald.pipeline import LanePipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--calibration", default="data/calibration.npz")
    args = ap.parse_args()

    if Path(args.calibration).exists():
        calib = Calibration.load(args.calibration)
    else:
        calib = calibrate("data/camera_cal")
        calib.save(args.calibration)
        print(f"calibrated on {len(calib.used)} images, RMS reprojection error {calib.rms:.3f} px")

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    pipe = LanePipeline(calib)
    rows, t_total, i = [], 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        out = pipe(frame)
        t_total += time.perf_counter() - t0
        writer.write(out)
        rows.append({"frame": i, "time_s": round(i / fps, 2),
                     "radius_m": round(pipe.last.get("radius_m", float("nan")), 1),
                     "offset_m": round(pipe.last.get("offset_m", float("nan")), 3)})
        i += 1
    writer.release()

    with open(out_path.with_suffix(".csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    s = pipe.tracker.stats
    summary = {"frames": s["frames"], "frames_without_plausible_new_fit": s["rejected"],
               "sliding_window_searches": s["sliding_window"],
               "ms_per_frame": round(1000 * t_total / max(i, 1), 1)}
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
