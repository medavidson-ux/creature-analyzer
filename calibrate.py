#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "opencv-contrib-python",
#     "numpy",
# ]
# ///
"""
Calibration helper.

Point this at two folders of your own manually-sorted photos:
  - a folder of photos you'd KEEP (good focus, good subject prominence)
  - a folder of photos you'd REJECT (blurry and/or subject too distant)

It prints the raw metric distributions for each group so you can pick
sensible normalization bounds and thresholds in QualityControl(), instead
of relying on the defaults (which were tuned on synthetic test images,
not real underwater photos).

Usage:
    uv run calibrate.py --keep good_photos/ --reject bad_photos/

Note: this imports directly from qc_pipeline.py, so run it from the same
directory (uv's inline script mode doesn't install local sibling files).
"""

import argparse
import os
import statistics as stats

from qc_pipeline import (
    localize_subject, combined_sharpness_score, combined_distance_score
)
import cv2


def collect_metrics(folder):
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    rows = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(exts):
            continue
        img = cv2.imread(os.path.join(folder, f))
        if img is None:
            continue
        roi = localize_subject(img)
        sharp = combined_sharpness_score(img, roi)
        dist = combined_distance_score(img, roi)
        rows.append({**sharp, **dist, "filename": f})
    return rows


def summarize(name, rows, keys):
    print(f"\n--- {name} (n={len(rows)}) ---")
    for k in keys:
        vals = [r[k] for r in rows]
        if not vals:
            continue
        print(f"  {k:16s} min={min(vals):8.3f} "
              f"mean={stats.mean(vals):8.3f} max={max(vals):8.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", required=True, help="Folder of good photos")
    parser.add_argument("--reject", required=True, help="Folder of bad photos")
    args = parser.parse_args()

    keys = ["laplacian", "tenengrad", "fft_ratio",
            "haze", "coverage", "local_contrast", "saturation"]

    keep_rows = collect_metrics(args.keep)
    reject_rows = collect_metrics(args.reject)

    summarize("KEEP", keep_rows, keys)
    summarize("REJECT", reject_rows, keys)

    print("\nSuggested approach: for each metric, set the *_bounds tuple in "
          "QualityControl() so the low end sits near the REJECT group's "
          "typical value and the high end sits near the KEEP group's "
          "typical value, then tune sharpness_min / distance_min on a "
          "held-out set of photos.")


if __name__ == "__main__":
    main()
