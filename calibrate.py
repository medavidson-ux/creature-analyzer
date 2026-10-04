#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "opencv-contrib-python",
#     "numpy",
#     "pillow",
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

from qc_pipeline import IMAGE_EXTS, METRIC_KEYS, load_image, measure


def collect_metrics(folder):
    rows = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        img = load_image(os.path.join(folder, f))
        if img is None:
            continue
        rows.append({**measure(img), "filename": f})
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

    keep_rows = collect_metrics(args.keep)
    reject_rows = collect_metrics(args.reject)

    summarize("KEEP", keep_rows, METRIC_KEYS)
    summarize("REJECT", reject_rows, METRIC_KEYS)

    print("\nSuggested approach: for each metric, set the *_bounds tuple in "
          "QualityControl() so the low end sits near the REJECT group's "
          "typical value and the high end sits near the KEEP group's "
          "typical value, then tune sharpness_min / distance_min on a "
          "held-out set of photos.")


if __name__ == "__main__":
    main()
