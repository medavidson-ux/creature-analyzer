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
Underwater Photo Quality Control Pipeline
==========================================

Evaluates underwater photos for:
  1. Subject sharpness (is the main subject in focus?)
  2. Subject prominence/distance (is the subject close/large enough to be
     visually appealing, based on contrast decay, haze, and frame coverage?)

Combines both into an approve/reject decision with a human-readable reason.

Photos shot in a burst of the same subject are grouped (by EXIF capture time
plus visual similarity) and ranked within the group, so the best frame of
each burst is selected and the rest are marked as redundant.

Usage (uv resolves and installs dependencies automatically, no venv setup):
    uv run qc_pipeline.py /path/to/photo_folder --out results.csv

    # or, after `chmod +x qc_pipeline.py`:
    ./qc_pipeline.py /path/to/photo_folder --out results.csv
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Loading / preprocessing
# ---------------------------------------------------------------------------

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# Every metric is computed per pixel, so the same photo scores differently at
# 1080px vs 6000px. Downscale to a fixed working size so bounds transfer
# between cameras. Bounds below were tuned at <=1600px.
WORK_MAX_SIDE = 1600


def load_image(path):
    """cv2.imread can't open non-ASCII paths on Windows; decoding from a
    byte buffer works everywhere. Returns None if unreadable."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def resize_to_working(img_bgr, max_side=WORK_MAX_SIDE):
    h, w = img_bgr.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return img_bgr
    return cv2.resize(img_bgr, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA)


def despeckle(img_bgr, strength=3):
    """
    Light median-filter pass to suppress backscatter (floating particulates
    lit by strobe) before sharpness is measured, so particulates don't
    inflate the sharpness score on a genuinely blurry photo.
    """
    k = strength if strength % 2 == 1 else strength + 1
    return cv2.medianBlur(img_bgr, k)


# ---------------------------------------------------------------------------
# 2. Subject localization (saliency-based, no training data required)
# ---------------------------------------------------------------------------

def _contour_score(contour, img_w, img_h):
    """
    Scores a candidate contour by how 'subject-like' it is: compact
    (not an elongated band -- those are usually water-column gradients or
    light shafts, not discrete subjects) and centrally located
    (photographers usually frame the subject near-center).
    Returns None if the contour should be disqualified outright.
    """
    x, y, bw, bh = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    frame_area = img_w * img_h

    area_frac = area / frame_area
    if area_frac < 0.005 or area_frac > 0.5:
        return None  # too tiny to trust, or basically the whole frame

    aspect = bw / float(bh + 1e-6)
    # Disqualify very elongated shapes spanning near-full width/height --
    # these are almost always banding artifacts, not subjects.
    if aspect > 4.0 or aspect < 0.25:
        return None
    if bw > 0.9 * img_w or bh > 0.9 * img_h:
        return None

    cx, cy = x + bw / 2.0, y + bh / 2.0
    dist_from_center = np.hypot(cx - img_w / 2.0, cy - img_h / 2.0)
    max_dist = np.hypot(img_w / 2.0, img_h / 2.0)
    centrality = 1.0 - (dist_from_center / max_dist)  # 1 = dead center

    compactness = min(bw, bh) / float(max(bw, bh))  # 1 = square-ish

    # Weighted score: centrality and compactness matter more than raw size
    return 0.45 * centrality + 0.35 * compactness + 0.20 * min(area_frac / 0.15, 1.0)


def localize_subject(img_bgr):
    """
    Returns (x, y, w, h) bounding box of the most likely subject region,
    using OpenCV's spectral-residual saliency detector as a first pass.
    Candidate blobs are filtered/ranked by compactness and centrality
    rather than raw area, since underwater saliency maps often highlight
    elongated water-column gradients or backscatter bands that are not
    the actual subject. Falls back to a center-weighted box if no
    candidate is trustworthy.
    """
    h, w = img_bgr.shape[:2]

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, sal_map = saliency.computeSaliency(img_bgr)

    if success:
        sal_map = (sal_map * 255).astype("uint8")
        sal_map = cv2.GaussianBlur(sal_map, (9, 9), 0)

        best_contour = None
        best_score = -1.0
        # Try a few thresholds; higher percentiles isolate tighter peaks,
        # which helps when the subject is small relative to the frame.
        for pct in (95, 90, 85, 75):
            thresh_val = np.percentile(sal_map, pct)
            _, mask = cv2.threshold(sal_map, thresh_val, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                     np.ones((15, 15), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                s = _contour_score(c, w, h)
                if s is not None and s > best_score:
                    best_score = s
                    best_contour = c

        if best_contour is not None:
            x, y, bw, bh = cv2.boundingRect(best_contour)
            return (x, y, bw, bh)

    # Fallback: center-weighted box (photographers usually frame centrally)
    bw, bh = int(w * 0.5), int(h * 0.5)
    x, y = (w - bw) // 2, (h - bh) // 2
    return (x, y, bw, bh)


# ---------------------------------------------------------------------------
# 3. Sharpness metrics
# ---------------------------------------------------------------------------

# Raw Laplacian/Tenengrad energy scales with contrast squared, and water
# removes contrast with distance -- so unnormalized, a sharp distant subject
# is indistinguishable from a defocused near one. Both are divided by the
# ROI's luminance variance to make them contrast-invariant. The floor keeps
# near-flat, noise-only ROIs from being blown up into "very sharp".
CONTRAST_FLOOR_VAR = (0.03 * 255) ** 2


def laplacian_sharpness(gray_roi):
    """Variance-of-Laplacian, normalized by ROI luminance variance."""
    g = gray_roi.astype(np.float64)
    return cv2.Laplacian(g, cv2.CV_64F).var() / (g.var() + CONTRAST_FLOOR_VAR)


def tenengrad_sharpness(gray_roi):
    """Mean gradient energy, normalized by ROI luminance variance."""
    g = gray_roi.astype(np.float64)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    fm = gx ** 2 + gy ** 2
    return np.mean(fm) / (g.var() + CONTRAST_FLOOR_VAR)


def fft_sharpness(gray_roi):
    """
    Ratio of high-frequency to total spectral magnitude. The mean is removed
    first so the DC term doesn't dominate the denominator; that makes the
    ratio contrast- and brightness-invariant. Caveat: sensor noise and
    backscatter are broadband, so this ratio *rises* with noise -- it is the
    least noise-robust of the three measures and is weighted accordingly.
    """
    g = gray_roi.astype(np.float64)
    f = np.fft.fft2(g - g.mean())
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = gray_roi.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8  # low-frequency cutoff radius

    mask = np.ones((h, w), dtype=bool)
    y, x = np.ogrid[:h, :w]
    mask[(y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2] = False

    high_freq_energy = magnitude[mask].sum()
    total_energy = magnitude.sum() + 1e-8
    return high_freq_energy / total_energy


def combined_sharpness_score(img_bgr, roi):
    """
    Combines three sharpness measures on the subject ROI into one
    normalized score. Raw values are scene-dependent, so this returns
    the raw metrics too -- calibrate thresholds against your own labeled
    photos rather than trusting fixed cutoffs out of the box.
    """
    x, y, w, h = roi
    crop = img_bgr[y:y + h, x:x + w]
    clean = despeckle(crop)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)

    lap = laplacian_sharpness(gray)
    ten = tenengrad_sharpness(gray)
    fft = fft_sharpness(gray)

    return {"laplacian": float(lap), "tenengrad": float(ten),
            "fft_ratio": float(fft)}


# ---------------------------------------------------------------------------
# 4. Distance / prominence metrics
# ---------------------------------------------------------------------------

def dark_channel_prior(img_bgr, patch_size=15):
    """
    Underwater dark channel prior (UDCP, Drews et al.) -- estimates water
    density between camera and scene. Higher values mean more water column
    (more distance) between the lens and that pixel.

    The original land-haze DCP (He et al.) takes min over B, G *and R*. Water
    absorbs red with distance, so for distant water R -> 0 and the classic
    dark channel reads as "clear/close" -- exactly backwards. UDCP ignores R.
    """
    b, g, _ = cv2.split(img_bgr.astype(np.float32) / 255.0)
    min_channel = np.minimum(b, g)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)
    return dark  # 0 (clear/close) .. 1 (hazy/far)


def contrast_saturation_decay(img_bgr, roi):
    """
    Water attenuates color and contrast with distance -- red channel first,
    then green. A distant subject looks flatter and cooler even when it's
    technically in focus. Returns local contrast (stddev of luminance) and
    red-channel retention within the subject ROI as distance proxies.

    Saturation is reported for calibration but NOT scored: HSV saturation is
    (max-min)/max, so blue water with its red absorbed is close to 1.0 --
    distant water reads as *more* saturated, not less.
    """
    x, y, w, h = roi
    crop = img_bgr[y:y + h, x:x + w]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean() / 255.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    local_contrast = gray.std() / 255.0

    # Red-channel retention relative to blue/green -- red drops off
    # fastest with distance underwater.
    b, g, r = cv2.split(crop.astype(np.float32))
    red_ratio = r.mean() / (b.mean() + g.mean() + 1e-6)

    return {
        "saturation": float(saturation),
        "local_contrast": float(local_contrast),
        "red_ratio": float(red_ratio),
    }


def frame_coverage(img_bgr, roi):
    """Fraction of the frame occupied by the subject bounding box --
    a simple, direct proxy for 'is the subject big enough to matter.'"""
    h, w = img_bgr.shape[:2]
    _, _, bw, bh = roi
    return (bw * bh) / float(w * h)


def combined_distance_score(img_bgr, roi):
    # Full-frame then cropped so the erosion window sees context at ROI edges.
    dark = dark_channel_prior(img_bgr)
    x, y, w, h = roi
    # Divide by the brightest dark-channel value in the frame (the "water
    # light" estimate in DCP) so haze is relative to the open water, not
    # absolute brightness -- otherwise an underexposed frame reads as
    # clearer/closer and can win its burst.
    water_light = float(np.percentile(dark, 99.9)) + 1e-3
    subject_haze = float(dark[y:y + h, x:x + w].mean()) / water_light

    decay = contrast_saturation_decay(img_bgr, roi)
    coverage = frame_coverage(img_bgr, roi)

    return {
        "haze": subject_haze,
        "coverage": coverage,
        **decay,
    }


def measure(img_bgr):
    """Single source of truth for the measurement pipeline (also used by
    calibrate.py): resize -> localize -> raw sharpness + distance metrics."""
    img = resize_to_working(img_bgr)
    roi = localize_subject(img)
    return {**combined_sharpness_score(img, roi),
            **combined_distance_score(img, roi),
            "roi": roi}


# ---------------------------------------------------------------------------
# 5. Combined decision
# ---------------------------------------------------------------------------

@dataclass
class QCResult:
    filename: str
    approved: bool
    reason: str
    sharpness_score: float      # 0-1 normalized
    distance_score: float       # 0-1 normalized (1 = subject close/prominent)
    raw_metrics: dict
    group: int = 0              # burst group id (photos of the same moment)
    group_rank: int = 1         # 1 = best in its group
    selected: bool = False      # best of group AND passes thresholds
    capture_time: datetime = None
    signature: np.ndarray = field(default=None, repr=False)


def normalize(value, low, high):
    """Clamp-and-scale a raw metric into 0-1 given expected low/high bounds."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


class QualityControl:
    """
    Thresholds below are reasonable starting points, NOT ground truth.
    Calibrate them against ~100-200 of your own manually sorted photos --
    see calibrate.py for a helper script that prints metric distributions
    for a labeled folder.
    """

    def __init__(self,
                 sharpness_min=0.35,
                 distance_min=0.35,
                 lap_bounds=(0, 0.1),
                 ten_bounds=(0, 1.2),
                 fft_bounds=(0.15, 0.7),
                 haze_bounds=(0.5, 0.95),   # inverted: low haze = good
                 coverage_bounds=(0.03, 0.4),
                 contrast_bounds=(0.05, 0.35),
                 red_bounds=(0.1, 1.0)):
        self.sharpness_min = sharpness_min
        self.distance_min = distance_min
        self.lap_bounds = lap_bounds
        self.ten_bounds = ten_bounds
        self.fft_bounds = fft_bounds
        self.haze_bounds = haze_bounds
        self.coverage_bounds = coverage_bounds
        self.contrast_bounds = contrast_bounds
        self.red_bounds = red_bounds

    def score_sharpness(self, raw):
        lap_n = normalize(raw["laplacian"], *self.lap_bounds)
        ten_n = normalize(raw["tenengrad"], *self.ten_bounds)
        fft_n = normalize(raw["fft_ratio"], *self.fft_bounds)
        # FFT ratio gets the least weight: it rises with broadband noise.
        return 0.4 * lap_n + 0.4 * ten_n + 0.2 * fft_n

    def score_distance(self, raw):
        haze_n = 1.0 - normalize(raw["haze"], *self.haze_bounds)
        cov_n = normalize(raw["coverage"], *self.coverage_bounds)
        contrast_n = normalize(raw["local_contrast"], *self.contrast_bounds)
        red_n = normalize(raw["red_ratio"], *self.red_bounds)
        return 0.35 * haze_n + 0.35 * cov_n + 0.15 * contrast_n + 0.15 * red_n

    def evaluate(self, path):
        img = load_image(path)
        if img is None:
            return QCResult(os.path.basename(path), False,
                             "unreadable file", 0.0, 0.0, {})

        work = resize_to_working(img)
        raw_metrics = measure(work)
        # Round once so the printed score and the reason string agree.
        sharp_score = round(self.score_sharpness(raw_metrics), 3)
        dist_score = round(self.score_distance(raw_metrics), 3)

        reasons = []
        if sharp_score < self.sharpness_min:
            reasons.append(f"subject out of focus (score {sharp_score:.2f})")
        if dist_score < self.distance_min:
            reasons.append(f"subject too distant/small (score {dist_score:.2f})")

        approved = len(reasons) == 0
        reason = "approved" if approved else "; ".join(reasons)

        return QCResult(os.path.basename(path), approved, reason,
                         sharp_score, dist_score, raw_metrics,
                         capture_time=capture_time(path),
                         signature=scene_signature(work))


# ---------------------------------------------------------------------------
# 6. Burst grouping / best-of-group selection
# ---------------------------------------------------------------------------

def capture_time(path):
    """EXIF DateTimeOriginal (+ sub-seconds, which bursts need), falling back
    to IFD0 DateTime. Returns None if the file has no usable timestamp."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
    except Exception:
        return None
    sub = exif.get_ifd(0x8769)  # Exif sub-IFD
    stamp = sub.get(0x9003) or exif.get(0x0132)
    if not stamp:
        return None
    try:
        t = datetime.strptime(str(stamp).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    frac = str(sub.get(0x9291) or "").strip()
    if frac.isdigit():
        t = t.replace(microsecond=int(frac.ljust(6, "0")[:6]))
    return t


def scene_signature(img_bgr):
    """Tiny blurred thumbnail, normalized per channel to zero mean / unit
    variance, as a unit vector. The dot product of two signatures is their
    correlation: ~1.0 across exposure changes (strobe misfires), focus
    changes and small camera moves within a burst; drops when the framing or
    subject changes. (A colour histogram doesn't work here: blue water
    dominates it, and exposure shifts wreck it.)"""
    t = cv2.resize(img_bgr, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    t = cv2.GaussianBlur(t, (3, 3), 0)
    t -= t.mean(axis=(0, 1))
    t /= t.std(axis=(0, 1)) + 1e-6
    v = t.flatten()
    return v / (np.linalg.norm(v) + 1e-6)


def group_photos(results, max_gap_s=5.0, min_similarity=0.7):
    """
    Assigns `group` ids. Photos are sorted by capture time and chained: a
    photo joins the previous photo's group if it was taken within
    `max_gap_s` seconds of it AND looks similar (signature correlation >=
    `min_similarity`; 0 disables the check). The default errs toward
    splitting: a false split just keeps one extra frame, while a false merge
    of two different subjects throws a good photo away. Comparing to the previous frame
    rather than the group's first lets slow drift through a burst stay
    grouped. Photos with no capture time each get their own group.
    """
    timed = sorted((r for r in results if r.capture_time is not None),
                   key=lambda r: (r.capture_time, r.filename))
    untimed = [r for r in results if r.capture_time is None]

    gid, prev = 0, None
    for r in timed:
        same = (prev is not None
                and (r.capture_time - prev.capture_time).total_seconds() <= max_gap_s
                and (min_similarity <= 0 or _similar(prev, r, min_similarity)))
        if not same:
            gid += 1
        r.group = gid
        prev = r
    for r in untimed:
        gid += 1
        r.group = gid
    return results


def _similar(a, b, min_similarity):
    if a.signature is None or b.signature is None:
        return True  # can't tell (e.g. unreadable file); trust the clock
    return float(a.signature @ b.signature) >= min_similarity


# Within a burst the water, light and subject are the same, so differences
# are mostly focus/motion blur and framing. Focus is weighted higher because
# it is the usual reason one frame of a burst beats another.
GROUP_RANK_WEIGHTS = (0.65, 0.35)  # (sharpness, distance)


def select_best(results, keep_per_group=1):
    """Ranks photos within each group and marks the top `keep_per_group` as
    selected -- but only if they also pass the absolute QC thresholds, so a
    burst where every frame is bad selects nothing."""
    ws, wd = GROUP_RANK_WEIGHTS
    groups = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)
    for members in groups.values():
        members.sort(key=lambda r: (not r.reason == "unreadable file",
                                    ws * r.sharpness_score + wd * r.distance_score),
                     reverse=True)
        for rank, r in enumerate(members, start=1):
            r.group_rank = rank
            r.selected = r.approved and rank <= keep_per_group
    return results


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

METRIC_KEYS = ["laplacian", "tenengrad", "fft_ratio", "haze", "coverage",
               "local_contrast", "red_ratio", "saturation"]


def main():
    parser = argparse.ArgumentParser(description="Underwater photo QC filter")
    parser.add_argument("folder", help="Folder of images to evaluate")
    parser.add_argument("--out", default="qc_results.csv",
                         help="Output CSV path")
    parser.add_argument("--sharpness-min", type=float, default=0.35)
    parser.add_argument("--distance-min", type=float, default=0.35)
    parser.add_argument("--no-groups", action="store_true",
                         help="Judge every photo independently (no burst grouping)")
    parser.add_argument("--max-gap", type=float, default=5.0,
                         help="Max seconds between consecutive shots in one group")
    parser.add_argument("--min-similarity", type=float, default=0.7,
                         help="Thumbnail correlation needed to stay in a group "
                              "(0 = group by time only)")
    parser.add_argument("--keep-per-group", type=int, default=1,
                         help="How many top frames to select from each group")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        parser.error(f"not a directory: {args.folder}")

    qc = QualityControl(sharpness_min=args.sharpness_min,
                         distance_min=args.distance_min)

    files = sorted(f for f in os.listdir(args.folder)
                   if f.lower().endswith(IMAGE_EXTS))

    if not files:
        print(f"No image files found in {args.folder}", file=sys.stderr)
        sys.exit(1)

    results = [qc.evaluate(os.path.join(args.folder, f)) for f in files]

    if args.no_groups:
        for gid, r in enumerate(results, start=1):
            r.group = gid
    else:
        group_photos(results, args.max_gap, args.min_similarity)
    select_best(results, args.keep_per_group)

    results.sort(key=lambda r: (r.group, r.group_rank))
    prev_group = None
    for r in results:
        if r.group != prev_group and not args.no_groups:
            size = sum(x.group == r.group for x in results)
            when = r.capture_time.strftime("%H:%M:%S") if r.capture_time else "no timestamp"
            print(f"\n-- group {r.group} ({size} photo{'s' if size > 1 else ''}, {when})")
            prev_group = r.group
        if r.selected:
            status = "SELECT "
        elif r.approved:
            status = f"DUP #{r.group_rank}"  # passes, but a better frame exists
        else:
            status = "REJECT "
        print(f"[{status}] {r.filename:40s} "
              f"sharp={r.sharpness_score:.2f} dist={r.distance_score:.2f}  {r.reason}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "selected", "group", "group_rank",
                          "capture_time", "approved", "reason",
                          "sharpness_score", "distance_score",
                          *METRIC_KEYS, "roi_x", "roi_y", "roi_w", "roi_h"])
        for r in results:
            m = r.raw_metrics
            writer.writerow([r.filename, r.selected, r.group, r.group_rank,
                              r.capture_time.isoformat() if r.capture_time else "",
                              r.approved, r.reason,
                              r.sharpness_score, r.distance_score,
                              *(round(m[k], 4) if k in m else "" for k in METRIC_KEYS),
                              *(m.get("roi") or ("", "", "", ""))])

    n_groups = len({r.group for r in results})
    n_selected = sum(r.selected for r in results)
    n_approved = sum(r.approved for r in results)
    print(f"\n{n_selected} selected from {n_groups} group(s); "
          f"{n_approved}/{len(results)} passed QC. Results written to {args.out}")


if __name__ == "__main__":
    main()
