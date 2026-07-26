#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "opencv-contrib-python",
#     "numpy",
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

Usage (uv resolves and installs dependencies automatically, no venv setup):
    uv run qc_pipeline.py /path/to/photo_folder --out results.csv

    # or, after `chmod +x qc_pipeline.py`:
    ./qc_pipeline.py /path/to/photo_folder --out results.csv
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------

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

def laplacian_sharpness(gray_roi):
    """Classic variance-of-Laplacian focus measure."""
    return cv2.Laplacian(gray_roi, cv2.CV_64F).var()


def tenengrad_sharpness(gray_roi):
    """Gradient-magnitude based focus measure; more robust to noise
    than Laplacian on grainy strobe-lit water."""
    gx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
    fm = gx ** 2 + gy ** 2
    return np.mean(fm)


def fft_sharpness(gray_roi):
    """
    Ratio of high-frequency to total spectral energy. More robust than
    spatial-domain metrics to backscatter speckle, which tends to be
    uniformly distributed rather than concentrated in true edge structure.
    """
    f = np.fft.fft2(gray_roi)
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

    return {"laplacian": lap, "tenengrad": ten, "fft_ratio": fft}


# ---------------------------------------------------------------------------
# 4. Distance / prominence metrics
# ---------------------------------------------------------------------------

def dark_channel_prior(img_bgr, patch_size=15):
    """
    Dark channel prior (He et al.) -- estimates haze/water density between
    camera and scene. Higher dark-channel values in a region generally mean
    more water column (more distance) between the lens and that region.
    """
    b, g, r = cv2.split(img_bgr.astype(np.float32) / 255.0)
    min_channel = np.minimum(np.minimum(b, g), r)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)
    return dark  # 0 (clear/close) .. 1 (hazy/far)


def contrast_saturation_decay(img_bgr, roi):
    """
    Water attenuates color and contrast with distance -- red channel first,
    then green. A distant subject looks flatter and cooler even when it's
    technically in focus. Returns local contrast (stddev of luminance) and
    mean saturation within the subject ROI as distance proxies.
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
    dark = dark_channel_prior(img_bgr)
    x, y, w, h = roi
    subject_haze = float(dark[y:y + h, x:x + w].mean())

    decay = contrast_saturation_decay(img_bgr, roi)
    coverage = frame_coverage(img_bgr, roi)

    return {
        "haze": subject_haze,
        "coverage": coverage,
        **decay,
    }


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


def normalize(value, low, high):
    """Clamp-and-scale a raw metric into 0-1 given expected low/high bounds."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


class QualityControl:
    """
    Thresholds below are reasonable starting points, NOT ground truth.
    Calibrate them against ~100-200 of your own manually sorted photos --
    see calibrate.py for a helper script that suggests thresholds from
    a labeled folder.
    """

    def __init__(self,
                 sharpness_min=0.35,
                 distance_min=0.35,
                 lap_bounds=(0, 800),
                 ten_bounds=(0, 4000),
                 fft_bounds=(0.15, 0.6),
                 haze_bounds=(0.05, 0.6),   # inverted: low haze = good
                 coverage_bounds=(0.03, 0.4),
                 contrast_bounds=(0.05, 0.35),
                 saturation_bounds=(0.1, 0.6)):
        self.sharpness_min = sharpness_min
        self.distance_min = distance_min
        self.lap_bounds = lap_bounds
        self.ten_bounds = ten_bounds
        self.fft_bounds = fft_bounds
        self.haze_bounds = haze_bounds
        self.coverage_bounds = coverage_bounds
        self.contrast_bounds = contrast_bounds
        self.saturation_bounds = saturation_bounds

    def score_sharpness(self, raw):
        lap_n = normalize(raw["laplacian"], *self.lap_bounds)
        ten_n = normalize(raw["tenengrad"], *self.ten_bounds)
        fft_n = normalize(raw["fft_ratio"], *self.fft_bounds)
        # Equal-weighted average; FFT metric gets slight extra weight since
        # it's most robust to backscatter noise.
        return 0.3 * lap_n + 0.3 * ten_n + 0.4 * fft_n

    def score_distance(self, raw):
        haze_n = 1.0 - normalize(raw["haze"], *self.haze_bounds)
        cov_n = normalize(raw["coverage"], *self.coverage_bounds)
        contrast_n = normalize(raw["local_contrast"], *self.contrast_bounds)
        sat_n = normalize(raw["saturation"], *self.saturation_bounds)
        return 0.35 * haze_n + 0.35 * cov_n + 0.15 * contrast_n + 0.15 * sat_n

    def evaluate(self, path):
        img = cv2.imread(path)
        if img is None:
            return QCResult(os.path.basename(path), False,
                             "unreadable file", 0.0, 0.0, {})

        roi = localize_subject(img)
        sharp_raw = combined_sharpness_score(img, roi)
        dist_raw = combined_distance_score(img, roi)

        sharp_score = self.score_sharpness(sharp_raw)
        dist_score = self.score_distance(dist_raw)

        reasons = []
        if sharp_score < self.sharpness_min:
            reasons.append(f"subject out of focus (score {sharp_score:.2f})")
        if dist_score < self.distance_min:
            reasons.append(f"subject too distant/small (score {dist_score:.2f})")

        approved = len(reasons) == 0
        reason = "approved" if approved else "; ".join(reasons)

        raw_metrics = {**sharp_raw, **dist_raw, "roi": roi}
        return QCResult(os.path.basename(path), approved, reason,
                         round(sharp_score, 3), round(dist_score, 3),
                         raw_metrics)


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Underwater photo QC filter")
    parser.add_argument("folder", help="Folder of images to evaluate")
    parser.add_argument("--out", default="qc_results.csv",
                         help="Output CSV path")
    parser.add_argument("--sharpness-min", type=float, default=0.35)
    parser.add_argument("--distance-min", type=float, default=0.35)
    args = parser.parse_args()

    qc = QualityControl(sharpness_min=args.sharpness_min,
                         distance_min=args.distance_min)

    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    files = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(exts))

    if not files:
        print(f"No image files found in {args.folder}")
        return

    results = []
    for f in files:
        path = os.path.join(args.folder, f)
        r = qc.evaluate(path)
        results.append(r)
        status = "APPROVE" if r.approved else "REJECT "
        print(f"[{status}] {r.filename:40s} "
              f"sharp={r.sharpness_score:.2f} dist={r.distance_score:.2f}  {r.reason}")

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "approved", "reason",
                          "sharpness_score", "distance_score"])
        for r in results:
            writer.writerow([r.filename, r.approved, r.reason,
                              r.sharpness_score, r.distance_score])

    n_approved = sum(r.approved for r in results)
    print(f"\n{n_approved}/{len(results)} approved. Results written to {args.out}")


if __name__ == "__main__":
    main()
