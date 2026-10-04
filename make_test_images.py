#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "opencv-contrib-python",
#     "numpy",
# ]
# ///
"""Generates synthetic test images approximating three underwater scenarios:
   1. sharp_close.jpg  - crisp, high-contrast, large subject (should APPROVE)
   2. blurry_close.jpg - same framing, heavily blurred (should REJECT: focus)
   3. sharp_far.jpg     - crisp but tiny/hazy/low-contrast (should REJECT: distance)

Usage:
    uv run make_test_images.py [out_dir]
"""
import os
import sys

import cv2
import numpy as np


def water_background(h, w, rng, haze=0.15):
    """Blue-green gradient background simulating open water + backscatter."""
    y = np.linspace(0, 1, h).reshape(-1, 1)
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :, 0] = 120 + 40 * y[:, 0][:, None].repeat(w, axis=1)   # B
    base[:, :, 1] = 90 + 30 * y[:, 0][:, None].repeat(w, axis=1)    # G
    base[:, :, 2] = 20 + 10 * y[:, 0][:, None].repeat(w, axis=1)    # R (attenuated)
    # backscatter speckle
    speckle = rng.normal(0, 15 * haze, (h, w, 3))
    img = np.clip(base + speckle, 0, 255).astype(np.uint8)
    return img


def draw_fish(img, cx, cy, size, color):
    """Draws a simple fish-like shape (ellipse body + triangle tail)."""
    overlay = img.copy()
    body_w, body_h = size, int(size * 0.55)
    cv2.ellipse(overlay, (cx, cy), (body_w, body_h), 0, 0, 360, color, -1)
    tail_pts = np.array([
        [cx - body_w, cy],
        [cx - body_w - int(size * 0.6), cy - int(size * 0.4)],
        [cx - body_w - int(size * 0.6), cy + int(size * 0.4)],
    ])
    cv2.fillPoly(overlay, [tail_pts], color)
    # eye + fine detail (this is what blur will destroy)
    cv2.circle(overlay, (cx + int(size * 0.5), cy - int(size * 0.1)),
               max(2, size // 12), (0, 0, 0), -1)
    for i in range(5):
        stripe_x = cx - body_w + int(i * body_w * 2 / 5)
        cv2.line(overlay, (stripe_x, cy - body_h), (stripe_x, cy + body_h),
                 (color[0] // 2, color[1] // 2, color[2] // 2), 1)

    return overlay  # blur, if any, is applied at image level for realism


def apply_attenuation(img, distance=0.0):
    """Simulates color/contrast loss with distance: reduce red & contrast,
    add haze veil, proportional to `distance` in [0,1]."""
    img_f = img.astype(np.float32)
    b, g, r = cv2.split(img_f)
    r *= (1.0 - 0.6 * distance)
    g *= (1.0 - 0.25 * distance)
    img_f = cv2.merge([b, g, r])

    # veiling haze: blend toward flat bluish-grey
    veil = np.full_like(img_f, (140, 110, 60))
    img_f = img_f * (1 - 0.5 * distance) + veil * (0.5 * distance)

    # contrast reduction
    mean = img_f.mean()
    img_f = mean + (img_f - mean) * (1 - 0.5 * distance)

    return np.clip(img_f, 0, 255).astype(np.uint8)


def make_all(out="test_images", seed=0):
    """Writes the three labeled fixtures. Seeded so verdicts are repeatable."""
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    H, W = 720, 1080

    # 1. Sharp, close subject
    img1 = water_background(H, W, rng, haze=0.1)
    img1 = draw_fish(img1, W // 2, H // 2, size=180, color=(60, 140, 220))
    cv2.imwrite(os.path.join(out, "sharp_close.jpg"), img1)

    # 2. Same framing, heavily blurred (simulates missed focus)
    img2 = cv2.GaussianBlur(img1, (31, 31), 12)
    cv2.imwrite(os.path.join(out, "blurry_close.jpg"), img2)

    # 3. Sharp detail, but tiny + hazy + low contrast (simulates distant subject)
    img3 = water_background(H, W, rng, haze=0.3)
    img3 = draw_fish(img3, W // 2, H // 2, size=30, color=(60, 140, 220))
    img3 = apply_attenuation(img3, distance=0.8)
    cv2.imwrite(os.path.join(out, "sharp_far.jpg"), img3)
    return out


if __name__ == "__main__":
    out = make_all(sys.argv[1] if len(sys.argv) > 1 else "test_images")
    print("Wrote 3 test images to", out)
