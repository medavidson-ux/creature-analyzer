"""Run with:  uv run --group dev pytest"""
import os
import shutil

import cv2
import numpy as np
import pytest

from make_test_images import make_all
from qc_pipeline import (QualityControl, combined_sharpness_score,
                         dark_channel_prior, load_image)


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    return make_all(str(tmp_path_factory.mktemp("imgs")), seed=0)


@pytest.mark.parametrize("name, approved, reason_has, reason_lacks", [
    ("sharp_close.jpg", True, "approved", None),
    ("blurry_close.jpg", False, "out of focus", None),
    ("sharp_far.jpg", False, "too distant", "out of focus"),
])
def test_documented_verdicts(images, name, approved, reason_has, reason_lacks):
    r = QualityControl().evaluate(os.path.join(images, name))
    assert r.approved is approved
    assert reason_has in r.reason
    if reason_lacks:
        assert reason_lacks not in r.reason


def test_sharpness_is_contrast_invariant(images):
    """Contrast loss from water must not look like defocus."""
    qc = QualityControl()
    img = cv2.imread(os.path.join(images, "sharp_close.jpg"))
    roi = (196, 219, 634, 282)
    f = img.astype(np.float32)
    # 20% contrast, no blur -- the original metrics rejected this as out of focus
    low = np.clip(f.mean() + (f - f.mean()) * 0.2, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    s_low = qc.score_sharpness(combined_sharpness_score(low, roi))
    s_blur = qc.score_sharpness(combined_sharpness_score(blurred, roi))
    assert s_low >= qc.sharpness_min
    assert s_low > s_blur


def test_haze_reads_far_water_as_far():
    """Red is absorbed first underwater; haze must not mistake that for clarity."""
    near_subject = np.full((40, 40, 3), (60, 120, 210), np.uint8)   # BGR
    far_water = np.full((40, 40, 3), (170, 90, 5), np.uint8)
    assert dark_channel_prior(far_water).mean() > dark_channel_prior(near_subject).mean()


def test_non_ascii_path(images, tmp_path):
    dst = tmp_path / "fisch_ä" / "ü.jpg"
    dst.parent.mkdir()
    shutil.copy(os.path.join(images, "sharp_close.jpg"), dst)
    assert load_image(str(dst)) is not None


# ---------------------------------------------------------------------------
# Burst grouping
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta

from PIL import Image

from qc_pipeline import (QCResult, capture_time, group_photos, scene_signature,
                         select_best)

T0 = datetime(2026, 10, 3, 10, 0, 0)


def _save_with_exif(img_bgr, path, when):
    im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")
    exif.get_ifd(0x8769)[0x9291] = f"{when.microsecond // 10000:02d}"
    im.save(path, exif=exif, quality=95)


def _r(name, secs, sig=None, sharp=0.5, dist=0.5, approved=True):
    return QCResult(name, approved, "approved" if approved else "bad",
                    sharp, dist, {}, capture_time=T0 + timedelta(seconds=secs),
                    signature=sig)


def test_capture_time_reads_subseconds(images, tmp_path):
    img = cv2.imread(os.path.join(images, "sharp_close.jpg"))
    when = T0 + timedelta(milliseconds=250)
    _save_with_exif(img, tmp_path / "a.jpg", when)
    assert capture_time(str(tmp_path / "a.jpg")) == when
    assert capture_time(os.path.join(images, "sharp_close.jpg")) is None


def test_grouping_by_time_gap():
    rs = [_r("a", 0), _r("b", 1), _r("c", 3), _r("d", 30), _r("e", 31)]
    group_photos(rs, max_gap_s=5, min_similarity=0)
    assert [r.group for r in rs] == [1, 1, 1, 2, 2]


def test_grouping_splits_on_scene_change(images):
    """Same burst survives an exposure change; a new subject splits off."""
    base = cv2.imread(os.path.join(images, "sharp_close.jpg"))
    dark = cv2.convertScaleAbs(base, alpha=0.5)          # strobe didn't fire
    new = base.copy()
    cv2.circle(new, (540, 360), 250, (40, 200, 90), -1)   # turned to a coral head
    rs = [_r(n, i, scene_signature(img))
          for i, (n, img) in enumerate([("a", base), ("b", dark), ("c", new)])]
    group_photos(rs, max_gap_s=5)
    assert [r.group for r in rs] == [1, 1, 2]


def test_untimed_photos_are_not_grouped():
    rs = [_r("a", 0), _r("b", 0)]
    for r in rs:
        r.capture_time = None
    group_photos(rs)
    assert rs[0].group != rs[1].group


def test_best_of_group_requires_passing_qc():
    rs = [_r("a", 0, sharp=0.9), _r("b", 1, sharp=0.6),
          _r("c", 60, sharp=0.2, approved=False), _r("d", 61, sharp=0.1, approved=False)]
    select_best(group_photos(rs, min_similarity=0))
    assert [r.selected for r in rs] == [True, False, False, False]
    assert [r.group_rank for r in rs] == [1, 2, 1, 2]


def test_burst_end_to_end(images, tmp_path):
    """Three frames of one subject with varying focus, then a new subject
    a minute later: the sharpest frame of the burst wins."""
    sharp = cv2.imread(os.path.join(images, "sharp_close.jpg"))
    far = cv2.imread(os.path.join(images, "sharp_far.jpg"))
    frames = [("b1_soft.jpg", cv2.GaussianBlur(sharp, (5, 5), 0), 0.0),
              ("b2_sharp.jpg", sharp, 0.4),
              ("b3_blurry.jpg", cv2.GaussianBlur(sharp, (15, 15), 0), 0.8),
              ("later.jpg", far, 60.0)]
    for name, img, secs in frames:
        _save_with_exif(img, tmp_path / name, T0 + timedelta(seconds=secs))

    qc = QualityControl()
    rs = [qc.evaluate(str(tmp_path / n)) for n, _, _ in frames]
    select_best(group_photos(rs))
    by = {r.filename: r for r in rs}
    assert by["b1_soft.jpg"].group == by["b2_sharp.jpg"].group == by["b3_blurry.jpg"].group
    assert by["later.jpg"].group != by["b2_sharp.jpg"].group
    assert by["b2_sharp.jpg"].selected and by["b2_sharp.jpg"].group_rank == 1
    assert not by["b1_soft.jpg"].selected


def test_underexposed_frame_does_not_look_closer(images):
    """Within a burst a strobe misfire must not win on 'less haze'."""
    from qc_pipeline import measure
    img = cv2.imread(os.path.join(images, "sharp_close.jpg"))
    full = measure(img)["haze"]
    dark = measure(cv2.convertScaleAbs(img, alpha=0.5))["haze"]
    assert dark >= full - 0.02
