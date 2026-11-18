# Code Review: Underwater Photo QC

Review date: 2026-10-03. Commit reviewed: `84bfd01`.
All findings below were verified by executing the code, not by inspection alone.

## Summary

The tool does what it says: it auto-rejects underwater photos on subject
sharpness and subject prominence. The structure is clean and the docstrings are
candid about limitations. The core problem is that **the sharpness metrics
measure local contrast, not focus**, so the focus gate and the distance gate are
measuring the same thing. The distance gate is close to inert for exactly the
case it was designed for.

There are no tests. The repo ships three labeled synthetic images with
documented expected verdicts, and nothing asserts them, which is why the
central defect went unnoticed.

## What it does

- `qc_pipeline.py:90` — `localize_subject()` proposes subject candidates with
  OpenCV's spectral-residual saliency and ranks blobs by compactness and
  centrality (`_contour_score`, `qc_pipeline.py:55`) rather than raw area.
- `qc_pipeline.py:177` — three focus measures on the subject ROI only:
  variance-of-Laplacian, Tenengrad, FFT high-frequency energy ratio.
- `qc_pipeline.py:250` — four distance proxies: dark-channel haze, frame
  coverage, local contrast, saturation.
- `qc_pipeline.py:286` — `QualityControl` min-max normalizes each metric
  against hardcoded bounds and gates the two group scores at 0.35.
- `calibrate.py` — prints min/mean/max per metric over two manually-sorted
  folders so the bounds can be hand-tuned.
- `make_test_images.py` — synthesizes three labeled images as a sanity check.

## Verified defect 1: the documented expected verdict is wrong

`README.md:14` and `make_test_images.py:12` both state that `sharp_far.jpg`
should be rejected **for distance**. Actual output:

```
[REJECT] sharp_far.jpg   sharp=0.28 dist=0.35   subject out of focus (score 0.28)
```

It was rejected by the focus gate. Its distance score of 0.354 passed
`distance_min=0.35`. The distance gate did not fire.

## Verified defect 2: root cause is metric confound, not localization

The first hypothesis — that subject localization failed and the center-box
fallback was measuring background — was wrong. Instrumenting the contour
candidates shows the ROI was found correctly:

```
sharp_far.jpg  best contour = (479, 326, 108, 68)  area_frac=0.0076
               -> no fallback used
```

The real cause is that `laplacian` and `tenengrad` are energy measures of the
local derivative field and therefore scale with roughly the square of local
contrast. Water attenuates contrast with distance, so a distant subject is
mathematically indistinguishable from a defocused one under these two metrics.

### Evidence: contrast-only sweep on a fixed, in-focus ROI

Contrast reduced, no blur applied:

| contrast gain | laplacian | tenengrad | fft_ratio | verdict |
|---|---|---|---|---|
| 1.00 | 105.0 | 1282.3 | 0.611 | APPROVE |
| 0.70 | 51.6 | 627.8 | 0.599 | APPROVE |
| 0.50 | 26.7 | 322.1 | 0.582 | APPROVE |
| 0.35 | 13.2 | 157.6 | 0.563 | APPROVE |
| 0.20 | 4.5 | 52.0 | 0.516 | REJECT (out of focus) |

### Evidence: real defocus at constant contrast

| blur ksize | laplacian | tenengrad | fft_ratio | sharp_score |
|---|---|---|---|---|
| 0 | 105.0 | 1282.3 | 0.611 | 0.536 |
| 3 | 20.1 | 996.4 | 0.471 | 0.368 |
| 7 | 2.5 | 536.6 | 0.264 | 0.143 |
| 15 | 0.7 | 248.2 | 0.131 | 0.019 |
| 31 | 0.7 | 113.7 | 0.144 | 0.009 |

A **sharp** image at 20% contrast scores `laplacian=4.5`. A genuinely
**blurred** image scores `laplacian=2.5`. The sharp image is judged blurrier
than the blurred one. This is defect 1 reproduced in isolation.

`fft_ratio` is the only measure that separates the two effects: it moves 1.2x
under extreme contrast loss but 4.7x under real blur. The comment at
`qc_pipeline.py:318` correctly identifies this, then weights the most
contrast-confounded metric (`laplacian`, 23x swing) at 0.3 and the robust one
at only 0.4.

**Consequence:** the current defaults cannot be fixed by recalibration alone.
The confound is inside the metric definition, not in the bounds.

**Fix direction:** make sharpness contrast-invariant, either by normalizing
gradient energy by local RMS contrast or by using a band-ratio measure that
references mid-frequency energy instead of absolute energy. Then re-tune bounds.

## Bugs and dead code

| # | Location | Issue |
|---|---|---|
| 1 | `qc_pipeline.py:29` | `import json` is never used. No JSON output is written anywhere. |
| 2 | `qc_pipeline.py:233` | `red_ratio` is computed and returned but never read. `score_distance` (`qc_pipeline.py:322`) ignores it and `calibrate.py:70` omits it from its key list. |
| 3 | `qc_pipeline.py:351` | `raw_metrics` is built and stored on `QCResult` but never persisted. The CSV header (`qc_pipeline.py:391`) has 5 columns. `README.md:48` claims "Full results also land in the CSV" — that is false. |
| 4 | `qc_pipeline.py:374` | `main()` crashes with a bare `FileNotFoundError` traceback if the folder does not exist. Confirmed by running it. |
| 5 | `qc_pipeline.py:378` | The empty-folder path returns exit code 0 and writes no CSV, so a caller cannot distinguish "no images found" from "everything rejected". |
| 6 | `qc_pipeline.py:394` | CSV writes Python `True`/`False` rather than `true`/`false`, awkward for `jq` or pandas consumers. |
| 7 | `qc_pipeline.py:373` | No directory recursion and no RAW/HEIC support. Real dive photo folders are nested with RAW+JPEG pairs. |
| 8 | `qc_pipeline.py:329`, `calibrate.py:47` | The load → localize → measure sequence is duplicated, as is the `exts` tuple. Any metric change must be made in two places. |
| 9 | `make_test_images.py:57` | `draw_fish`'s `sharp` parameter is dead — both branches return `overlay`. |
| 10 | `make_test_images.py:33` | `np.random.normal` is unseeded, so output differs every run and no future test can rely on these fixtures. |
| 11 | `make_test_images.py` | Module-level side effects with no `if __name__ == "__main__"` guard, unlike the other two scripts. |
| 12 | `qc_pipeline.py:366` | Only `sharpness_min`/`distance_min` are exposed as flags, yet `README.md:86` instructs editing `*_bounds` in source. |
| 13 | `calibrate.py:79` | Prints stats under a "Suggested approach" heading but does not actually suggest bounds. Could emit ready-to-paste values. |
| 14 | `qc_pipeline.py:200` | `dark_channel_prior` docstring says "in a region" but it is computed over the whole frame then cropped. Correct in effect, misleading in wording. |
| 15 | `.gitignore` | Does not cover `test_images/`, `*.csv`, or `.uv-cache/`. |

## Structural and methodology gaps

- **No tests.** A labeled fixture set with documented expected verdicts exists
  and is unused. Wiring it up as an assertion on both verdict *and* reason
  string would have caught defect 1 immediately.
- **Subject metrics include background.** Contrast, saturation, and haze are
  averaged over the full bounding box. For a small subject that box is mostly
  water, diluting the reading toward background values. A silhouette mask
  would be more faithful.
- **Sequential processing.** The full-image `dark_channel_prior` and the ROI
  FFT dominate runtime. A `ThreadPoolExecutor` would help across cores, since
  OpenCV releases the GIL.
- **`QualityControl` is a plain class** that only mirrors constructor args into
  attributes. A frozen dataclass would be less code and directly serializable
  to a config file, which would also solve gap 12.
- **No confidence output.** A binary verdict with no margin gives no way to
  sort borderline cases for manual review.

## Recommended order of work

1. Make sharpness contrast-invariant (band-ratio or contrast-normalized) and
   drop or heavily down-weight `laplacian`. This is the only change that
   alters core behavior.
2. Add a test asserting the three documented verdicts including the reason
   string.
3. Fix the factual defects: unused `json` import, dead `red_ratio`,
   `raw_metrics` missing from the CSV, and the false "full results" claim in
   the README.
4. Add the folder-exists check, recursion, and non-JPEG extensions.
5. Deduplicate the metric pipeline between `qc_pipeline.py` and
   `calibrate.py`; expose the bounds on the CLI.
6. Only then consider a trained object detector, threading, and silhouette
   masking. Those are wasted effort on top of a metric that cannot separate
   its own two criteria.

## Reproducing the evidence

```bash
uv run make_test_images.py
uv run qc_pipeline.py test_images --out out.csv
```

For the contrast and blur sweeps, iterate over a fixed ROI from
`sharp_close.jpg` (196, 219, 634, 282), applying either
`mean + (img - mean) * gain` or `cv2.GaussianBlur`, and record
`laplacian`, `tenengrad`, `fft_ratio` plus `QualityControl.score_sharpness()`.
