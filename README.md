# Underwater Photo Quality Control

Automated backend filter that approves/rejects underwater photos based on:

1. **Subject sharpness** — is the main subject in focus?
2. **Subject prominence/distance** — is the subject close and large enough
   in frame to be visually appealing?

## Files

- `qc_pipeline.py` — core pipeline + CLI. Run this on a folder of images.
- `calibrate.py` — helper to inspect metric distributions on your own
  manually-sorted keep/reject photos, so you can set real thresholds.
- `make_test_images.py` — generates synthetic test images (sharp/close,
  blurry, sharp-but-distant) used to sanity-check the pipeline logic.
- `test_pipeline.py` — asserts the three synthetic verdicts (and reasons),
  plus contrast-invariance of the focus score. Run: `uv run --group dev pytest`.
- `pyproject.toml` — optional project-style dependency spec, if you'd
  rather use `uv sync` than run the scripts standalone.

## Quickstart (uv)

These scripts carry inline dependency metadata (PEP 723), so `uv run`
resolves and installs everything they need — `opencv-contrib-python`
(the contrib build, needed for `cv2.saliency`) and `numpy` — into an
isolated environment automatically. No manual venv or `pip install` step.

Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/

```bash
uv run qc_pipeline.py /path/to/photos --out results.csv
```

The first run downloads and caches the dependencies (a few seconds);
subsequent runs are fast. Alternatively, `chmod +x qc_pipeline.py` and
run it directly as `./qc_pipeline.py ...` since it has a `uv run --script`
shebang.

### Project-style alternative

If you'd rather work in a synced virtual environment (e.g. for editing
in an IDE with import resolution) instead of running scripts standalone:

```bash
uv sync
uv run python qc_pipeline.py /path/to/photos --out results.csv
```

Each image prints an APPROVE/REJECT verdict with sharpness/distance scores
(0–1) and, on rejection, the specific reason. The CSV also holds every raw
metric and the subject ROI, which is what you need for calibration.

## Burst grouping (best-of-group selection)

Photos of the same subject taken moments apart are grouped, and only the
best frame of each group is **selected**; the other passing frames are
marked as duplicates (`DUP #n`) instead of all being approved.

- **Grouping** uses EXIF capture time (including sub-seconds), sorted and
  chained: a photo joins the previous one's group if taken within
  `--max-gap` seconds (default 5) *and* the scene still looks the same
  (`--min-similarity`, default 0.7, on a brightness-normalized thumbnail —
  robust to strobe misfires and small camera moves, splits when you turn to
  a new subject). The default errs toward splitting: a false split keeps one
  extra frame; a false merge would discard a different subject.
- **Ranking** within a group is `0.65 × sharpness + 0.35 × distance`.
  Comparing frames of the same scene is far more reliable than absolute
  thresholds, since water, light and subject are held constant.
- **Selection** still requires passing the absolute thresholds, so a burst
  where every frame is bad selects nothing. `--keep-per-group N` keeps the
  top N.
- Photos without EXIF time (e.g. edited exports) are judged individually.
  `--no-groups` disables grouping entirely.

The CSV gains `selected`, `group`, `group_rank` and `capture_time` columns.

## How it works

0. **Resize** — images are downscaled to a 1600px long edge first. All
   metrics are per-pixel, so without this the same photo scores very
   differently from a 20MP vs a 6MP camera.
1. **Despeckle** — a light median filter suppresses backscatter
   (particulates lit by strobe) before sharpness is measured, so floating
   debris doesn't fake out the focus metric.
2. **Subject localization** — OpenCV spectral-residual saliency proposes
   candidate regions; candidates are filtered/ranked by *compactness* and
   *centrality* rather than raw size, because underwater saliency maps
   often flag elongated water-column gradients or light shafts that
   aren't the actual subject. Falls back to a center-weighted box if
   nothing scores well.
3. **Sharpness** — three focus measures computed on the subject region
   only (not the whole frame): variance-of-Laplacian, Tenengrad gradient
   energy, and an FFT high-frequency ratio. All three are
   **contrast-normalized**: water strips contrast with distance, and raw
   gradient energy can't tell "low contrast" from "out of focus". The FFT
   ratio is weighted lowest because it rises with sensor noise/backscatter.
4. **Distance/prominence** — four proxies, since there's no depth data:
   - underwater dark-channel haze (UDCP: min of blue/green only — the
     classic land-haze version includes red, which water absorbs, so it
     reads distant water as "clear")
   - frame coverage (subject bounding-box area / image area)
   - local contrast (water flattens contrast with distance)
   - red retention (red is absorbed first with distance). HSV saturation
     is reported but not scored: blue water is highly saturated, so it
     rises with distance rather than falling.
5. **Combine** — each group is normalized and weighted into a 0–1 score;
   both must clear their thresholds to approve.

## Calibration (do this before trusting it)

The default thresholds in `QualityControl()` were tuned against synthetic
test images, **not real underwater photos** — they're a reasonable
starting point, not ground truth. Before relying on this:

1. Manually sort ~100–200 of your own photos into `good/` and `bad/`
   folders.
2. Run:
   ```bash
   uv run calibrate.py --keep good/ --reject bad/
   ```
3. Use the printed min/mean/max per metric to adjust the `*_bounds`
   tuples and `sharpness_min` / `distance_min` in `QualityControl()`.

## Known limitations

- Saliency-based localization can still fail on very low-contrast or
  camouflaged subjects — if you have consistent subject categories (fish,
  divers, coral heads), swapping in a trained object detector (e.g. YOLO)
  for `localize_subject()` will be more reliable than saliency alone.
- Distance metrics are optical proxies, not true depth — extreme cases
  (turbid water making a close subject look hazy, or a strobe-lit close
  subject against a naturally saturated background) can fool them.
- No general aesthetic scoring (composition, color grading) is included;
  this pipeline only handles focus and subject prominence as scoped.
