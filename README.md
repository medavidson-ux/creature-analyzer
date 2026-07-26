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
(0–1) and, on rejection, the specific reason. Full results also land in
the CSV.

## How it works

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
   magnitude, and an FFT high-frequency-energy ratio (most robust to
   backscatter speckle).
4. **Distance/prominence** — four proxies, since there's no depth data:
   - dark-channel haze estimate (more water column = hazier)
   - frame coverage (subject bounding-box area / image area)
   - local contrast (water flattens contrast with distance)
   - saturation (color also decays with distance)
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
