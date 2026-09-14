# Step 15 — Full-Pipeline Evaluation (Fusion + Scene Understanding)

Implements a reproducible full-pipeline demo/evaluation that runs the **same**
RGB image through segmentation (U-Net) and relative-depth (MiDaS), fuses the
two outputs, runs scene analysis, and computes rule-based difficulty
indicators. This document covers purpose, architecture, the same-image
contract, fusion logic, the difficulty score, thresholds, limitations, and
reproducibility.

## Purpose

Measure how the *existing* trained U-Net segmentation and *existing* MiDaS
depth outputs combine for scene understanding on standalone street images.

- Uses existing modules only (`models/unet/`, `models/midas/`,
  `scene_understanding/`, `visualization/`, `evaluation/`).
- Uses the existing trained checkpoints (`checkpoints/unet_cityscapes.pth`,
  `checkpoints/dpt_large_384.pt`).
- Produces structured JSON results plus visualization files.

This is **not** a model-level benchmark. Model-level metrics remain the
separate, authoritative results from Step 13 (U-Net on Cityscapes GT:
mIoU / pixel accuracy / Dice) and Step 14 (MiDaS on KITTI GT:
RMSE / MAE / AbsRel / delta1..3). Those two benchmarks are never merged.

## Architecture

```
data/pipeline/images/{image}.png
        │  (SAME image object)
        ├──────────────► U-Net  ──────────► [H,W] trainId mask
        │
        └──────────────► MiDaS  ──────────► [H,W] relative inverse depth
                                  │
                                  ▼
            alignment to the source resolution (nearest for IDs,
            bilinear for depth)
                                  │
                                  ▼
            fusion (scene_understanding.fusion.fuse)  → FusionResult
                                  │
                                  ▼
            scene analysis (scene_understanding.analyzer.analyze_fusion)
                                  │
                                  ▼
            difficulty analysis (evaluation.difficulty_analysis)
                                  │
                                  ▼
            JSON results + visualizations
```

Key files:

- `evaluation/evaluate_pipeline.py` — orchestrator + CLI.
- `evaluation/difficulty_analysis.py` — rule-based difficulty score.
- `scene_understanding/pipeline.py` — existing single-image orchestration
  (reused for spatial-alignment helpers and conventions).
- `scene_understanding/fusion.py` — existing analytical fusion.
- `scene_understanding/analyzer.py` — existing scene report.
- `visualization/` — existing segmentation / depth / fusion / scene rendering.
- `tests/test_evaluate_pipeline.py` — offline, mock/synthetic tests.

## Why the datasets are NOT paired

Cityscapes and KITTI are two different datasets with different cameras,
capture geometries, and annotation schemes. There is no valid correspondence
between a Cityscapes RGB image and a KITTI depth map.

- A Cityscapes image must **never** be paired with a KITTI depth map.
- Step 13 (U-Net) uses Cityscapes ground truth **only**.
- Step 14 (MiDaS) uses KITTI ground truth **only**.
- Step 15 runs both models on the **same standalone RGB image** (no external
  ground truth at all).

The two benchmark spaces are therefore kept strictly separate.

## Same-image contract

For every evaluated image the pipeline guarantees:

- exactly one RGB image is loaded;
- the *identical* image object feeds U-Net **and** MiDaS;
- both predictions are aligned to the *same* source resolution before fusion
  (segmentation with nearest-neighbour, depth with bilinear);
- fusion operates only on outputs derived from that one image.

This is enforced in `evaluate_single_image` and verified by the unit tests.

## Fusion logic

Existing rule-based fusion (`scene_understanding/fusion.py`):

- Requires the segmentation and depth to share spatial dimensions.
- Non-finite depth pixels and void (255) class pixels are excluded from all
  statistics via the analyzed mask.
- Per-class statistics: pixel count/ratio, mean/median/min/max inverse depth.
- Relative depth regions derived from the prediction itself (default terciles):
  near = top third, middle = middle third, far = bottom third.
- Depth convention is preserved: **larger inverse depth = closer**, smaller =
  farther; values are unit-less, never meters.

Median scaling is only used in Step 14's KITTI evaluation to align relative
predictions to GT; it is not applied to the raw demo depth maps here.

## Difficulty score definition

The difficulty score is a transparent, rule-based **pipeline difficulty
score** / **scene complexity score**. It is **not** an objective
ground-truth difficulty label.

The default score is a weighted sum of five normalized indicators (each
clamped to `[0, 1]`):

| # | indicator                   | definition                                                    | weight |
|---|-----------------------------|---------------------------------------------------------------|--------|
| 1 | `segmentation_uncertainty`  | `1 - mean_confidence` (mean U-Net softmax confidence)         | 0.25   |
| 2 | `depth_variation`           | coefficient of variation of finite inverse depth              | 0.20   |
| 3 | `scene_complexity`          | present classes / 19 Cityscapes trainIds                      | 0.20   |
| 4 | `foreground_fraction`       | `near` depth-region pixel ratio                               | 0.20   |
| 5 | `object_density`            | ROI-class (person..bicycle) pixel ratio                       | 0.15   |

```
difficulty_score = clamp( Σᵢ weightᵢ × indicatorᵢ , 0, 1 )
```

Weights and bins are configurable (`configs/pipeline.yaml` →
`difficulty.factor_weights`, `difficulty.bins`) and documented in
`evaluation/difficulty_analysis.py`.

## Easy / Medium / Hard thresholds

```
score ≤ 0.4  →  easy
score ≤ 0.7  →  medium
else         →  hard
```

Threshold defaults live in `difficulty.bins` in `configs/pipeline.yaml`
(`easy: 0.4`, `medium: 0.7`).

## Evaluation output

One entry per image (written to `outputs/analysis/pipeline_evaluation.json`):

- `image` — file name.
- `segmentation` — `num_classes_present`, `mean_confidence`.
- `depth` — `mean_inverse_depth`, `std_inverse_depth`, `min_inverse_depth`,
  `max_inverse_depth`.
- `scene` — the existing scene report (semantic distribution, depth
  distribution, per-class regions, traffic context, interpretation).
- `difficulty` — `score`, `level`, `components` (per-indicator values).

Plus aggregate statistics:

- number of images;
- average difficulty score;
- Easy / Medium / Hard counts;
- average segmentation confidence;
- average depth variation;
- class occurrence across images.

When `--save-visualizations` is set, per-image files are written to
`outputs/segmentation/`, `outputs/depth/`, and `outputs/analysis/`:
original, segmentation, depth, fusion overlay, overview figure, and a
per-image scene report.

## Limitations and assumptions

- Difficulty is a rule-based heuristic derived from predicted outputs; it is
  not a ground-truth label and cannot be compared across settings directly.
- Raw MiDaS output is relative inverse depth (unit-less); nothing here
  produces metric (meter) depth.
- Segmentation confidence requires a predictor that supports
  `predict(image, return_confidence=True)` (the project's U-Net wrapper does);
  otherwise `mean_confidence` is `null` and the uncertainty indicator is
  treated as neutral (0.5).
- Class counts and proximities describe what the models *predict*, not ground
  truth.
- Demo images are standalone; they are not paired with any dataset ground
  truth.

## Reproducibility commands

Full pytest regression suite (offline; no models/downloads needed):

```bash
.venv/bin/python -m pytest -q
```

Pipeline smoke test on standalone RGB images (2 images, CUDA when available):

```bash
.venv/bin/python -m evaluation.evaluate_pipeline \
    --config configs/pipeline.yaml \
    --input-dir data/pipeline/images \
    --limit 2 \
    --save-visualizations \
    --device auto
```

Useful options: `--limit N`, `--device cpu|cuda|auto`,
`--save-visualizations`, `--output PATH/TO/DIR`,
`--unet-checkpoint PATH`, `--midas-weights PATH`.