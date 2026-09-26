# Step 08 — Depth Evaluation

## Objective

Implement depth evaluation for the MiDaS + KITTI pipeline.

Focus ONLY on depth metrics and depth prediction vs ground-truth evaluation.

Do NOT implement:
- segmentation
- U-Net
- Cityscapes evaluation
- fusion
- scene understanding
- visualization
- full pipeline
- CI/CD
- training

---

## Read existing project

Before implementation, read:

- prompts/01_architecture.md
- prompts/02_config_utils.md
- prompts/03_dataset.md
- prompts/04_unet.md
- prompts/05_unet_inference.md
- prompts/06_segmentation_evaluation.md
- prompts/07_midas.md

Inspect:

- models/midas/
- preprocessing/kitti.py
- configs/midas.yaml
- configs/pipeline.yaml
- utils/
- existing tests

Reuse existing utilities and conventions.

---

# 1. Important MiDaS depth convention

MiDaS produces monocular RELATIVE INVERSE DEPTH.

The project convention established in Step 07 is:

- raw MiDaS prediction is relative, not metric
- larger prediction value means closer to camera
- raw prediction must not be interpreted as meters

Do NOT silently treat raw MiDaS output as metric depth.

---

# 2. Evaluation representations

Implement explicit depth representations.

At minimum support:

### A. Relative inverse-depth representation

Use the MiDaS raw prediction directly.

Do not convert it to meters.

### B. Aligned metric-depth representation

For metric metrics against KITTI ground truth, implement an explicit scale-alignment step.

Use median scaling:

    scale = median(gt_valid) / median(pred_valid)

    pred_aligned = pred * scale

where:

- gt_valid = valid KITTI metric depth values
- pred_valid = corresponding MiDaS prediction values
- scale = scalar alignment factor

Document clearly that this is evaluation-time alignment and not metric-depth prediction by MiDaS.

---

# 3. Valid mask

Evaluation must only use valid KITTI depth pixels.

Conditions:

- GT depth > 0
- finite GT
- finite prediction
- prediction > 0 if required by the selected metric/representation

Do not evaluate invalid KITTI pixels.

If no valid pixels exist, raise a clear error.

---

# 4. Metrics

Implement:

## RMSE

    RMSE = sqrt(mean((pred - gt)^2))

## MAE

    MAE = mean(abs(pred - gt))

## Absolute Relative Error

    AbsRel = mean(abs(pred - gt) / gt)

## Threshold accuracy

Implement:

    ratio = max(pred / gt, gt / pred)

    delta_1 = mean(ratio < 1.25)
    delta_2 = mean(ratio < 1.25^2)
    delta_3 = mean(ratio < 1.25^3)

All threshold metrics must operate only on valid pixels.

---

# 5. Relative-depth evaluation

Because MiDaS is relative inverse depth, do not claim RMSE/MAE in meters for raw MiDaS output.

Provide a clearly named representation/API for relative evaluation.

If comparing relative predictions with transformed GT, document exactly what transformation is used.

Do not invent a metric-depth interpretation.

---

# 6. Median scaling

Implement a reusable function such as:

    median_scale(pred, gt, valid_mask)

Requirements:

- validate shapes
- validate finite values
- require positive valid values
- compute scale only from valid pixels
- return aligned prediction and scale
- raise a clear error when valid data is insufficient

Do not modify the original prediction in-place.

---

# 7. KITTI depth compatibility

Use the existing KITTI preprocessing conventions from Step 03.

KITTI GT is metric depth in meters after preprocessing.

Do not change the KITTI loader.

Do not assume MiDaS and KITTI images are paired by filename unless the existing dataset loader explicitly provides a valid pair.

---

# 8. Resolution handling

Prediction and GT must have matching spatial dimensions before metric evaluation.

If resizing is required:

- depth prediction → continuous interpolation
- GT depth → appropriate depth resize handling
- valid mask must remain correct after resizing

Do not use nearest-neighbor interpolation for continuous predicted depth.

Prefer evaluating at a clearly defined common resolution.

---

# 9. API

Create:

    evaluation/depth_metrics.py

Implement reusable functions such as:

    rmse(...)
    mae(...)
    abs_relative_error(...)
    delta_accuracy(...)
    median_scale(...)
    evaluate_depth(...)

Suggested aggregate result:

    {
        "rmse": ...,
        "mae": ...,
        "abs_rel": ...,
        "delta1": ...,
        "delta2": ...,
        "delta3": ...,
    }

Keep naming consistent with existing segmentation metrics.

---

# 10. NumPy and PyTorch

Support NumPy and PyTorch where practical.

Internally convert to a consistent representation.

Use float64 or numerically stable computation for metric accumulation where appropriate.

Do not introduce unnecessary dependencies.

---

# 11. Validation

Validate:

- prediction shape
- GT shape
- valid mask shape
- numeric dtype
- finite values
- positive GT
- positive prediction where required
- non-empty valid pixels

Raise clear ValueError or project-specific error messages.

---

# 12. Tests

Create/update:

    tests/test_depth_metrics.py

Tests must be deterministic and offline.

Do NOT:

- download MiDaS
- access KITTI
- access Cityscapes
- require GPU

Test at minimum:

1. RMSE known example
2. MAE known example
3. AbsRel known example
4. delta1
5. delta2
6. delta3
7. valid mask
8. invalid GT ignored
9. invalid prediction ignored
10. median scaling known example
11. median scaling does not mutate input
12. empty valid mask
13. shape mismatch
14. non-finite values
15. negative/zero GT handling
16. aggregate evaluate_depth()
17. NumPy input
18. Torch input
19. perfect prediction
20. constant-scale prediction corrected by median scaling

Use small concrete arrays with hand-checkable expected values.

---

# 13. Configuration

Update configs only if necessary.

If configuration is added, make the evaluation convention explicit, for example:

    depth:
      representation: relative
      metric_alignment: median

Do not introduce unnecessary training parameters.

---

# 14. Scope restriction

Do NOT implement:

- MiDaS model changes
- U-Net
- Cityscapes
- segmentation metrics
- fusion
- scene analyzer
- visualization
- training
- CI/CD
- full pipeline

STOP after Step 08.

---

# 15. Existing tests

Run the complete test suite:

    pytest

Also verify imports:

    python -c "from evaluation.depth_metrics import *; print('Depth metrics OK')"

Preserve all previous tests.

---

# 16. Report

After implementation report:

1. files created
2. files modified
3. metrics implemented
4. valid-mask behavior
5. median scaling behavior
6. MiDaS relative-depth handling
7. KITTI compatibility
8. validation behavior
9. test strategy
10. test results
11. assumptions
12. problems/blockers

STOP after Step 08.