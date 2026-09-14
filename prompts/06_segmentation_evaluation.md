# Step 06 — Semantic Segmentation Evaluation

## Objective

Implement evaluation metrics for the U-Net semantic segmentation task.

This step compares:

Ground Truth
    +
U-Net Prediction
    ↓
Segmentation Metrics

Focus ONLY on segmentation evaluation.

Do NOT implement:

- U-Net training
- MiDaS
- depth estimation
- depth metrics
- fusion
- scene understanding
- visualization
- full pipeline
- CI/CD

---

# 1. Read Existing Specifications

Before making any changes:

1. Read prompts/01_architecture.md.
2. Read prompts/02_config_utils.md.
3. Read prompts/03_dataset.md.
4. Read prompts/04_unet.md.
5. Read prompts/05_unet_inference.md.

Inspect the existing:

- evaluation/
- preprocessing/
- models/unet/
- tests/
- configs/

Reuse existing project conventions.

Do NOT redesign the project.

Do NOT unnecessarily rename, delete, or move existing files.

---

# 2. Implementation Location

Implement segmentation metrics in:

```text
evaluation/segmentation_metrics.py
```
Implement tests in:

tests/test_segmentation_metrics.py

If the file already contains a partial implementation, improve/reuse it instead of creating a duplicate metric system.

# 3. Supported Metrics

Implement the following metrics:

Pixel Accuracy
Per-class IoU
Mean IoU (mIoU)
Per-class Dice
Mean Dice

The implementation must support:

num_classes = 19

by default.

Do NOT hard-code 19 inside the metric formulas.

# 4. Input Representation

Ground truth and prediction use Cityscapes train IDs:

0 ... 18

Ignore label:

255

Expected segmentation maps:

[H, W]

or optionally:

[B, H, W]

Support single-image evaluation at minimum.

Batch support may be implemented if it fits naturally with the existing architecture.

Do not require one-hot encoded inputs.

# 5. Ignore Label Handling

This is critical.

Cityscapes uses:

255 = ignore

Pixels with ground-truth value 255 MUST NOT contribute to:

Pixel Accuracy
IoU
Dice

Example:

Ground truth:

[0, 0, 1, 255]

Prediction:

[0, 1, 1, 2]

The pixel with GT = 255 must be excluded completely.

Do NOT treat 255 as an additional semantic class.

# 6. Pixel Accuracy

Implement:

Pixel Accuracy =
number of correctly classified valid pixels
/
number of valid pixels

Where:

valid pixel = ground_truth != ignore_index

Example:

GT:

[0, 0, 1, 255]

Prediction:

[0, 1, 1, 2]

Valid pixels:

3

Correct:

2

Pixel Accuracy:

2 / 3 = 0.6667

Handle the edge case where there are zero valid pixels.

Do not return NaN unexpectedly.

Use a clearly documented convention such as 0.0 or another consistent safe value.

# 7. IoU

For class c:

IoU_c = TP_c / (TP_c + FP_c + FN_c)

Where:

TP = true positives
FP = false positives
FN = false negatives

Ignore pixels where:

GT == 255

For each class, return its IoU.

Example output:

{
    0: ...,
    1: ...,
    ...
    18: ...
}

A list/array representation is also acceptable if consistent with the project API.

# 8. Mean IoU

Calculate:

mIoU = mean(IoU_c)

Important:

If a class has no ground-truth pixels and no predicted pixels, its IoU denominator is zero.

Do NOT automatically count such a class as IoU = 1.

Use a clearly documented convention.

Prefer excluding undefined classes from the mean:

mIoU = mean(valid class IoUs)

where valid means:

TP + FP + FN > 0

This avoids artificially inflating the metric.

If all classes are undefined, return a safe documented value.

# 9. Dice Score

For class c:

Dice_c = 2 * TP_c / (2 * TP_c + FP_c + FN_c)

Equivalent:

Dice_c = 2 * intersection / (prediction + ground_truth)

Ignore GT = 255.

Return per-class Dice and mean Dice.

Use the same undefined-class convention as IoU.

# 10. Confusion Matrix

Implement a reusable confusion-matrix calculation if useful.

Expected conceptual structure:

                 Predicted
              0   1   2   ... 18
Actual  0
        1
        2
       ...
       18

The confusion matrix should contain only valid ground-truth pixels.

Ignore label 255 must not appear as a class.

This matrix can then be reused to calculate:

Pixel Accuracy
IoU
Dice

Avoid duplicating the same pixel-counting logic unnecessarily.

# 11. API Design

Provide a clean API.

For example:

metrics = evaluate_segmentation(
    prediction,
    target,
    num_classes=19,
    ignore_index=255,
)

It may return something similar to:

{
    "pixel_accuracy": ...,
    "iou_per_class": ...,
    "miou": ...,
    "dice_per_class": ...,
    "mean_dice": ...,
}

The exact API may follow existing project conventions.

Also provide individual functions if useful, for example:

pixel_accuracy(...)
intersection_over_union(...)
dice_score(...)
mean_iou(...)

Avoid unnecessary complexity.

# 12. Input Validation

Validate inputs clearly.

Detect at least:

prediction/target shape mismatch
invalid tensor dimensions
invalid num_classes
invalid ignore_index
unsupported input types where appropriate

Do not silently reshape unrelated arrays into matching shapes.

Predictions should contain integer class IDs.

Ground truth should contain integer class IDs.

# 13. Tensor and NumPy Support

If practical, support both:

PyTorch tensors
NumPy arrays

This is useful because:

U-Net inference produces PyTorch tensors.
Dataset/visualization code may use NumPy arrays.

Do not duplicate the metric implementation for NumPy and PyTorch.

Convert inputs to a common internal representation where appropriate.

The implementation must remain simple and readable.

# 14. Numerical Stability

Metrics should return floating-point values.

Expected range:

0.0 <= Pixel Accuracy <= 1.0
0.0 <= IoU <= 1.0
0.0 <= mIoU <= 1.0
0.0 <= Dice <= 1.0
0.0 <= Mean Dice <= 1.0

Avoid unnecessary floating-point instability.

Do not use arbitrary large epsilons that materially alter the metric.

# 15. Tests

Create:

tests/test_segmentation_metrics.py

Tests must NOT require:

Cityscapes
KITTI
U-Net weights
GPU
internet

Use small synthetic segmentation maps.

Test at minimum:

Test 1 — Perfect prediction

GT:

[[0, 1],
 [2, 3]]

Prediction:

[[0, 1],
 [2, 3]]

Expected:

Pixel Accuracy = 1
mIoU = 1
Mean Dice = 1
Test 2 — Completely wrong prediction

Use a small example where every valid prediction is incorrect.

Verify that the resulting accuracy/IoU behave correctly.

Test 3 — Ignore index

GT:

[[0, 1],
 [2, 255]]

Prediction:

[[0, 1],
 [0, 2]]

Verify that the 255 pixel does NOT affect the metrics.

Test 4 — Pixel Accuracy

Use a manually calculable example and verify the exact result.

Test 5 — IoU

Use a small example where TP, FP and FN can be calculated manually.

Verify the IoU.

Test 6 — Dice

Use a manually calculable example.

Verify the Dice score.

Test 7 — Multiple classes

Use at least 3 classes and verify per-class metrics.

Test 8 — Missing class

Create a case where one class does not occur in GT or prediction.

Verify that undefined IoU/Dice does not incorrectly become 1 and does not silently inflate mIoU.

Test 9 — Shape mismatch

Verify that prediction and target with different spatial dimensions raise a clear error.

Test 10 — NumPy input

If NumPy support is implemented, verify it produces the same result as tensor input.

Test 11 — Torch input

Verify PyTorch tensors work correctly.

Test 12 — Batch input

If batch support is implemented, verify:

[B, H, W]

works correctly.

Test 13 — Range validation

Verify returned metrics remain within:

[0, 1]
# 16. Manual Verification

Use a tiny manually calculable example.

For example:

GT:
0 0
1 1

Prediction:
0 1
1 0

Calculate the expected metrics manually and verify the implementation against them.

Do not rely only on random test data.

# 17. Existing Tests

Run:

pytest

All previous tests must continue to pass:

Step 02
Step 03
Step 04
Step 05

Do not remove or weaken existing tests.

# 18. Import Test

Run:

python -c "from evaluation.segmentation_metrics import *; print('Segmentation metrics OK')"

It must complete without errors.

Also run a small metric smoke test.

# 19. Important Scope Restriction

This step evaluates semantic segmentation predictions only.

Do NOT:

train U-Net
download Cityscapes
download model weights
evaluate depth
implement MiDaS
implement fusion
implement scene understanding
implement visualization
implement full pipeline

The metric functions must remain independent from U-Net.

They should accept prediction + ground truth rather than directly loading a model.

# 20. Report

After implementation, report:

Files created
Files modified
Metric API
Pixel Accuracy implementation
IoU implementation
mIoU implementation
Dice implementation
Ignore-index handling
Undefined-class handling
Confusion matrix design
NumPy/PyTorch support
Input validation
Tests implemented
Tests executed
Test results
Import/smoke-test result
Problems or assumptions

STOP after Step 06.

Do NOT continue to Step 07.