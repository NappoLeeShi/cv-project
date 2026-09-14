# Step 14 — Evaluate MiDaS DPT-Large on KITTI

You are working inside the existing CV-PROJECT repository.

## Goal

Evaluate the existing pretrained MiDaS DPT-Large model on the FULL KITTI
validation set available in the project.

This is an EVALUATION and INFERENCE step only.

DO NOT train MiDaS.

The pretrained checkpoint is:

    checkpoints/dpt_large_384.pt

The KITTI dataset is already prepared as:

    data/kitti/
    ├── images/
    └── depth/

There should be 1000 RGB images and 1000 corresponding ground-truth depth maps.

---

## Existing project context

The project already contains the MiDaS implementation:

    models/midas/model.py
    models/midas/inference.py

and depth metrics:

    evaluation/depth_metrics.py

Reuse these existing implementations.

Do not replace them with another depth model or another metric library.

The current MiDaS implementation uses DPT-Large with:

    dpt_large_384.pt

MiDaS output is RELATIVE INVERSE DEPTH.

Important semantic contract:

    larger predicted value = closer to camera

MiDaS does NOT directly output metric depth in meters.

Do not describe raw MiDaS predictions as meters.

---

## Step 1 — Inspect existing code

Before modifying anything, inspect:

- models/midas/model.py
- models/midas/inference.py
- preprocessing/kitti.py
- evaluation/depth_metrics.py
- configs/midas.yaml
- configs/pipeline.yaml
- utils/config.py
- utils/seed.py

Understand the existing APIs before implementing evaluation.

Reuse existing functionality whenever possible.

Do not duplicate model loading, preprocessing, or metric formulas.

---

## Step 2 — Verify the dataset

Verify that:

    data/kitti/images/

contains 1000 RGB images.

Verify that:

    data/kitti/depth/

contains 1000 ground-truth depth maps.

Verify that images and depth maps are correctly paired according to the
existing KITTI loader.

Do not pair files merely by directory order if the existing loader already
provides a safer pairing mechanism.

Do not use the anonymous KITTI test set because its ground truth is not
available.

---

## Step 3 — Verify the checkpoint

Verify:

    checkpoints/dpt_large_384.pt

exists and can be loaded by the existing MiDaS implementation.

Do not download another model.

Do not change the model architecture.

Do not switch to another DPT/BEiT model.

---

## Step 4 — Create the evaluation module

Create:

    evaluation/evaluate_midas.py

The module should:

1. Load the existing MiDaS configuration.
2. Select CUDA when available.
3. Load the KITTI dataset.
4. Load the pretrained MiDaS DPT-Large checkpoint.
5. Set the model to evaluation mode.
6. Use torch.no_grad().
7. Run inference on ALL 1000 KITTI validation images.
8. Align the prediction with the ground-truth resolution as required.
9. Build the valid evaluation mask.
10. Apply median scaling for evaluation if supported by the existing
    depth-metric implementation.
11. Compute the existing depth metrics.
12. Aggregate the metrics over the complete 1000-image dataset.
13. Save the final evaluation results as JSON.

Do not train or update model parameters.

---

## Step 5 — Depth metric definitions

Reuse:

    evaluation/depth_metrics.py

Report at least:

- RMSE
- MAE
- AbsRel
- delta1
- delta2
- delta3

Respect the existing valid-depth mask.

Do not treat invalid/zero KITTI depth pixels as valid ground truth.

If the existing metric implementation supports median scaling, use it
consistently.

Clearly document in the JSON that median scaling is evaluation-time
alignment and does not mean MiDaS produces metric depth.

---

## Step 6 — Important depth semantics

The project uses MiDaS relative inverse depth.

Therefore:

    larger prediction → closer object
    smaller prediction → farther object

Do not invert this relationship.

Do not convert the raw MiDaS output to meters.

If median scaling is used:

    scaled_prediction = prediction * scale

where the scale is derived from the valid ground truth and prediction.

The purpose is to align the relative prediction with KITTI depth values
for evaluation.

---

## Step 7 — Aggregate evaluation correctly

Do NOT simply average arbitrary per-image metrics if the existing metric
implementation provides a better aggregation strategy.

Use the existing depth metric utilities.

For metrics where global aggregation is appropriate, accumulate the required
statistics over the complete validation set.

The final reported metrics must represent all 1000 KITTI samples.

Do not evaluate only a subset for the final real evaluation.

---

## Step 8 — Save results

Create:

    outputs/analysis/midas_kitti_evaluation.json

The JSON should contain information similar to:

{
    "model": "MiDaS DPT-Large",
    "checkpoint": "checkpoints/dpt_large_384.pt",
    "dataset": "KITTI",
    "split": "val_selection_cropped",
    "num_samples": 1000,
    "relative_inverse_depth": true,
    "larger_value_is_closer": true,
    "median_scaling": true,
    "metrics": {
        "rmse": ...,
        "mae": ...,
        "absrel": ...,
        "delta1": ...,
        "delta2": ...,
        "delta3": ...
    }
}

Use actual values.

Do not fabricate or estimate metrics.

---

## Step 9 — Optional prediction outputs

Add a CLI option such as:

    --save-predictions

if this fits the existing project style.

If enabled, save predicted depth maps under:

    outputs/depth/midas_kitti/

Do not enable prediction saving by default if it creates unnecessarily
large files.

Do not save only visualized images as the prediction data.

---

## Step 10 — CLI

The module should support:

    python -m evaluation.evaluate_midas \
        --config midas \
        --checkpoint checkpoints/dpt_large_384.pt

Add useful CLI overrides only if they fit the existing project style.

---

## Step 11 — Tests

Create:

    tests/test_evaluate_midas.py

Use tiny synthetic data and mocks where appropriate.

Do NOT run all 1000 KITTI images inside unit tests.

Test at least:

- checkpoint path handling
- model evaluation mode
- no-gradient inference
- prediction/GT shape alignment
- valid depth mask handling
- median scaling behavior
- metric result structure
- JSON output structure
- CLI argument parsing

Tests must not require the 1.4 GB real checkpoint unless explicitly marked
as a real integration test.

---

## Step 12 — Run tests

First run focused tests:

    pytest tests/test_evaluate_midas.py \
           tests/test_depth_metrics.py \
           tests/test_config.py -q

Then run the complete regression suite:

    pytest -q

Do not break existing tests.

---

## Step 13 — Real evaluation

After implementation and tests pass, run:

    python -m evaluation.evaluate_midas \
        --config midas \
        --checkpoint checkpoints/dpt_large_384.pt

The final real evaluation MUST use all 1000 KITTI validation samples.

Use CUDA when available.

Report:

- number of samples
- RMSE
- MAE
- AbsRel
- delta1
- delta2
- delta3
- checkpoint path
- output JSON path
- whether median scaling was used

---

## Constraints

- DO NOT train MiDaS.
- DO NOT modify the U-Net.
- DO NOT modify the U-Net checkpoint.
- DO NOT modify the fusion pipeline.
- DO NOT switch to another MiDaS architecture.
- DO NOT download another checkpoint.
- DO NOT use the anonymous KITTI test set.
- DO NOT fabricate metrics.
- DO NOT evaluate only a subset for the final evaluation.
- DO NOT describe raw MiDaS output as metric depth.
- Preserve the project's existing relative-inverse-depth semantics.
- Reuse existing depth metrics.

Stop after Step 14.

At the end, report:
1. Files created.
2. Files modified.
3. Focused test results.
4. Full pytest results.
5. Real KITTI evaluation results.
6. Saved JSON path.