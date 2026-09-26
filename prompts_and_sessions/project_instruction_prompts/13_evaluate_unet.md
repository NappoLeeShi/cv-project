# Step 13 — Evaluate trained U-Net on Cityscapes validation

You are working inside the existing CV-PROJECT repository.

## Goal

Evaluate the already-trained U-Net checkpoint on the FULL Cityscapes validation split.

The trained checkpoint is:

    checkpoints/unet_cityscapes.pth

The goal is to produce reproducible segmentation evaluation results using the existing
dataset loader, U-Net model, inference code, and segmentation metrics.

This is an EVALUATION step only.

DO NOT train the model again.

---

## Existing project context

The project implements:

Cityscapes image
    ↓
U-Net
    ↓
semantic segmentation prediction
    ↓
segmentation metrics

Existing components that MUST be reused:

- preprocessing/cityscapes.py
- models/unet/model.py
- models/unet/inference.py
- evaluation/segmentation_metrics.py
- configs/unet.yaml
- utils/config.py
- utils/seed.py
- existing device utilities if applicable

The trained model has:

- 19 semantic classes
- ignore_index = 255
- image size = 256 × 512 during training
- checkpoint = checkpoints/unet_cityscapes.pth

Training result:

- best epoch = 20
- validation loss = 0.3135
- validation mIoU = 0.4262
- validation pixel accuracy = 0.9021

These numbers came from the training validation loop. Step 13 should independently
run evaluation using the saved checkpoint and verify the result.

---

## Requirements

### 1. Inspect the existing implementation first

Before changing anything, inspect:

- models/unet/model.py
- models/unet/inference.py
- preprocessing/cityscapes.py
- evaluation/segmentation_metrics.py
- configs/unet.yaml
- training/trainer.py

Do not duplicate functionality that already exists.

Do not change existing model architecture or metric definitions unless there is a
clear bug that prevents evaluation.

---

### 2. Create an evaluation module

Create:

    evaluation/evaluate_unet.py

It should:

1. Load the existing config.
2. Select CUDA when available.
3. Load the existing Cityscapes validation dataset.
4. Load:

       checkpoints/unet_cityscapes.pth

5. Construct the existing U-Net architecture using the checkpoint/config.
6. Load model_state_dict.
7. Set model.eval().
8. Use torch.no_grad().
9. Run inference on ALL 500 Cityscapes validation images.
10. Compare predictions with the corresponding Cityscapes GT labels.
11. Compute the existing segmentation metrics.

Do not train or update model parameters.

---

### 3. Metrics

Use the existing implementation in:

    evaluation/segmentation_metrics.py

Report at least:

- Pixel Accuracy
- Mean IoU (mIoU)
- Mean Dice
- IoU for each of the 19 classes
- Accuracy for each class when supported by the existing metric implementation

Use:

    ignore_index = 255

Do not silently include ignored pixels.

Use the existing Cityscapes class mapping/names from the project.

---

### 4. Save evaluation results

Create:

    outputs/analysis/unet_cityscapes_evaluation.json

The JSON should contain information similar to:

{
    "checkpoint": "...",
    "split": "val",
    "num_samples": 500,
    "num_classes": 19,
    "ignore_index": 255,
    "image_size": [256, 512],
    "pixel_accuracy": ...,
    "miou": ...,
    "mean_dice": ...,
    "per_class": {
        "road": {
            "iou": ...,
            "dice": ...,
            "accuracy": ...
        },
        ...
    }
}

Use the actual class names from the existing project instead of inventing names.

Also include the checkpoint epoch if it is available from the checkpoint metadata.

---

### 5. Optional prediction outputs

If practical, add a CLI option such as:

    --save-predictions

When enabled, save the predicted class-index masks under:

    outputs/segmentation/unet_cityscapes/

Do not enable this by default if it would unnecessarily consume a large amount of
disk space.

The saved prediction mask should preserve class IDs, not only a colored visualization.

---

### 6. CLI

The module should support:

    python -m evaluation.evaluate_unet --config unet

and preferably:

    python -m evaluation.evaluate_unet \
        --config unet \
        --checkpoint checkpoints/unet_cityscapes.pth

Add useful options only if they fit the existing project style.

Do not introduce unnecessary dependencies.

---

### 7. Tests

Add focused tests for the new evaluation code.

Tests should use tiny synthetic data or mocked model outputs where possible.

Test at least:

- checkpoint loading
- evaluation runs without gradients
- output dimensions are handled correctly
- ignore_index is respected
- JSON result structure
- CLI argument parsing

Do NOT run all 500 Cityscapes images inside unit tests.

---

### 8. Real evaluation

After implementation and tests pass, run the real evaluation:

    python -m evaluation.evaluate_unet \
        --config unet \
        --checkpoint checkpoints/unet_cityscapes.pth

It MUST evaluate all 500 validation samples.

Report:

- number of samples evaluated
- Pixel Accuracy
- mIoU
- Mean Dice
- best/worst class IoU
- per-class IoU if available
- output JSON path

Do not fabricate or estimate any metric.

---

### 9. Regression testing

Run the complete pytest suite.

The existing project currently has a large regression suite.

Do not break existing behavior.

Report:

- total tests
- passed
- skipped
- failed

---

## Important constraints

- DO NOT retrain U-Net.
- DO NOT train MiDaS.
- DO NOT modify MiDaS.
- DO NOT modify the fusion pipeline.
- DO NOT change U-Net architecture.
- DO NOT change the Cityscapes class mapping.
- DO NOT replace the existing segmentation metric implementation.
- DO NOT use a black-box segmentation library.
- DO NOT evaluate only a small subset for the final real evaluation.
- The final real evaluation must use all 500 Cityscapes validation images.

Stop after Step 13.

At the end, report exactly what files were created/modified, test results, and the
real evaluation metrics.