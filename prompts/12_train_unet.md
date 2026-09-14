# STEP 12 — TRAIN U-NET ON CITYSCAPES

## 1. GOAL

Implement the training pipeline for the existing U-Net model using the existing Cityscapes dataset.

Current project status before Step 12:

- Step 02 → Step 11 completed
- pytest result:
  332 passed, 1 skipped, 333 collected
- Existing U-Net architecture already works.
- Existing U-Net inference already works.
- Existing Cityscapes dataset/preprocessing already works.
- Existing segmentation metrics already work.
- Existing full pipeline already works.
- Do NOT break any existing functionality.

The goal of Step 12 is:

Cityscapes images + semantic labels
        ↓
   U-Net training
        ↓
 validation
        ↓
 best checkpoint
        ↓
 checkpoints/unet_cityscapes.pth

This checkpoint will later be used by the real Step 11 pipeline.

---

## 2. IMPORTANT SCOPE

Step 12 is ONLY about training U-Net.

DO NOT:

- rewrite U-Net architecture
- change U-Net model structure
- rewrite CityscapesDataset
- rewrite preprocessing
- rewrite segmentation metrics
- rewrite MiDaS
- train MiDaS
- modify fusion logic
- modify analyzer logic
- modify visualization logic
- modify Step 11 pipeline architecture
- add object detection
- add tracking
- add lane detection
- add 3D reconstruction
- add an agent
- add a neural fusion network
- download datasets automatically
- download pretrained weights automatically
- require internet
- create fake/random checkpoint and call it trained
- silently fall back to random U-Net weights for real inference

Reuse existing modules wherever possible.

Before editing anything, inspect the repository.

---

# 3. READ EXISTING CODE FIRST

Inspect at minimum:

- models/unet/model.py
- models/unet/inference.py
- preprocessing/cityscapes.py
- evaluation/segmentation_metrics.py
- configs/unet.yaml
- configs/pipeline.yaml
- utils/config.py
- utils/device.py
- utils/seed.py
- main.py
- tests/test_unet.py
- tests/test_unet_inference.py
- tests/test_cityscapes.py
- scene_understanding/pipeline.py

Also inspect the existing project structure and conventions.

Do NOT create duplicate dataset/model implementations.

---

# 4. EXISTING U-NET CONTRACT

The existing U-Net must remain unchanged unless a very small compatibility fix is absolutely required.

Expected behavior:

Input:

    [B, 3, H, W]

Output:

    [B, NUM_CLASSES, H, W]

The output must be RAW LOGITS.

Do NOT apply softmax inside the model.

Training will use:

    CrossEntropyLoss

with target:

    [B, H, W]

and ignore index:

    255

because Cityscapes preprocessing already maps ignored classes to 255.

The existing Cityscapes mapping must remain the source of truth.

There are 19 trainable semantic classes:

0  road
1  sidewalk
2  building
3  wall
4  fence
5  pole
6  traffic light
7  traffic sign
8  vegetation
9  terrain
10 sky
11 person
12 rider
13 car
14 truck
15 bus
16 train
17 motorcycle
18 bicycle

Ignore:

255

Do not invent a different class mapping.

---

# 5. TRAINING MODULE

Create a clean training module, preferably:

training/
├── __init__.py
├── trainer.py
└── train_unet.py

If the repository already has an appropriate training structure, reuse it instead of duplicating it.

Responsibilities:

## trainer.py

Implement a reusable U-Net trainer.

It should support:

- model
- train dataloader
- validation dataloader
- optimizer
- loss function
- device
- number of epochs
- checkpoint directory
- seed/configuration
- training loop
- validation loop
- metric tracking
- best-checkpoint saving

Do not put CLI parsing inside the trainer.

---

# 6. TRAINING LOOP

For every training epoch:

1. model.train()

2. Iterate over Cityscapes training batches.

3. Move image and target to device.

4. Forward:

    logits = model(images)

5. Compute:

    loss = CrossEntropyLoss(ignore_index=255)(logits, targets)

6. optimizer.zero_grad()

7. loss.backward()

8. optimizer.step()

9. Track training loss.

Validation:

1. model.eval()

2. torch.no_grad()

3. Run validation images.

4. Compute validation loss.

5. Compute useful segmentation metrics using EXISTING metrics implementation.

Do not duplicate metric formulas if existing evaluation functions can be reused.

---

# 7. OPTIMIZER

Use a simple reliable optimizer.

Default:

    Adam

Suggested initial learning rate:

    1e-4

Do not introduce complicated schedulers unless the existing config/project already requires them.

Make hyperparameters configurable.

At minimum:

- learning_rate
- weight_decay
- batch_size
- epochs
- num_workers
- device
- checkpoint_dir
- seed

---

# 8. CHECKPOINT

Save the best model based on validation performance.

Preferred checkpoint:

    checkpoints/unet_cityscapes.pth

The checkpoint must contain enough information to reproduce/load the trained model safely.

Prefer a structure such as:

    {
        "model_state_dict": ...,
        "optimizer_state_dict": ...,
        "epoch": ...,
        "val_loss": ...,
        "val_miou": ...,
        "config": ...
    }

If the existing inference loader expects a different checkpoint structure, inspect it and make the checkpoint compatible with the EXISTING inference implementation.

This is important:

The resulting checkpoint must be loadable by:

    models/unet/inference.py

Do not create a checkpoint format that inference cannot load.

---

# 9. BEST MODEL SELECTION

Primary validation metric:

    mIoU

Use the existing segmentation metric implementation.

Save the checkpoint whenever:

    current_val_miou > best_val_miou

Also record:

- best epoch
- best validation loss
- best validation mIoU

Print progress clearly, for example:

Epoch 1/20
Train Loss: ...
Val Loss: ...
Val mIoU: ...

Best checkpoint saved.

Do not claim a model is good merely because training completes.

---

# 10. CONFIGURATION

Update the existing U-Net configuration only if necessary.

Prefer something similar to:

unet:
  num_classes: 19

training:
  epochs: 20
  batch_size: 4
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_workers: 2
  seed: 42

checkpoint:
  dir: checkpoints
  filename: unet_cityscapes.pth

Use the project's existing YAML structure/conventions instead of blindly replacing it.

Do not hard-code all training parameters into Python.

If GPU is unavailable, training must be able to use CPU.

However, do not silently pretend CPU training is fast.

---

# 11. DEVICE

Reuse the existing device utility if available.

Expected behavior:

- CUDA when explicitly/configured and available
- otherwise CPU

Do not require CUDA.

Do not add GPU-specific assumptions.

---

# 12. RANDOM SEED

Reuse:

    utils/seed.py

or the existing seed implementation.

Set deterministic/reproducible seeds where appropriate.

Default seed:

    42

Training should report the seed being used.

---

# 13. DATASET

Use the EXISTING:

    CityscapesDataset

from:

    preprocessing/cityscapes.py

Do not create another Cityscapes dataset loader.

Use existing train/validation split logic.

Expected data:

    data/cityscapes/images/leftImg8bit
    data/cityscapes/labels/gtFine

Current dataset status:

- train images: 2975
- train labels: 2975
- val images: 500
- val labels: 500

Do not download Cityscapes.

If the dataset path is missing, fail with a clear error.

Example:

    Cityscapes dataset not found at ...

Do not silently generate fake training data.

---

# 14. DATA AUGMENTATION

Reuse existing preprocessing/transforms.

Do not introduce a completely separate preprocessing pipeline.

If the current Cityscapes preprocessing already provides train/validation transforms, use them.

If resizing/cropping is necessary for practical training, follow the existing project's preprocessing conventions.

Important:

Image and semantic mask must receive spatially consistent transformations.

Never independently resize the mask using bilinear interpolation.

Semantic masks must use nearest-neighbor interpolation when resized.

---

# 15. LOSS

Use:

    torch.nn.CrossEntropyLoss(ignore_index=255)

The model outputs raw logits.

Do NOT:

- apply softmax before CrossEntropyLoss
- convert target to one-hot unnecessarily
- treat 255 as a real class
- include ignored pixels in the loss

---

# 16. METRICS

During validation, calculate at least:

- Pixel Accuracy
- mIoU

Reuse:

    evaluation/segmentation_metrics.py

Do not duplicate IoU/mIoU formulas inside trainer.py.

If existing evaluation API requires a particular format, adapt to it.

The purpose is to make training validation consistent with Step 06.

---

# 17. TRAINING HISTORY

Record per-epoch:

- epoch
- train_loss
- val_loss
- val_pixel_accuracy
- val_miou

Prefer saving:

    outputs/analysis/unet_training_history.json

or another existing project output convention.

JSON must contain only JSON-serializable values.

Do not dump tensors or NumPy arrays directly.

Example structure:

{
    "epochs": [
        {
            "epoch": 1,
            "train_loss": ...,
            "val_loss": ...,
            "val_pixel_accuracy": ...,
            "val_miou": ...
        }
    ],
    "best_epoch": ...,
    "best_val_miou": ...
}

---

# 18. CLI

Create a training CLI entry point.

Preferred usage:

    python -m training.train_unet

Support arguments such as:

    --config configs/unet.yaml
    --epochs
    --batch-size
    --learning-rate
    --num-workers
    --device
    --output-dir
    --seed

Arguments should override config values.

Do not change the existing Step 11 inference CLI behavior unnecessarily.

If the project prefers using:

    python main.py train

inspect existing main.py first and follow the existing architecture.

Do NOT break:

    python main.py --image path/to/image.jpg

---

# 19. CHECKPOINT RESUME

If practical within the existing architecture, support:

    --resume path/to/checkpoint.pth

Resume:

- model state
- optimizer state
- epoch
- best metric

If implementing resume would require invasive changes, do not over-engineer it. Report it as not implemented.

Do not make resume mandatory.

---

# 20. SMALL OFFLINE / UNIT TESTS

Training tests MUST NOT train the full Cityscapes dataset.

Tests must:

- use tiny synthetic tensors/datasets
- use a tiny number of samples
- use CPU
- avoid internet
- avoid CUDA requirement
- avoid real Cityscapes files
- avoid multi-minute training

Create:

    tests/test_training.py

Test at minimum:

1. trainer construction
2. one tiny training epoch
3. validation works
4. CrossEntropyLoss ignore_index=255
5. metrics are returned
6. checkpoint is created
7. checkpoint contains required fields
8. best checkpoint logic
9. training history is serializable
10. deterministic behavior where practical
11. missing dataset/config error handling if applicable
12. checkpoint can be loaded by existing U-Net inference loader if practical

Use a tiny deterministic model or the existing U-Net with very small spatial inputs where feasible.

Do NOT mock away the entire training loop.

The test should actually execute at least one tiny optimization step.

---

# 21. REAL TRAINING SCRIPT SAFETY

The real training command must NEVER:

- download Cityscapes
- download MiDaS
- download pretrained weights
- fabricate labels
- fabricate a checkpoint
- silently use random weights as a completed model
- overwrite an existing best checkpoint without clear behavior

If checkpoint already exists, choose a safe behavior:

- require explicit overwrite flag
OR
- save a timestamped/new run directory

Do not silently destroy a previous trained checkpoint.

---

# 22. LOGGING

Training output should clearly show:

- device
- dataset sizes
- number of classes
- batch size
- learning rate
- epochs
- seed
- model parameter count if easy
- epoch progress
- train loss
- validation loss
- validation mIoU
- best epoch
- checkpoint path

Do not produce excessively noisy per-pixel/per-batch output.

---

# 23. MEMORY / PERFORMANCE

Cityscapes is large.

Do not load the entire dataset into RAM.

Use DataLoader streaming.

Use:

    pin_memory=True

only when appropriate for CUDA.

Use:

    persistent_workers=True

only when num_workers > 0 and compatible with the environment.

Do not introduce unnecessary caching.

If the current image resolution causes excessive GPU memory usage, make image size/batch size configurable.

Do not silently reduce resolution without documenting/configuring it.

---

# 24. IMPORTANT: DO NOT CHANGE MODEL ARCHITECTURE

The U-Net architecture from Step 04 is already tested.

Do not change:

- encoder
- decoder
- skip connections
- channel sizes
- bottleneck
- final layer
- logits behavior

unless an actual training compatibility bug is discovered.

If a change is absolutely necessary:

1. explain why
2. make the smallest change
3. run all regression tests

---

# 25. REGRESSION REQUIREMENT

After implementation run:

    pytest

All existing Step 02 → Step 11 tests must continue passing.

Previous baseline:

    332 passed, 1 skipped

Step 12 should add training tests without breaking the existing tests.

Report:

    X passed, Y skipped

Do not hide failures.

---

# 26. REAL DATA SMOKE TEST

Before declaring Step 12 complete, perform a SAFE real-data smoke test.

Do NOT necessarily run the full 20-epoch Cityscapes training if it is impractical.

At minimum verify:

1. Cityscapes train loader opens real data.
2. Cityscapes validation loader opens real data.
3. One real batch can pass through U-Net.
4. Loss can be computed.
5. Backward pass works.
6. Optimizer step works.
7. Validation can run on a small real subset if practical.

If you perform a short real training run, clearly report:

- number of epochs
- number of training samples/batches
- batch size
- device
- training time
- final validation loss
- final validation mIoU
- checkpoint path

Do NOT call a short smoke test the final trained model.

---

# 27. FULL TRAINING

After all implementation/tests are successful, provide the command for the user to run the actual training.

Example:

    python -m training.train_unet --config configs/unet.yaml

The actual full training should be user-controlled.

Do not automatically start a long multi-hour training job unless explicitly requested.

The implementation step should finish with the training command ready.

---

# 28. INFERENCE COMPATIBILITY TEST

After creating a real checkpoint, if a real checkpoint is available during this step:

Verify:

    checkpoint
        ↓
    existing UNetInference
        ↓
    segmentation prediction
        ↓
    existing SceneUnderstandingPipeline

Do not create a second inference implementation.

If no real checkpoint was generated because only the implementation/unit tests were run, explicitly state that real inference compatibility still needs to be verified after training.

---

# 29. OUTPUTS

Expected possible outputs:

training/
├── __init__.py
├── trainer.py
└── train_unet.py

tests/
└── test_training.py

checkpoints/
└── unet_cityscapes.pth

outputs/
└── analysis/
    └── unet_training_history.json

Modify existing config only when necessary.

Do not create unrelated folders.

---

# 30. FINAL REPORT

At the end, report exactly:

## Files created
- ...

## Files modified
- ...

## Training architecture
Explain:

Cityscapes
→ Dataset
→ DataLoader
→ U-Net
→ CrossEntropyLoss
→ Backpropagation
→ Validation
→ mIoU
→ Best checkpoint

## Hyperparameters
- epochs
- batch size
- learning rate
- optimizer
- weight decay
- num workers
- seed
- device

## Checkpoint
- path
- format
- compatible with existing U-Net inference: yes/no

## Tests
- training tests:
- full pytest:
- passed:
- skipped:
- failed:

Previous baseline was:

    332 passed, 1 skipped

## Real-data smoke test
Report exactly what was actually executed.

## Full training status
Clearly state:

- implementation ready
- full training not started
OR
- full training completed

Do NOT claim full training was completed unless it actually ran.

## Remaining issues
List only real remaining issues.

---

# 31. STOP CONDITION

This is STEP 12 ONLY.

Do not proceed to:

- MiDaS training
- pipeline optimization
- final evaluation
- difficulty analysis improvements
- deployment
- CI/CD
- cloud deployment
- report writing

STOP after Step 12.

The next step will be handled separately.