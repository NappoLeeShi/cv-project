# Step 05 — U-Net Inference

## Objective

Implement the U-Net inference pipeline for semantic segmentation.

This step connects:

Cityscapes RGB image
        ↓
Preprocessing
        ↓
Trained U-Net checkpoint
        ↓
Inference
        ↓
Semantic segmentation prediction
        ↓
Prediction visualization/output

Focus ONLY on U-Net inference.

Do NOT implement segmentation metrics, MiDaS, depth estimation, fusion, scene understanding, or training.

---

## 1. Read Existing Specifications

Before making changes:

1. Read prompts/01_architecture.md.
2. Read prompts/02_config_utils.md.
3. Read prompts/03_dataset.md.
4. Read prompts/04_unet.md.
5. Inspect the existing implementation of:
   - models/unet/model.py
   - preprocessing/cityscapes.py
   - preprocessing/transforms.py
   - configs/unet.yaml
   - configs/pipeline.yaml
   - utils/

Reuse the existing architecture and utilities.

Do NOT redesign the project.

Do NOT unnecessarily rename, delete, or move existing files.

---

# 2. Scope

Implement:

- models/unet/inference.py
- tests for U-Net inference
- necessary inference configuration

Do NOT implement:

- U-Net training
- training loop
- optimizer
- scheduler
- segmentation metrics
- MiDaS
- depth evaluation
- fusion
- scene analyzer
- full pipeline
- CI/CD

---

# 3. Inference Requirements

The inference module must provide a clean API for running U-Net on an RGB image.

The basic flow is:

RGB image
↓
Preprocess
↓
Tensor [B, 3, H, W]
↓
U-Net
↓
Logits [B, C, H, W]
↓
argmax over C
↓
Predicted class map [H, W]

For a single image:

Input:

[H, W, 3]

Output:

[H, W]

where every pixel contains a semantic class index.

For Cityscapes:

C = 19

and valid predicted class IDs are:

0 ... 18

---

# 4. Model Loading

Inference must be able to load a trained U-Net checkpoint.

Do not assume that a checkpoint is always available.

The implementation should support:

1. Creating a model without a checkpoint for testing.
2. Loading a checkpoint when a checkpoint path is provided.

The checkpoint loading API should be robust to common checkpoint formats, such as:

### Format A

```python
torch.save(model.state_dict(), path)
``` 

Format B
torch.save({
    "model_state_dict": model.state_dict(),
    ...
}, path)

If supporting both formats is practical, implement both.

Do NOT silently ignore incompatible checkpoints.

If the checkpoint cannot be loaded, raise a clear informative error.

Use:

model.eval()

for inference.

Do not modify model parameters during inference.

# 5. No Gradient During Inference

Inference must use:

torch.no_grad()

or an equivalent inference context.

There must be no unnecessary gradient computation.

# 6. Device Handling

Support:

CPU
CUDA when available

Do not hard-code:

.cuda()

Use the existing device utility/configuration from Step 02.

If CUDA is requested but unavailable, produce a clear error or follow the existing project device policy.

Do not automatically download anything.

# 7. Preprocessing

Reuse the existing preprocessing utilities where appropriate.

The preprocessing must:

convert RGB image to tensor
convert layout from HWC to CHW
add batch dimension
resize according to configured inference image size when required
apply the project's configured image normalization

Do not create a second independent normalization system.

Important:

The semantic label preprocessing from Step 03 must NOT be used for the RGB input.

RGB image interpolation may use bilinear interpolation.

# 8. Prediction

The model produces:

[B, 19, H, W]

Convert logits to class prediction using:

prediction = logits.argmax(dim=1)

For a single image, remove the batch dimension:

[1, 19, H, W]
→
[H, W]

The returned prediction must contain integer class IDs.

Do NOT apply thresholding.

Do NOT use sigmoid.

Do NOT use softmax for obtaining the class index.

Argmax over class logits is sufficient.

# 9. Output Resolution

If the model output resolution differs from the original RGB image resolution because preprocessing resized the image, resize the predicted class map back to the original image dimensions.

IMPORTANT:

Segmentation predictions must be resized using:

NEAREST NEIGHBOR

Never use bilinear interpolation for class IDs.

The final prediction for an input image of:

H × W

must be:

H × W
# 10. Confidence

If practical, expose an optional confidence output.

Confidence can be calculated from the model logits using softmax:

probabilities = torch.softmax(logits, dim=1)
confidence = probabilities.max(dim=1).values

Important:

softmax may be used for confidence calculation
softmax must NOT be used as the primary class prediction mechanism
class prediction remains argmax(logits)

If confidence is implemented, document its meaning clearly.

Do not make confidence unnecessarily complicated.

# 11. API Design

Provide a clean inference interface.

For example:

inferencer = UNetInference(
    model=model,
    device=device,
)

prediction = inferencer.predict(image)

or an equivalent clean design consistent with the existing codebase.

The API should make it easy for later pipeline code to call U-Net inference.

Avoid coupling the inference class directly to Cityscapes-specific filesystem logic.

The inference module should work with an RGB image array/PIL image/tensor according to the existing preprocessing conventions.

# 12. Batch Inference

If practical, support batch inference.

Expected input:

[B, 3, H, W]

Expected prediction:

[B, H, W]

Single-image inference may internally use a batch size of 1.

Do not implement a complicated data loader here.

Dataset loading remains the responsibility of preprocessing/dataset modules.

# 13. Label Mapping

The U-Net output uses Cityscapes train IDs:

0 ... 18

Do not convert predictions back to raw Cityscapes labelIds in the core inference function unless an existing project API explicitly requires it.

Keep the prediction representation as train IDs.

The official mapping should remain consistent with Step 03.

# 14. Checkpoint Safety

The inference implementation must validate the loaded model/checkpoint sufficiently to detect incompatible configurations.

For example, a checkpoint trained with:

num_classes = 19

should not silently load into:

num_classes = 5

unless explicitly handled.

Prefer clear errors over silent partial loading.

Do not use:

strict=False

just to suppress checkpoint errors.

# 15. Tests

Create/update:

tests/test_unet_inference.py

Tests must not require:

Cityscapes
KITTI
pretrained weights
internet
GPU

Use randomly initialized U-Net models and synthetic RGB images.

Test at minimum:

Test 1 — Inference object creation

Create a U-Net and inference wrapper successfully.

Test 2 — Single image inference

Use a synthetic image.

Example:

256 × 512 × 3

Expected prediction:

256 × 512
Test 3 — Class range

Verify every prediction is within:

0 ... num_classes - 1

For Cityscapes:

0 ... 18
Test 4 — Different image sizes

Test at least:

128 × 256
256 × 512
100 × 150

The final prediction must match the original image dimensions.

Test 5 — Batch inference

Input:

[B, 3, H, W]

Output:

[B, H, W]
Test 6 — No gradient

Verify that inference does not create gradients.

Test 7 — Eval mode

Verify the model is placed into evaluation mode.

Test 8 — Checkpoint state_dict

Save a synthetic model state_dict to a temporary file.

Load it through the inference API.

Verify inference succeeds.

Test 9 — Dictionary checkpoint

If dictionary checkpoint support is implemented, test:

{
    "model_state_dict": model.state_dict()
}
Test 10 — Invalid checkpoint

Provide an incompatible/corrupted checkpoint and verify that a clear error is raised.

Test 11 — Nearest-neighbor output resize

Use an input whose dimensions differ from the model preprocessing dimensions.

Verify the final class map uses discrete class IDs and retains the original spatial dimensions.

Test 12 — Raw prediction

Verify that the returned segmentation result is integer class IDs rather than probability values.

# 16. Existing Tests

All existing tests must continue to pass.

Run:

pytest

Do not remove or weaken existing tests.

The expected result is:

all Step 02 tests pass
all Step 03 tests pass
all Step 04 tests pass
all new inference tests pass
# 17. Import Test

Run:

python -c "from models.unet.model import UNet; from models.unet.inference import *; print('U-Net inference modules OK')"

It must complete without errors.

Also run a real forward/inference smoke test using a random tensor/image.

# 18. Important Dataset Constraint

Do NOT download Cityscapes.

Do NOT require a real Cityscapes dataset to run the tests.

The inference implementation should be compatible with the real dataset later.

Do not create fake Cityscapes labels just to demonstrate model quality.

The current model is randomly initialized unless a trained checkpoint is supplied.

Therefore:

IMPORTANT:

Do NOT claim that random U-Net predictions are meaningful semantic segmentation.

This step only verifies that the inference pipeline works technically.

Actual segmentation quality will be evaluated later after training/loading a meaningful checkpoint.

# 19. Report

After implementation, report:

Files created
Files modified
Inference API
Preprocessing flow
Model loading/checkpoint handling
Device handling
Prediction conversion
Output resizing
Confidence handling, if implemented
Tests implemented
Tests executed
Test results
Import/smoke-test result
Problems or assumptions

STOP after Step 05.

Do NOT continue to Step 06.