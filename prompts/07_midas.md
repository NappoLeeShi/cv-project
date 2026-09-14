# Step 07 — MiDaS Monocular Depth Estimation

## Objective

Implement MiDaS monocular depth estimation for the scene-understanding project.

This step implements:

RGB image
    ↓
MiDaS
    ↓
Relative depth prediction

Focus ONLY on MiDaS model integration and inference.

Do NOT implement:

- KITTI depth metrics
- depth evaluation
- segmentation
- U-Net
- fusion
- scene understanding
- visualization
- full pipeline
- CI/CD

---

# 1. Read Existing Specifications

Before making changes:

1. Read prompts/01_architecture.md.
2. Read prompts/02_config_utils.md.
3. Read prompts/03_dataset.md.
4. Read prompts/04_unet.md.
5. Read prompts/05_unet_inference.md.
6. Read prompts/06_segmentation_evaluation.md.

Inspect:

- models/midas/
- preprocessing/
- configs/midas.yaml
- configs/pipeline.yaml
- utils/
- tests/

Reuse existing project utilities and conventions.

Do NOT redesign the project.

Do NOT unnecessarily rename, delete, or move existing files.

---

# 2. Important MiDaS Depth Convention

MiDaS is a monocular depth estimation model.

The standard MiDaS prediction should be treated as:

RELATIVE DEPTH

not directly as metric depth in meters.

Do NOT claim:

"MiDaS output = meters"

Do NOT convert the raw MiDaS output to meters using an arbitrary scale.

Preserve the raw model prediction.

The exact direction/convention of the raw prediction must be verified from the selected MiDaS implementation rather than assumed blindly.

Document clearly how the implementation represents the raw prediction.

Do not implement KITTI scale alignment in this step.

That belongs to Step 08.

---

# 3. Implementation Files

Implement:

```text
models/midas/model.py
models/midas/inference.py
```
Create/update:

tests/test_midas.py

Update if necessary:

configs/midas.yaml

Keep the existing project structure.

# 4. Model Integration

Use an established MiDaS implementation rather than implementing the complete MiDaS neural architecture manually.

The implementation may use an existing PyTorch/Torch Hub/compatible MiDaS model source if appropriate.

However:

IMPORTANT:

Do not automatically download large model weights during unit tests.
Do not make tests depend on internet access.
Do not require external network access for the test suite.
Do not make the whole test suite fail because pretrained weights are unavailable.

Separate model construction from weight loading where practical.

The project must support a local/preloaded model for tests.

# 5. Model API

Provide a clean model API.

For example:

model = MiDaSModel(...)
depth = model(x)

or another equivalent API consistent with the project.

The model wrapper should clearly define:

model name/type
device
input requirements
output representation

Do not expose unnecessary implementation details to the rest of the project.

# 6. Supported MiDaS Variants

Make the model name/configuration explicit.

For example:

model:
  name: midas
  variant: DPT_Hybrid

If the existing architecture specification specifies another variant, follow that specification.

Do not add support for many variants unnecessarily.

One stable variant is sufficient for the coursework baseline.

# 7. Configuration

Inspect:

configs/midas.yaml

Use the existing configuration system.

A reasonable configuration structure may contain:

model:
  name: midas
  variant: DPT_Hybrid

inference:
  device: auto
  image_size: null
  normalize: true

depth:
  representation: relative

Do not add arbitrary training parameters.

Do not claim a metric depth scale.

# 8. Input Preprocessing

MiDaS requires model-specific preprocessing.

Use the preprocessing appropriate for the selected MiDaS implementation.

The preprocessing must:

accept RGB input
convert it into the expected tensor format
perform the required resize/normalization
preserve RGB semantics
avoid using semantic segmentation label preprocessing

Accepted input forms should be similar to U-Net inference where practical:

PIL RGB image
HWC NumPy RGB array
suitable PyTorch tensor

Do not create a completely unrelated preprocessing framework.

If the selected MiDaS implementation already provides official transforms, prefer those transforms rather than manually duplicating them.

# 9. Inference

Inference must use:

model.eval()

and:

torch.no_grad()

or an equivalent inference context.

The model must not accumulate gradients during prediction.

The basic flow is:

RGB image
↓
MiDaS preprocessing
↓
MiDaS model
↓
raw relative depth
↓
restore spatial dimensions if necessary
↓
depth prediction

# 10. Output

For a single image, return a 2D depth map:

[H, W]

For batch inference:

[B, H, W]

The output should use a floating-point dtype.

Do NOT return class IDs.

Do NOT threshold the output.

Do NOT convert the output to integer values.

Do NOT convert to meters.

# 11. Output Resolution

MiDaS may operate at a different internal resolution.

The final depth prediction for an input image:

H × W

must be returned at:

H × W

Use appropriate continuous interpolation for depth.

Do NOT use nearest-neighbor interpolation as the default for continuous depth values.

Bilinear interpolation is acceptable for restoring continuous depth maps.

# 12. Depth Value Handling

Do not arbitrarily normalize the depth prediction to:

0 ... 1

inside the core model prediction unless the API explicitly labels it as a visualization/normalization operation.

The raw relative depth output should remain available.

If a helper for visualization normalization is implemented, keep it separate from the raw prediction.

Example:

raw relative depth
        ↓
optional visualization normalization

Do not confuse visualization normalization with metric depth calibration.

# 13. Device Handling

Support:

CPU
CUDA when available

Reuse:

utils/device.py

or the existing device utility.

Do NOT hard-code:

.cuda()

If CUDA is requested but unavailable, follow the existing device policy and provide a clear error.

# 14. Weight Loading

Provide a clear way to load pretrained MiDaS weights.

Possible approaches:

local checkpoint path
selected official model source
already initialized model

Do not force downloading weights during import.

Do not silently use random weights and pretend they are pretrained MiDaS weights.

If weights are unavailable, produce a clear informative error when real inference is requested.

# 15. Testing Strategy

Create/update:

tests/test_midas.py

Tests must NOT require:

internet
downloading MiDaS weights
KITTI
Cityscapes
GPU

Use a lightweight mock/stub or a small compatible test model where necessary to test the inference wrapper.

The tests should verify the inference pipeline/API rather than MiDaS's actual depth quality.

# 16. Required Tests

Implement at least:

Test 1 — Configuration

Verify MiDaS configuration can be loaded.

Test 2 — Model wrapper creation

Verify the MiDaS wrapper can be constructed in test mode without downloading weights.

Test 3 — Input preprocessing

Verify an RGB image can be converted into the expected model input.

Test at least:

PIL image
NumPy HWC image

if supported.

Test 4 — Single-image inference

Use a lightweight test model/stub.

Input:

128 × 256 × 3

Expected:

128 × 256
Test 5 — Different image sizes

Test:

128 × 256
256 × 512
100 × 150

Final depth output must match the original spatial dimensions.

Test 6 — Batch inference

If batch inference is supported:

[B, 3, H, W]

must produce:

[B, H, W]
Test 7 — Floating-point output

Verify depth prediction uses a floating-point dtype.

Test 8 — Relative depth representation

Verify that the implementation exposes the output as relative depth and does not claim the values are meters.

Test 9 — No gradient

Verify inference does not create gradients.

Test 10 — Evaluation mode

Verify the model is placed in evaluation mode before inference.

Test 11 — CPU compatibility

Verify inference works on CPU.

CUDA may be tested optionally when available.

Test 12 — Output resizing

Verify depth is restored to the original image dimensions using continuous interpolation.

Test 13 — No arbitrary meter conversion

Verify that the core inference output is not divided/multiplied by an arbitrary scale and labeled as meters.

Test 14 — Invalid input

Verify unsupported input types or invalid shapes raise clear errors.

# 17. Mocking / Test Isolation

Do NOT download MiDaS weights inside pytest.

If the real MiDaS implementation requires network access:

isolate that part from unit tests
use a mock/stub model
verify the wrapper logic independently

A separate optional smoke test may be provided for real pretrained MiDaS if useful, but it must NOT be required for:

pytest
# 18. Real Model Smoke Test

If pretrained weights are already available locally, provide a way to run a real MiDaS smoke test.

If weights are not available:

do not download them automatically
report that the real-model smoke test was not executed

Do not mark the entire implementation as failed because weights are absent.

# 19. Important Dataset Relationship

KITTI provides depth ground truth.

MiDaS produces a depth prediction.

However:

KITTI image + KITTI depth GT

is used later for depth evaluation.

Do NOT implement the evaluation here.

Do NOT assume MiDaS raw output is already in KITTI meters.

Step 08 will handle depth evaluation and any required alignment/calibration strategy.

# 20. Scope Restriction

Do NOT implement:

KITTI depth metrics
RMSE
MAE
AbsRel
δ accuracy
scale alignment
median scaling
segmentation
U-Net
fusion
scene analyzer
visualization pipeline
full pipeline
training
CI/CD

Only implement:

MiDaS model integration
+
MiDaS inference
+
tests
+
configuration
# 21. Existing Tests

Run:

pytest

All previous tests must continue to pass:

Step 02
Step 03
Step 04
Step 05
Step 06

Do not remove or weaken existing tests.

# 22. Import Test

Run:

python -c "from models.midas.model import *; from models.midas.inference import *; print('MiDaS modules OK')"

It must complete without errors.

If importing the real MiDaS backend would trigger network access, structure the implementation so importing the module itself does not download weights.

# 23. Report

After implementation, report:

Files created
Files modified
MiDaS variant used
Model integration approach
Weight-loading approach
Input preprocessing
Output representation
Relative-depth convention
Output resizing
Device handling
Test strategy
Tests implemented
Tests executed
Test results
Real-model smoke-test status
Problems or assumptions

STOP after Step 07.

Do NOT continue to Step 08.