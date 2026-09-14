# Step 04 — U-Net Semantic Segmentation Model

## Objective

Implement the U-Net semantic segmentation model for the scene-understanding project.

This step focuses ONLY on the U-Net model architecture and its unit tests.

Do NOT implement inference, training loops, evaluation metrics, MiDaS, fusion, visualization, or the full pipeline.

---

## 1. Read Existing Specifications

Before implementing anything, read:

- prompts/01_architecture.md
- prompts/02_config_utils.md
- prompts/03_dataset.md

Also inspect the current implementation of:

- models/
- preprocessing/
- configs/
- utils/
- tests/

Reuse the existing project architecture and utilities.

Do NOT redesign the project.

Do NOT unnecessarily rename, delete, or move existing files.

---

# 2. U-Net Requirements

Implement a standard U-Net suitable for semantic segmentation.

Expected structure:

Input RGB image
        ↓
Encoder
        ↓
Bottleneck
        ↓
Decoder
        ↓
1×1 convolution
        ↓
Class logits

The model must contain:

### Encoder

Multiple convolution blocks with spatial downsampling.

Each encoder stage should:

- process image features using convolution layers
- increase feature channels
- reduce spatial resolution using pooling/downsampling

### Bottleneck

The deepest representation of the image.

It should have:

- the highest feature-channel dimension
- the lowest spatial resolution

### Decoder

Multiple upsampling stages.

Each decoder stage should:

1. Upsample the feature map.
2. Match its spatial dimensions with the corresponding encoder feature map.
3. Concatenate the encoder feature map using a skip connection.
4. Apply a convolution block.

The skip connection must preserve spatial information lost during downsampling.

### Output layer

Use a `1×1` convolution to produce:

[B, num_classes, H, W]

where:

- B = batch size
- num_classes = number of semantic classes
- H = image height
- W = image width

Do NOT apply softmax inside the model.

The model must return raw logits.

---

# 3. Cityscapes Configuration

The project uses the standard 19 trainable Cityscapes classes.

Default:

```yaml
num_classes: 19
```
The implementation must obtain num_classes from the existing configuration system where appropriate.

Do NOT hard-code 19 throughout the model implementation.

The model should remain reusable for another number of classes.

# 4. Input Requirements

The model should accept tensors in PyTorch image format:

[B, 3, H, W]

Example:

[B, 3, 256, 512]

The implementation must support arbitrary reasonable spatial dimensions where possible.

Do not assume that only one fixed image resolution will ever be used.

# 5. Architecture Design

Implement reusable components rather than writing the entire network as one large forward function.

At minimum, use reusable blocks for:

convolution block
encoder/downsampling stage
decoder/upsampling stage

A convolution block may use the standard pattern:

Conv2d
↓
BatchNorm2d
↓
ReLU
↓
Conv2d
↓
BatchNorm2d
↓
ReLU

Keep the implementation simple and readable.

Avoid unnecessary architectural complexity.

Do NOT add attention mechanisms, transformers, ASPP, residual blocks, or other advanced modules unless already required by prompts/01_architecture.md.

This is a baseline U-Net implementation.

# 6. Spatial Dimension Handling

The decoder must correctly handle possible spatial-size differences between:

upsampled decoder features
encoder skip features

Do not assume that every input size is perfectly divisible by all pooling factors.

Implement safe spatial alignment before concatenation.

The final output must have the same spatial height and width as the input whenever possible.

Expected:

Input:

[B, 3, H, W]

Output:

[B, num_classes, H, W]

# 7. Device Compatibility

The model must work on:

CPU
CUDA when available

Do not hard-code .cuda() inside the model.

The model should follow normal PyTorch device handling.

Use the existing utility/config system where appropriate.

# 8. Model API

Implement the model in:

models/unet/model.py

Provide a clean public API.

For example:

model = UNet(num_classes=19)
output = model(x)

The exact internal class names may follow existing project conventions if already present.

Do not create a second configuration system.

# 9. Configuration

Inspect:

configs/unet.yaml

If necessary, update it so that the U-Net architecture parameters are explicitly configurable.

Possible parameters include:

model:
  name: unet
  num_classes: 19
  in_channels: 3
  base_channels: 64

Use sensible defaults.

Do not add unnecessary hyperparameters.

Keep model architecture configuration separate from training hyperparameters.

Do NOT implement training in this step.

# 10. Tests

Implement tests in:

tests/test_unet.py

Tests must NOT require:

Cityscapes download
KITTI download
pretrained weights
GPU
internet connection

Use randomly generated tensors.

Test at minimum:

Test 1 — Model creation
model = UNet(num_classes=19)

must succeed.

Test 2 — Forward pass

Example:

x = torch.randn(2, 3, 128, 256)
y = model(x)

Expected:

y.shape == (2, 19, 128, 256)
Test 3 — Different number of classes

Verify that:

UNet(num_classes=5)

produces:

[B, 5, H, W]
Test 4 — Different image sizes

Test at least two different spatial resolutions.

For example:

128 × 256
256 × 512

The output spatial dimensions must match the input.

Test 5 — Batch size

Verify that batch size > 1 works.

Test 6 — Gradient flow

Run a simple loss:

loss = output.mean()
loss.backward()

Verify that gradients are produced.

This checks that the network is differentiable.

Test 7 — CPU compatibility

The model must run on CPU.

If CUDA is available, optionally test CUDA without making CUDA mandatory.

Test 8 — Output logits

Verify that the final output has:

[B, num_classes, H, W]

and that the model does NOT return a softmax probability tensor.

# 11. Existing Tests

All previous tests must continue to pass.

Run:

pytest

The expected result is that all existing Step 02 + Step 03 tests remain passing, plus the new U-Net tests.

# 12. Import Test

Run:

python -c "from models.unet.model import UNet; import torch; m=UNet(num_classes=19); x=torch.randn(1,3,128,256); y=m(x); print(y.shape)"

Expected:

torch.Size([1, 19, 128, 256])
# 13. Important Scope Restriction

STOP after implementing the U-Net model and its tests.

Do NOT implement:

U-Net training
training loop
optimizer
scheduler
checkpoint training
inference.py
segmentation metrics
Cityscapes training
MiDaS
depth estimation
fusion
scene analyzer
visualization
CI/CD
full pipeline

Those belong to later steps.

# 14. Report

After implementation, report:

Files created
Files modified
U-Net architecture
Encoder structure
Bottleneck
Decoder structure
Skip connections
Output tensor format
Config parameters
Tests implemented
Tests executed
Test results
Import/forward-pass result
Any assumptions or problems

Do not modify files after reporting.

STOP after Step 04.