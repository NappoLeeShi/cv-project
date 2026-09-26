# Step 02 — Config + Utils

## Context

The project is:

Segmentation + Depth → Scene Understanding

Main components:

- Cityscapes → U-Net → Semantic Segmentation
- KITTI → Depth evaluation
- MiDaS → Monocular Depth Estimation
- Fusion → Scene Understanding

The overall architecture is already defined in:

prompts/01_architecture.md

Follow that architecture.

DO NOT redesign the project structure.

---

# Objective

Implement the project's basic configuration and utility layer.

This step prepares the project for later implementation of:

- Dataset
- U-Net
- MiDaS
- Evaluation
- Fusion
- Full pipeline

Do NOT implement those components yet.

---

# Scope

Implement only:

1. YAML configuration
2. Configuration loader
3. Random seed utility
4. Logger utility
5. Device utility
6. Unit tests for these utilities

Do NOT implement:

- Cityscapes dataset
- KITTI dataset
- U-Net
- MiDaS
- Model training
- Fusion
- Scene analyzer
- Evaluation metrics
- Visualization
- Full pipeline

Do NOT download datasets or model weights.

---

# 1. Configuration Files

Create:

configs/
├── unet.yaml
├── midas.yaml
└── pipeline.yaml

## configs/unet.yaml

Include configuration for:

- number of classes
- image size
- batch size
- learning rate
- epochs
- optimizer
- loss
- checkpoint directory
- device
- seed

Example:

```yaml
model:
  num_classes: 19

data:
  image_size: [512, 1024]
  batch_size: 4

training:
  learning_rate: 0.001
  epochs: 60
  optimizer: adam
  loss: cross_entropy

checkpoint:
  directory: outputs/checkpoints/unet

system:
  device: auto
  seed: 42
```
Use the architecture specification as the source of truth.

configs/midas.yaml

Include:

model name
device
seed
depth representation
relevant input configuration

Example:

model:
  name: DPT_Large

depth:
  representation: relative

system:
  device: auto
  seed: 42

Important:

MiDaS produces relative depth.

Do NOT assume its output is directly measured in meters.

Do not implement depth conversion in this step.

configs/pipeline.yaml

Include basic configuration for:

input paths
output paths
device
seed
logging
visualization
fusion

Keep this configuration simple.

Do not add unnecessary parameters.

# 2. Configuration Loader

Create:

utils/config.py

Implement a simple YAML configuration loader.

Example usage:

from utils.config import load_config

config = load_config("configs/unet.yaml")

Requirements:

Load YAML files
Return a convenient Python representation
Check whether the file exists
Raise a clear error if the file does not exist
Raise a clear error for invalid YAML
Keep the implementation simple

Do not create a complicated configuration framework.

# 3. Random Seed

Create:

utils/seed.py

Implement:

set_seed(seed)

It should set seeds for:

Python random
NumPy
PyTorch
CUDA when available

The purpose is reproducibility.

Do not claim that complete determinism is guaranteed on every GPU/software environment.

# 4. Logger

Create:

utils/logger.py

Implement a reusable logger.

Example:

from utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Test message")

Requirements:

Console logging
Optional file logging
INFO / WARNING / ERROR
Clear format
Avoid duplicated handlers when get_logger() is called multiple times

Keep it lightweight.

Do not add an external logging framework.

# 5. Device Utility

Implement a small utility for selecting the device.

Expected behavior:

device = auto

       ↓

CUDA available?
   /        \
 yes        no
 ↓           ↓
cuda        cpu

The utility should:

support auto
support explicit cpu
support explicit cuda
gracefully fall back to CPU when appropriate
report the selected device

Do not hardcode a specific GPU.

# 6. Tests

Create tests for the utilities.

Use pytest.

Test at least:

Config
valid YAML loads correctly
missing file raises a clear error
invalid YAML raises a clear error
Seed

Verify that repeated calls with the same seed produce reproducible values for:

Python random
NumPy
PyTorch

If CUDA is available, test CUDA reproducibility where practical.

Logger

Verify:

logger can be created
calling get_logger multiple times does not duplicate handlers
Device

Verify:

CPU selection works
auto selection works
CUDA selection behaves correctly depending on CUDA availability

Tests must NOT require:

Cityscapes
KITTI
downloaded model weights
internet connection
# 7. Requirements

Check requirements.txt.

Add only packages required for this step if they are missing.

At minimum:

PyYAML
pytest

Do not unnecessarily add packages.

# 8. Documentation

Add short documentation explaining:

Why YAML is used
What config.py does
What seed.py does
What logger.py does
What the device utility does
How to run tests

Keep the explanation suitable for a coursework project.

# 9. Validation

Run:

pytest

Also test configuration loading:

python -c "from utils.config import load_config; print(load_config('configs/unet.yaml'))"

The project must work without downloading any dataset or model.

# 10. Important Constraints

DO NOT:

redesign the architecture
modify the folder structure unnecessarily
implement U-Net
implement MiDaS
implement datasets
train models
implement fusion
implement evaluation
download datasets
download model weights

This is only the foundational infrastructure step.

Expected Output

After completing Step 02, report:

Files created
Files modified
What each utility does
Example config usage
Tests executed
Test results
Any problems or assumptions

STOP after Step 02.

Do NOT automatically continue to Step 03.