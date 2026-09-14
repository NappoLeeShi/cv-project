# Step 03 — Dataset Loaders and Preprocessing

## Context

The project is:

Segmentation + Depth → Scene Understanding

Main components:

- Cityscapes → U-Net → Semantic Segmentation
- KITTI → Depth-related evaluation
- MiDaS → Monocular Depth Estimation
- Fusion → Scene Understanding

The overall architecture is defined in:

prompts/01_architecture.md

Step 02 has implemented:

- YAML configuration
- Config loader
- Random seed utility
- Logger utility
- Device utility
- Unit tests

Now implement the dataset and preprocessing layer.

Follow the existing architecture.

DO NOT redesign the project structure.

---

# Objective

Implement robust dataset loaders and preprocessing for:

1. Cityscapes
2. KITTI

The loaders must be designed for the real datasets from the beginning.

Do NOT create a separate "small dataset" implementation.

The same loader must work for:

- a small subset during development/testing
- the complete dataset during actual experiments

Use configuration to control how many samples are loaded.

Example:

```yaml
data:
  max_samples: 100
For the full dataset:

data:
  max_samples: null
```
Changing from a subset to the full dataset must NOT require changing the loader implementation.

Scope

Implement only:

Cityscapes dataset loader
KITTI dataset loader
Dataset preprocessing
Dataset configuration
Dataset validation
Dataset-related tests

Do NOT implement:

U-Net
MiDaS
Training
Fusion
Scene analyzer
Evaluation metrics
Visualization
Full pipeline
Model inference

Do NOT download the datasets automatically.

The user may download the official datasets separately.

# 1. Dataset Directory Structure

Follow the existing project architecture.

Expected structure:

data/
├── cityscapes/
│   ├── images/
│   └── labels/
│
└── kitti/
    ├── images/
    └── depth/

However, do NOT blindly assume that the downloaded archives have exactly this structure.

The preprocessing/loader code must document the expected directory layout and validate it.

If the actual official dataset structure differs, support it through configuration or a clear preprocessing path rather than silently assuming incorrect paths.

# 2. Cityscapes Dataset

Implement:

preprocessing/cityscapes.py

The loader must support:

RGB images
semantic segmentation labels
train split
validation split
configurable dataset root
configurable maximum number of samples

The loader should return a consistent sample representation, for example:

{
    "image": image,
    "label": label,
    "image_path": image_path,
    "label_path": label_path
}

Use an appropriate PyTorch Dataset interface if compatible with the existing architecture.

# 3. Cityscapes File Matching

Do NOT pair files simply by sorting two independent lists.

Use the Cityscapes filename structure to match each RGB image with its corresponding semantic label.

Validate that every selected image has the expected label.

If a label is missing:

raise a clear error, or
explicitly skip it with a warning

Do not silently create incorrect image-label pairs.

# 4. Cityscapes Label Mapping

This is important.

The Cityscapes dataset contains raw label IDs and training IDs.

The model will eventually use the standard semantic segmentation training representation.

The loader must distinguish between:

raw Cityscapes label IDs
train IDs

Do NOT use the class mapping written in an earlier architecture document as an unquestioned source of truth.

Before implementation, verify the mapping against the official Cityscapes label definitions or the official dataset metadata available in the project.

The standard 19 trainable classes are expected to be:

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

Use:

255

as the ignore label where appropriate for pixels that are not part of the 19 trainable classes.

Do not silently treat arbitrary raw label IDs as train IDs.

Document the mapping used by the implementation.

# 5. Cityscapes Image Preprocessing

Implement preprocessing suitable for later U-Net training/inference.

Requirements:

Load RGB image correctly
Convert to the expected tensor representation
Resize consistently when configured
Normalize image values consistently
Preserve segmentation label semantics during resizing

IMPORTANT:

For RGB images:

bilinear interpolation or another appropriate continuous interpolation may be used.

For segmentation masks:

NEVER use bilinear interpolation.

Use nearest-neighbor interpolation for labels.

Otherwise class IDs will become invalid fractional values.

# 6. Cityscapes Dataset Configuration

Add Cityscapes-related parameters to the appropriate configuration.

Example:

data:
  root: data/cityscapes
  split: train
  image_size: [512, 1024]
  max_samples: null

Support:

train
val

Do not assume test labels are available for local supervised evaluation.

The implementation should make the split configurable.

# 7. KITTI Dataset

Implement:

preprocessing/kitti.py

The loader must support:

RGB image
depth ground truth
configurable dataset root
configurable split
configurable maximum number of samples

Return a consistent sample representation, for example:

{
    "image": image,
    "depth": depth,
    "image_path": image_path,
    "depth_path": depth_path
}

Use an appropriate PyTorch Dataset interface if compatible with the existing architecture.

# 8. KITTI File Matching

Do NOT assume that RGB images and depth files can always be paired simply by sorting filenames.

Inspect the actual directory/file naming convention supported by the selected KITTI dataset subset.

Implement explicit and deterministic matching.

If the selected KITTI split requires a file list or official split definition, support that rather than inventing a pairing rule.

If the exact KITTI subset/split is not present locally, make the loader fail clearly with an informative message instead of silently producing incorrect samples.

# 9. KITTI Depth Representation

KITTI depth ground truth must be treated as depth data, not semantic labels.

The loader should:

load the depth values correctly
preserve invalid/missing depth information
expose a valid-depth mask when appropriate
avoid converting invalid pixels into valid depth values

A useful representation is:

{
    "image": image,
    "depth": depth,
    "valid_mask": valid_mask,
    "image_path": image_path,
    "depth_path": depth_path
}

Do not assume every pixel has valid depth.

Do not assume MiDaS output is already metric depth.

The relationship between MiDaS relative depth and KITTI metric depth will be handled later during the depth evaluation step.

# 10. KITTI Image Preprocessing

Implement preprocessing suitable for later MiDaS inference/evaluation.

Requirements:

RGB image loading
consistent tensor conversion
configurable resizing
preserve depth semantics

IMPORTANT:

If depth maps are resized:

use an interpolation method appropriate for continuous depth values
preserve invalid-depth information
do not use nearest-neighbor blindly for continuous depth unless there is a specific reason

Document the chosen method.

# 11. Do Not Couple Cityscapes and KITTI

Cityscapes and KITTI are separate datasets.

Do NOT assume:

Cityscapes image #1 == KITTI image #1

Do NOT create artificial pairs between the datasets.

Their roles are:

Cityscapes
    ↓
RGB + semantic labels
    ↓
Segmentation task


KITTI
    ↓
RGB + depth ground truth
    ↓
Depth evaluation

The actual segmentation and depth models will later be applied independently.

Fusion will later operate on predictions generated from the same RGB image when such an image is available.

# 12. Dataset Validation

Implement validation utilities that check:

dataset root exists
required directories exist
files are readable
image/label matching is valid
image/depth matching is valid
unsupported/malformed samples produce clear errors
split is valid
max_samples is valid

Errors should be informative.

Example:

Cityscapes dataset not found:
data/cityscapes
Expected image directory:
...

Avoid cryptic errors such as:

IndexError

when a clearer dataset error can be provided.

# 13. max_samples Behavior

Implement max_samples as a configuration-level development convenience.

Examples:

max_samples: 10

means:

Load at most 10 valid samples

while:

max_samples: null

means:

Load all valid samples

Important:

The implementation must use the exact same dataset class and preprocessing code in both cases.

Do NOT create:

SmallCityscapesDataset
FullCityscapesDataset

or similar duplicate implementations.

# 14. Deterministic Ordering

Dataset file discovery should be deterministic.

Sort file paths or otherwise use a deterministic ordering before applying max_samples.

This ensures:

max_samples: 10

selects the same samples between runs when the dataset contents are unchanged.

Do not use random selection unless explicitly configured later.

# 15. Dataset Tests

Create tests for the dataset/preprocessing layer.

The tests must NOT require downloading the complete Cityscapes or KITTI datasets.

Use temporary directories and small synthetic fixtures where possible.

Test at least:

Cityscapes
valid dataset structure
image discovery
image-label matching
invalid/missing label handling
train/validation split handling
max_samples
deterministic ordering
label conversion/mapping
nearest-neighbor mask resizing if resizing is implemented
KITTI
valid dataset structure
image-depth matching
invalid/missing depth handling
max_samples
deterministic ordering
valid depth mask
depth resizing behavior

The tests should verify actual behavior rather than only checking that functions can be imported.

# 16. Real Dataset Compatibility

The implementation must be designed for the real official datasets.

Do not invent fake directory structures and then claim that the loader supports Cityscapes or KITTI.

If the exact official dataset structure cannot be verified because the dataset has not been downloaded yet:

Implement the loader around the documented expected structure.
Clearly document the assumption.
Make the code validate the structure.
Do not silently guess file locations.
Make it easy to update the configured root/split later.

Do NOT download datasets automatically.

# 17. Configuration Integration

Use the configuration system implemented in Step 02.

Do not hardcode paths such as:

/home/user/...

Dataset paths must come from configuration.

For example:

data:
  cityscapes:
    root: data/cityscapes
    split: train
    max_samples: null

  kitti:
    root: data/kitti
    split: ...
    max_samples: null

Adapt this structure if a cleaner design already exists in Step 02.

Do not duplicate configuration logic.

# 18. Documentation

Document:

expected Cityscapes directory structure
expected KITTI directory structure
how to configure dataset paths
how to select train/validation split
how max_samples works
Cityscapes trainId mapping
KITTI valid-depth handling
how to run dataset tests

Keep documentation suitable for a coursework project.

# 19. Requirements

Only add packages that are actually needed.

Prefer the packages already used by the project:

Python
PyTorch
NumPy
Pillow
PyYAML
pytest

Do not introduce unnecessary dataset frameworks.

# 20. Validation

Run:

pytest

All existing Step 02 tests must continue to pass.

The new dataset tests must also pass.

If the real datasets are not downloaded yet, tests must still pass using synthetic fixtures.

Also perform a simple import check:

python -c "from preprocessing.cityscapes import *; from preprocessing.kitti import *; print('Dataset modules OK')"

Do not download any dataset during this step.

# 21. Important Architecture Constraints

Follow:

utils
  ↓
preprocessing
  ↓
models
  ↓
scene_understanding
  ↓
evaluation / visualization
  ↓
main.py

The dataset layer must NOT import:

U-Net
MiDaS
Fusion
Scene Analyzer
Evaluation modules

Keep dependencies one-directional.

# 22. Expected Output

After implementation, report:

Files created
Files modified
Cityscapes loader design
KITTI loader design
Preprocessing decisions
Cityscapes label mapping used
KITTI split/structure assumptions
How max_samples works
Tests executed
Test results
Any issues or assumptions

Do NOT continue to U-Net or MiDaS.

STOP after Step 03.