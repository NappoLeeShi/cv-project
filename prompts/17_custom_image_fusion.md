# Step 17 — Custom Image Fusion Demo

## Objective

Implement a user-facing custom street-image demo for the existing scene-understanding pipeline.

The user should be able to place arbitrary RGB street-scene images into:

data/pipeline/images/

Then run the pipeline without specifying an individual image path.

The pipeline must process the images using the existing:

RGB image
→ U-Net semantic segmentation
→ MiDaS monocular depth
→ spatial alignment
→ output-level fusion
→ scene analysis
→ difficulty analysis
→ visualization
→ JSON scene report

This step is for DEMO / INFERENCE only.

Do NOT retrain models.
Do NOT modify model architectures.
Do NOT change evaluation metrics.
Do NOT change existing training or evaluation behavior.

---

# 1. Input Folder

Use this fixed input directory:

data/pipeline/images/

Example:

data/pipeline/images/
├── street1.jpg
├── street2.png
└── street3.jpeg

Supported image extensions should include common RGB formats such as:

- .jpg
- .jpeg
- .png
- .bmp
- .webp

The program should automatically discover supported images inside the folder.

Do not require the user to specify an individual image path.

---

# 2. Command Line Interface

The main usage should be:

python main.py --input-dir data/pipeline/images

Optional:

python main.py \
    --input-dir data/pipeline/images \
    --device auto

Optional testing limit:

python main.py \
    --input-dir data/pipeline/images \
    --limit 1

The `--limit` argument is only for limiting the number of input images processed during testing/demo.

If `--limit` is omitted, process all supported images in the input folder.

If the existing CLI already has useful options, preserve backward compatibility where possible.

Do not break existing Step 11 / Step 15 commands.

---

# 3. Model Checkpoints

Use the existing project checkpoints.

U-Net:

checkpoints/unet_cityscapes.pth

MiDaS:

checkpoints/dpt_large_384.pt

Do not download weights automatically.

If a checkpoint is missing, show a clear error message explaining which file is missing.

Reuse the existing model loading and inference implementations.

Do not create another implementation of U-Net or MiDaS.

---

# 4. Same-Image Contract

For every input image, the exact same RGB image must be passed to both models.

Conceptually:

image
 ├──→ U-Net → segmentation
 └──→ MiDaS → relative inverse depth

Then:

segmentation + depth
→ alignment
→ fusion
→ analyzer

Do not accidentally use different images, resized files, or different preprocessing sources between the two models.

The existing pipeline implementation should be reused whenever possible.

---

# 5. Image Validation

Before inference, validate every input image.

Handle:

- missing input directory
- empty input directory
- unsupported file extension
- corrupted image
- grayscale image
- invalid image shape

The demo should provide clear error messages.

RGB images should be accepted.

For grayscale images, either reject them clearly or convert them only if the existing project convention already supports this. Prefer preserving the project's existing behavior.

Do not silently process invalid files.

---

# 6. Pipeline

For each image:

1. Load RGB image.
2. Run U-Net semantic segmentation.
3. Run MiDaS depth prediction.
4. Align output spatial dimensions with the original image.
5. Run existing fusion logic.
6. Run existing scene analyzer.
7. Run existing difficulty analysis.
8. Save visualization outputs.
9. Save a JSON scene report.

Reuse existing modules:

- models/unet/
- models/midas/
- scene_understanding/
- visualization/
- evaluation/difficulty_analysis.py

Do not duplicate their logic inside `main.py`.

---

# 7. Output Directory

All custom-image demo results must be written to:

outputs/custom/

The program should automatically create this directory if it does not exist.

For an input:

data/pipeline/images/street1.jpg

produce:

outputs/custom/street1_original.png
outputs/custom/street1_segmentation.png
outputs/custom/street1_depth.png
outputs/custom/street1_fusion.png
outputs/custom/street1_overview.png
outputs/custom/street1_scene_report.json

For multiple images, repeat the same naming scheme using each input filename stem.

Example:

data/pipeline/images/
├── street1.jpg
└── street2.png

outputs/custom/
├── street1_original.png
├── street1_segmentation.png
├── street1_depth.png
├── street1_fusion.png
├── street1_overview.png
├── street1_scene_report.json
├── street2_original.png
├── street2_segmentation.png
├── street2_depth.png
├── street2_fusion.png
├── street2_overview.png
└── street2_scene_report.json

Do not overwrite outputs from different input filenames.

---

# 8. Visualization

Reuse the existing visualization functions.

Generate:

1. Original image
2. Semantic segmentation
3. Relative inverse depth
4. Fusion visualization
5. Full overview

The depth visualization must preserve the project's existing terminology:

"Relative inverse depth (larger = closer)"

Do not describe raw MiDaS output as metric depth in meters.

---

# 9. Scene Report

For each image, save:

outputs/custom/<image_stem>_scene_report.json

The JSON should contain the existing scene-analysis information, including where already supported:

- image information
- segmentation information
- depth information
- fusion information
- scene analysis
- difficulty score
- difficulty level
- difficulty components

Reuse the existing analyzer and difficulty-analysis output format.

Do not invent new metrics unless necessary.

---

# 10. MiDaS Depth Semantics

Preserve the existing project convention.

MiDaS / DPT-Large produces relative inverse depth.

It is NOT metric depth.

Therefore:

- larger inverse-depth value = closer
- smaller inverse-depth value = farther

Do not report raw MiDaS predictions as meters.

The demo does not require ground-truth labels.

---

# 11. No Ground Truth Required

This custom-image demo is inference only.

The user can put any suitable street-scene image into:

data/pipeline/images/

The image does not need:

- Cityscapes labels
- KITTI depth ground truth
- annotations
- masks

The demo simply runs both pretrained/trained models and fuses their predictions.

Do not attempt to calculate supervised evaluation metrics for arbitrary custom images.

---

# 12. Difficulty Analysis

Reuse the existing rule-based difficulty analysis.

Do not redesign the difficulty formula.

The current project uses the existing indicators such as:

- segmentation uncertainty
- depth variation
- scene complexity
- foreground fraction
- object/dynamic-class density

The difficulty score should remain a heuristic.

Do not describe it as ground-truth human difficulty.

---

# 13. Tests

Add automated tests for the new folder-based custom-image CLI.

Tests must NOT load the real 1.4GB MiDaS checkpoint.

Use mocked or synthetic predictors.

At minimum test:

1. Input directory exists.
2. Empty input directory is handled.
3. Supported image extensions are discovered.
4. Unsupported files are ignored or handled correctly.
5. `--limit 1` processes only one image.
6. Multiple images are processed.
7. Output directory is created automatically.
8. Expected output filenames are generated.
9. Same image is passed to both model predictors.
10. Scene report JSON is generated.
11. Existing Step 11 / Step 15 behavior is not broken.

Keep tests deterministic.

---

# 14. Main.py Integration

Inspect the existing `main.py` before making changes.

Integrate the custom-image demo into the existing CLI rather than creating an unrelated standalone program.

The preferred usage is:

python main.py --input-dir data/pipeline/images

Do not require:

python main.py --image path/to/image.jpg

The fixed folder-based workflow is intentional.

If the project already has another input-directory argument, reuse it instead of creating a duplicate option.

---

# 15. Documentation

Update README.md with a short section:

## Custom Image Fusion Demo

Explain:

1. Put street-scene images into:

data/pipeline/images/

2. Run:

python main.py --input-dir data/pipeline/images

3. For a quick test:

python main.py --input-dir data/pipeline/images --limit 1

4. Results are saved to:

outputs/custom/

Explain that no ground-truth annotation is required for this demo.

---

# 16. Prompt File

Create this file:

prompts/17_custom_image_fusion.md

This file should document the implementation requirements of this step.

---

# 17. Regression Safety

Do not modify:

- U-Net architecture
- MiDaS architecture
- training hyperparameters
- trained checkpoints
- segmentation metric formulas
- depth metric formulas
- existing Step 11 pipeline behavior
- existing Step 13 U-Net evaluation
- existing Step 14 MiDaS evaluation
- existing Step 15 pipeline evaluation

Only add the custom-image folder demo and the required CLI/tests/documentation.

---

# 18. Final Verification

After implementation:

Run the focused tests for Step 17.

Then run the full test suite:

pytest -q

Report:

- files created
- files modified
- focused test result
- full pytest result
- exact command used for the demo
- exact output directory
- any assumptions or compatibility changes

Do not claim success without actually running the tests.

---

# Expected Final Workflow

The final user workflow should be:

1. Put an image into:

data/pipeline/images/

Example:

data/pipeline/images/my_street.jpg

2. Run:

python main.py --input-dir data/pipeline/images

3. The system performs:

my_street.jpg
    ↓
U-Net
    ↓
Semantic Segmentation

my_street.jpg
    ↓
MiDaS DPT-Large
    ↓
Relative Inverse Depth

Segmentation + Depth
    ↓
Alignment
    ↓
Fusion
    ↓
Scene Analysis
    ↓
Difficulty Analysis
    ↓
Visualization + JSON

4. Results appear in:

outputs/custom/

This is the complete requirement for Step 17.