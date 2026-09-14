# Step 09 — Fusion + Scene Understanding

## Objective

Implement the final scene-understanding analysis by combining semantic segmentation and monocular relative depth predictions generated from the SAME input street image.

This step must NOT combine Cityscapes images with KITTI images.

Cityscapes and KITTI are used only for model evaluation in previous steps.

Focus ONLY on:
- same-image segmentation + depth fusion
- region-level depth analysis
- traffic-scene context analysis
- scene understanding output

Do NOT implement:
- training
- Cityscapes evaluation
- KITTI evaluation
- new neural networks
- depth metrics
- segmentation metrics
- CI/CD
- deployment

---

## Read existing project

Read:

- prompts/01_architecture.md
- prompts/02_config_utils.md
- prompts/03_dataset.md
- prompts/04_unet.md
- prompts/05_unet_inference.md
- prompts/06_segmentation_evaluation.md
- prompts/07_midas.md
- prompts/08_depth_evaluation.md

Inspect:

- models/unet/
- models/midas/
- scene_understanding/
- configs/pipeline.yaml
- configs/unet.yaml
- configs/midas.yaml
- utils/
- existing tests

Reuse existing APIs.

---

# 1. Core architecture

The final demo pipeline is:

    New Street Image
           |
       +---+---+
       |       |
      U-Net   MiDaS
       |       |
       ↓       ↓
    Segmentation  Relative Depth
       \       /
        \     /
         Fusion
           |
           ↓
    Scene Understanding

Both models MUST receive the exact same RGB input image.

Do NOT pair a Cityscapes image with a KITTI image.

Do NOT use Cityscapes GT segmentation and KITTI GT depth as if they belong to the same scene.

---

# 2. Fusion type

Use analytical/rule-based fusion.

Do NOT create a new neural fusion network.

The goal is to demonstrate how semantic information and depth information complement each other.

---

# 3. Input contract

Fusion should accept:

- segmentation prediction: [H, W]
- depth prediction: [H, W]
- optional confidence map: [H, W]

Both segmentation and depth must have identical spatial dimensions.

Validate shape compatibility.

Depth is the raw relative inverse-depth representation produced by MiDaS.

Do not convert it to meters.

---

# 4. Class-aware depth analysis

Implement region-level depth statistics.

For each selected semantic class, compute statistics over pixels belonging to that class.

At minimum support:

- pixel count
- pixel ratio
- mean depth
- median depth
- minimum depth
- maximum depth

Important:

MiDaS uses inverse relative depth where larger values indicate closer regions.

Therefore document this convention consistently.

---

# 5. Traffic-relevant classes

Support analysis for classes such as:

- road
- car
- person
- truck
- bus
- motorcycle
- bicycle
- sidewalk
- sky

Do not assume every class exists in every prediction.

Missing classes should produce an empty/absent result rather than an error.

---

# 6. Relative depth regions

Provide a simple relative-depth categorization.

For example:

- near
- middle
- far

The thresholds must be derived from the prediction itself rather than pretending the values are meters.

A percentile-based approach is acceptable.

For example:

    near   = top 33% of depth values
    middle = middle 33%
    far    = bottom 33%

Because MiDaS output is inverse depth:

    larger value → nearer
    smaller value → farther

Document the exact convention.

Do not call these categories "meters".

---

# 7. Scene-level statistics

Implement scene-level analysis such as:

- total analyzed pixels
- semantic class distribution
- road coverage ratio
- object-class pixel ratios
- relative depth distribution
- number of detected semantic regions/classes

Keep the analysis deterministic.

---

# 8. Traffic context

Implement simple interpretable rules.

Examples:

### Vehicle proximity

For each vehicle class:

    vehicle_mask = segmentation == class_id

    analyze depth values inside vehicle_mask

Return a relative proximity description such as:

- near
- middle
- far

based on the scene depth thresholds.

### Pedestrian context

For person pixels:

    person_mask = segmentation == person_class

Analyze their relative depth distribution.

### Road context

Analyze depth statistics inside the road mask.

Do NOT claim physical distance in meters.

---

# 9. Output

Return a structured dictionary that is easy to serialize as JSON.

Suggested structure:

    {
        "scene": {
            "height": ...,
            "width": ...,
            "analyzed_pixels": ...
        },

        "semantic_distribution": {
            "road": ...,
            "car": ...,
            "person": ...
        },

        "depth_distribution": {
            "near": ...,
            "middle": ...,
            "far": ...
        },

        "regions": {
            "road": {
                "pixel_count": ...,
                "pixel_ratio": ...,
                "mean_depth": ...,
                "median_depth": ...
            },

            "car": {
                ...
            }
        },

        "traffic_context": {
            ...
        }
    }

Keep the output interpretable.

---

# 10. API

Implement reusable functions/classes in:

    scene_understanding/fusion.py
    scene_understanding/analyzer.py

Suggested responsibilities:

### fusion.py

Low-level operations:

- validate segmentation/depth
- create semantic masks
- calculate class-wise depth statistics
- calculate depth regions
- combine semantic and depth information

### analyzer.py

High-level scene interpretation:

- scene statistics
- traffic context
- structured final result

Do not duplicate metric implementations from evaluation/.

---

# 11. Model integration

Provide a high-level function that can run:

    image
      ↓
    U-Net inference
      ↓
    segmentation

and:

    image
      ↓
    MiDaS inference
      ↓
    depth

then pass both outputs into the fusion/analyzer layer.

Reuse:

- models/unet/inference.py
- models/midas/inference.py

Do not duplicate preprocessing or model-loading logic.

---

# 12. No ground truth requirement for demo

The final demo does NOT require:

- Cityscapes GT
- KITTI GT
- paired datasets

The demo is inference-only.

Ground truth remains part of the separate model-evaluation stages.

---

# 13. Tests

Create:

    tests/test_fusion.py
    tests/test_scene_analyzer.py

Tests must be deterministic and offline.

Do NOT download models or datasets.

Test:

1. segmentation/depth shape mismatch
2. empty masks
3. class pixel count
4. class pixel ratio
5. class mean depth
6. class median depth
7. class min/max depth
8. missing semantic class
9. near/middle/far categorization
10. inverse-depth convention
11. road analysis
12. car analysis
13. person analysis
14. multiple classes
15. scene-level statistics
16. structured output
17. deterministic results
18. same-image shape contract
19. invalid NaN depth
20. invalid segmentation values where appropriate

Use small hand-checkable arrays.

---

# 14. Important interpretation restriction

Never make claims such as:

    "The car is 10 meters away."

The depth is relative.

Use statements such as:

    "The car is in the near relative-depth region."

or:

    "This vehicle has a larger relative inverse-depth value than that vehicle."

---

# 15. Configuration

Update configs/pipeline.yaml only if necessary.

Keep configuration simple.

Do not add training parameters.

---

# 16. Existing tests

Run:

    pytest

Preserve all previous tests.

Verify imports:

    python -c "from scene_understanding.fusion import *; from scene_understanding.analyzer import *; print('Scene understanding modules OK')"

---

# 17. Scope restriction

Do NOT implement:

- new neural networks
- model training
- Cityscapes/KITTI pairing
- segmentation evaluation
- depth evaluation
- visualization
- CI/CD
- deployment

STOP after Step 09.

---

# 18. Report

Report:

1. files created
2. files modified
3. fusion design
4. same-image contract
5. semantic-depth interaction
6. relative-depth handling
7. traffic-context rules
8. output structure
9. tests
10. full pytest result
11. assumptions
12. problems/blockers

STOP after Step 09.