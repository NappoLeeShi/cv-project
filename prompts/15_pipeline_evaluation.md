# Step 15 — Full Pipeline Evaluation: Fusion + Scene Understanding

You are working on the existing CV-PROJECT scene-understanding coursework.

IMPORTANT:
- Continue from the current repository state.
- Do NOT redesign existing models.
- Do NOT retrain U-Net.
- Do NOT train MiDaS.
- Do NOT modify the existing U-Net/MiDaS architectures unless absolutely required for compatibility.
- Preserve all existing tests and behavior.
- Cityscapes and KITTI are NOT paired datasets. Never pair a Cityscapes image with a KITTI depth map.
- This step evaluates the FULL PIPELINE on the SAME INPUT IMAGE:
    image -> U-Net segmentation
          -> MiDaS relative inverse depth
          -> fusion
          -> scene analysis
          -> difficulty analysis

## Goal

Implement a reproducible full-pipeline evaluation/demo that measures how the existing segmentation and depth outputs can be combined for scene understanding.

The pipeline should use the existing modules:

- models/unet/
- models/midas/
- scene_understanding/fusion.py
- scene_understanding/analyzer.py
- scene_understanding/pipeline.py
- visualization/
- evaluation/

Use the existing trained/checkpointed models:

- checkpoints/unet_cityscapes.pth
- checkpoints/dpt_large_384.pt

## Step 1 — Inspect the existing implementation

Before changing anything:

1. Inspect:
   - scene_understanding/fusion.py
   - scene_understanding/analyzer.py
   - scene_understanding/pipeline.py
   - models/unet/inference.py
   - models/midas/inference.py
   - visualization/
   - evaluation/
   - configs/pipeline.yaml
2. Understand their existing APIs.
3. Reuse existing functionality instead of duplicating it.
4. Run the existing test suite before modifications.

Do not make speculative architectural changes.

## Step 2 — Define pipeline evaluation

Create:

    evaluation/evaluate_pipeline.py

The evaluator should:

1. Load configuration.
2. Load the trained U-Net checkpoint.
3. Load the pretrained MiDaS DPT-Large checkpoint.
4. Select CUDA when available.
5. Run both models in eval/no_grad mode.
6. For each selected input image:
   - load exactly one RGB image
   - pass the SAME image to U-Net and MiDaS
   - obtain semantic segmentation
   - obtain relative inverse depth
   - explicitly align output spatial dimensions
   - fuse the two outputs
   - run scene analyzer
   - compute difficulty indicators
7. Save structured results as JSON.

## Step 3 — Important depth semantics

Preserve the existing MiDaS semantics:

    raw MiDaS output = relative inverse depth

Therefore:

    larger inverse-depth value -> closer
    smaller inverse-depth value -> farther

Do NOT describe raw MiDaS output as meters.

Do NOT compute metric depth unless an existing supported conversion is explicitly available.

Median scaling is an evaluation-time operation only and must not change the meaning of the raw demo depth map.

## Step 4 — Difficulty analysis

Implement or extend the existing difficulty analysis so that each image receives interpretable difficulty indicators.

Possible indicators:

- segmentation confidence
- mean / variance of depth
- depth discontinuity
- number of detected semantic regions/classes
- proportion of important traffic classes
- segmentation uncertainty
- depth variation
- foreground/background separation
- scene complexity

Do not invent a fake "ground truth difficulty".

Instead, clearly define a transparent rule-based difficulty score.

For example, if appropriate to the existing code:

    difficulty_score =
        weighted combination of normalized complexity indicators

Then classify:

    Easy
    Medium
    Hard

The exact weights must be documented and configurable.

Do not claim that this is an objective ground-truth difficulty label.
Call it:

    pipeline difficulty score

or

    scene complexity score

## Step 5 — Dataset handling

Because Cityscapes and KITTI are not paired:

DO NOT do:

    Cityscapes image -> KITTI depth GT

Instead support two types of evaluation:

### A. Same-image pipeline demo/evaluation

Use a standalone set of RGB street images.

For each image:

    same RGB image
       ├── U-Net
       └── MiDaS
              ↓
            Fusion
              ↓
        Scene Analysis

If a suitable existing image set is already available in the repository, reuse it.

Otherwise create a small configurable demo-input directory:

    data/pipeline/images/

Do not duplicate the entire Cityscapes/KITTI datasets.

### B. Model-level evaluation remains separate

Keep Step 13 and Step 14 results as the official model-level metrics:

U-Net:
- Cityscapes GT
- mIoU
- Pixel Accuracy
- Dice

MiDaS:
- KITTI GT
- RMSE
- MAE
- AbsRel
- delta1
- delta2
- delta3

Do not merge these datasets into one paired benchmark.

## Step 6 — Evaluation output

Create:

    outputs/analysis/pipeline_evaluation.json

For every evaluated image save fields similar to:

{
    "image": "...",
    "segmentation": {
        "num_classes_present": ...,
        "mean_confidence": ...
    },
    "depth": {
        "mean_inverse_depth": ...,
        "std_inverse_depth": ...,
        "min_inverse_depth": ...,
        "max_inverse_depth": ...
    },
    "scene": {
        ...
    },
    "difficulty": {
        "score": ...,
        "level": "easy|medium|hard",
        "components": {
            ...
        }
    }
}

Also save aggregate statistics:

- number of images
- average difficulty score
- Easy count
- Medium count
- Hard count
- average segmentation confidence
- average depth variation
- class occurrence statistics if already supported

## Step 7 — Visual outputs

For at least a small configurable number of images, save:

    outputs/segmentation/
    outputs/depth/
    outputs/analysis/

Use the existing visualization utilities.

For each demo image, ideally produce:

1. original image
2. segmentation result
3. depth visualization
4. fusion visualization
5. scene-analysis result/report

Do not create misleading metric-depth labels.

## Step 8 — CLI

The evaluator should support something similar to:

    python -m evaluation.evaluate_pipeline \
        --config configs/pipeline.yaml \
        --input-dir data/pipeline/images

Support useful options such as:

    --limit N
    --device cuda
    --save-visualizations
    --output PATH

Do not require the full dataset for development.

A small smoke test should be possible with:

    --limit 2

## Step 9 — Tests

Create:

    tests/test_evaluate_pipeline.py

Use synthetic/mock data where appropriate.

Test:

1. configuration loading
2. same image object is supplied to both predictors
3. segmentation/depth spatial alignment
4. fusion receives outputs from the same image
5. difficulty score is deterministic
6. Easy/Medium/Hard thresholds work
7. JSON output schema
8. --limit works
9. CPU-safe execution with mocks
10. no accidental Cityscapes/KITTI pairing

Do not require real 4GB-GPU inference inside unit tests.

## Step 10 — Full regression test

Run:

    pytest -q

Expected result:

- all previous tests remain passing
- no regression in Steps 02–14

Report:

- number of tests
- passed
- skipped
- failed

## Step 11 — Real-data smoke test

Before any large evaluation:

run a very small real-input test, preferably:

    --limit 2

Verify:

- U-Net checkpoint loads
- MiDaS checkpoint loads
- CUDA works
- same image is passed to both models
- segmentation output is valid
- depth output is valid
- fusion works
- analyzer works
- difficulty score is produced
- visualization files are created
- JSON is valid

Do NOT run a large dataset automatically.

## Step 12 — Documentation

Create:

    prompts/15_pipeline_evaluation.md

Document:

- purpose
- architecture
- why datasets are not paired
- same-image contract
- fusion logic
- difficulty-score definition
- Easy/Medium/Hard thresholds
- limitations
- reproducibility commands

## Final report

After implementation, report:

1. files created
2. files modified
3. tests passed/skipped/failed
4. real-data smoke-test result
5. output JSON path
6. visualization paths
7. exact difficulty-score formula
8. Easy/Medium/Hard distribution if real evaluation was run
9. any limitations or assumptions

Do not proceed to a large-scale pipeline benchmark unless explicitly requested.