Step 16 — Coursework Report

File này gộp cả Prompt cho OpenCode và Report Template.
Có thể dùng duy nhất file này cho Step 16.

# PART A — PROMPT FOR OPENCODE

Step 16 --- Coursework Report

You are working on the existing CV-PROJECT scene-understanding
coursework.

The implementation from Steps 02--15 is complete.

Main requirement

Prepare the coursework report using:

English section headings

Vietnamese explanatory content

Keep important technical terms in English

Markdown format (.md)

The report should be easy for a Vietnamese university student to
understand and present orally.

Important rules

Do NOT modify implementation code.

Do NOT retrain U-Net.

Do NOT train/fine-tune MiDaS.

Do NOT run expensive new experiments.

Do NOT invent metrics, hyperparameters, dataset statistics or
conclusions.

Read actual project files and JSON outputs before writing numerical
results.

Cityscapes and KITTI are NOT paired datasets.

Never claim raw MiDaS output is depth in meters.

Raw MiDaS output is relative inverse depth.

The difficulty score is a heuristic, not ground truth.

Do not call the 2-image pipeline smoke test a statistically
significant benchmark.

# Step 1 --- Inspect evidence

Inspect:

README.md

configs/unet.yaml

configs/midas.yaml

configs/pipeline.yaml

models/unet/

models/midas/

preprocessing/

scene_understanding/

evaluation/

visualization/

docs/

outputs/analysis/unet_cityscapes_evaluation.json

outputs/analysis/midas_kitti_evaluation.json

outputs/analysis/pipeline_evaluation.json

Also inspect relevant prompt files from Steps 02--15 if necessary.

# Step 2 --- Verify actual numbers

Extract actual values for:

U-Net

architecture

number of classes

image size

batch size

learning rate

optimizer

weight decay

epochs

mixed precision

best epoch

validation loss

mIoU

Pixel Accuracy

Mean Dice

per-class IoU

MiDaS

DPT-Large

pretrained checkpoint

number of KITTI validation samples

RMSE

MAE

AbsRel

delta1

delta2

delta3

median scaling

relative inverse depth semantics

Pipeline

same-image contract

spatial alignment

fusion

scene analyzer

difficulty formula

thresholds

smoke-test image count

actual difficulty scores

Easy/Medium/Hard distribution

# Step 3 --- Write the report

Create:

docs/report/coursework_report.md

Use exactly this high-level structure:

Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

# 1. Introduction

1.1 Background

1.2 Motivation

# 2. Problem Definition

2.1 Input

2.2 Output

# 3. Objectives

# 4. Scope and Out of Scope

4.1 In Scope

4.2 Out of Scope

# 5. Dataset

5.1 Cityscapes

5.2 KITTI

5.3 Dataset Relationship

# 6. Methodology

6.1 Overall Architecture

6.2 U-Net

6.3 MiDaS DPT-Large

6.4 Feature Fusion

6.5 Scene Analysis

6.6 Difficulty Analysis

# 7. Implementation

7.1 Project Structure

7.2 Data Preprocessing

7.3 Model Training and Inference

7.4 Evaluation

7.5 Visualization

7.6 CLI and Reproducibility

# 8. Experimental Setup

8.1 U-Net Training

8.2 U-Net Evaluation

8.3 MiDaS Evaluation

8.4 Full Pipeline Smoke Test

# 9. Results

9.1 U-Net Results

9.2 MiDaS Results

9.3 Per-Class Segmentation Results

9.4 Full Pipeline Results

# 10. Discussion

10.1 U-Net Discussion

10.2 MiDaS Discussion

10.3 Fusion and Scene Understanding

10.4 Difficulty Analysis

10.5 Dataset Limitations

# 11. Limitations

# 12. Conclusion

# 13. References

Step 4 --- Technical explanations

Explain concepts in Vietnamese but preserve English terminology.

For example:

"Semantic Segmentation là quá trình gán một class cho từng pixel trong
ảnh."

Do NOT translate every technical term into awkward Vietnamese.

Explain important formulas and variables clearly.

For the difficulty score, use the actual implementation:

difficulty_score = 0.25 * seg_uncertainty + 0.20 * depth_variation +
0.20 * scene_complexity + 0.20 * foreground_fraction + 0.15 *
object_density

Explain:

seg_uncertainty = 1 - mean_confidence

depth_variation = std / mean

scene_complexity = present classes / 19

foreground_fraction = near-region pixel ratio

object_density = ROI-class pixel ratio

Use the exact thresholds from the implementation rather than guessing
them.

Step 5 --- Results tables

Use Markdown tables.

Do not round results excessively.

Use the actual verified results, including:

U-Net: - Pixel Accuracy - mIoU - Mean Dice - per-class IoU

MiDaS: - RMSE - MAE - AbsRel - delta1 - delta2 - delta3

Pipeline: - actual 2-image smoke-test scores and levels

Clearly label pipeline results as a smoke test.

Step 6 --- Visualizations

If useful, reference the existing generated visualizations rather than
creating new expensive experiments.

Use relative paths where appropriate, for example:

outputs/segmentation/... outputs/depth/... outputs/analysis/...

Do not invent filenames.

Step 7 --- References

Include academically appropriate references for:

U-Net

MiDaS

Cityscapes

KITTI

Do not invent DOI numbers or publication details.

Step 8 --- Quality check

After writing:

Check every numerical result against the actual JSON/config.

Check that Cityscapes/KITTI are described as separate, non-paired
datasets.

Check that MiDaS is described as relative inverse depth.

Check that difficulty is described as heuristic.

Check that no implementation files were modified.

Keep the report readable and not excessively long.

Avoid large source-code dumps.

Final response

Report:

report path

sections created

numerical results used

references included

any missing information

confirmation that implementation files were not modified

PART B — COURSEWORK REPORT TEMPLATE

Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

# 1. Introduction

1.1 Background

[Viết phần giới thiệu về Scene Understanding, Semantic Segmentation và
Monocular Depth Estimation bằng tiếng Việt.]

1.2 Motivation

[Giải thích vì sao cần kết hợp thông tin ngữ nghĩa và độ sâu trong ảnh
đường phố.]

# 2. Problem Definition

[Bài toán nhận một ảnh RGB đường phố làm input và thực hiện semantic
segmentation, monocular depth estimation, fusion và scene/difficulty
analysis.]

2.1 Input

[Describe input.]

2.2 Output

[Describe segmentation, relative inverse depth, fusion and scene
analysis outputs.]

# 3. Objectives

[Objective 1]

[Objective 2]

[Objective 3]

[Objective 4]

[Objective 5]

# 4. Scope and Out of Scope

4.1 In Scope

Semantic Segmentation

Monocular Depth Estimation

Feature Fusion

Scene Understanding

Difficulty Analysis

Model Evaluation

Visualization

Reproducible CLI Pipeline

4.2 Out of Scope

MiDaS training/fine-tuning

Object Detection and Tracking

3D Reconstruction

Autonomous-driving control

Metric-depth prediction from raw MiDaS output

Paired Cityscapes-KITTI benchmarking

Ground-truth difficulty labeling

# 5. Dataset

5.1 Cityscapes

[Describe Cityscapes, semantic segmentation ground truth, splits and
why it is used.]

5.2 KITTI

[Describe KITTI depth data, validation subset and why it is used.]

5.3 Dataset Relationship

[Explain clearly that Cityscapes and KITTI are not paired. Model-level
evaluation is performed separately. Full pipeline uses the same RGB
image as input to both models.]

# 6. Methodology

6.1 Overall Architecture

                    RGB Image
                        |
             +----------+----------+
             |                     |
             v                     v
          U-Net                 MiDaS
      Segmentation              Depth
             |                     |
             |              Relative Inverse
             |                  Depth
             +----------+----------+
                        |
                      Fusion
                        |
                        v
                Scene Understanding
                        |
                        v
                Difficulty Analysis

6.2 U-Net

[Explain encoder, bottleneck, decoder, skip connections, output logits,
loss, optimizer and training process.]

6.3 MiDaS DPT-Large

[Explain pretrained DPT-Large, monocular depth estimation and relative
inverse depth.]

Important: Raw MiDaS output is relative inverse depth, not metric
depth in meters. Larger values indicate relatively closer regions in
the project's visualization/interpretation.

6.4 Feature Fusion

[Describe the actual implementation in scene_understanding/fusion.py.
Do not invent a fusion mechanism.]

6.5 Scene Analysis

[Describe the actual implementation in
scene_understanding/analyzer.py.]

6.6 Difficulty Analysis

[Describe the actual rule-based difficulty score.]

The implemented score is:

[ D = 0.25U + 0.20V + 0.20C + 0.20F + 0.15O ]

where:

(U): segmentation uncertainty

(V): depth variation

(C): scene complexity

(F): foreground fraction

(O): object density

The score is clamped to ([0,1]) and mapped to Easy/Medium/Hard
according to the implemented thresholds.

This is a heuristic pipeline difficulty score, not a ground-truth
difficulty label.

# 7. Implementation

7.1 Project Structure

[Insert concise project structure.]

7.2 Data Preprocessing

[Describe preprocessing.]

7.3 Model Training and Inference

[Describe U-Net training and MiDaS inference.]

7.4 Evaluation

[Describe model-level and pipeline evaluation.]

7.5 Visualization

[Describe generated visual outputs.]

7.6 CLI and Reproducibility

[Include important commands.]

# 8. Experimental Setup

8.1 U-Net Training

[Use actual values from configs/unet.yaml and training history.]

8.2 U-Net Evaluation

[Describe Cityscapes validation evaluation.]

8.3 MiDaS Evaluation

[Describe KITTI validation evaluation and median scaling.]

8.4 Full Pipeline Smoke Test

[Describe the 2-image real-data smoke test. Do not call it a
statistically significant benchmark.]

# 9. Results

9.1 U-Net Results

Metric             Result

Pixel Accuracy
mIoU
Mean Dice

9.2 MiDaS Results

Metric     Result

RMSE
MAE
AbsRel
δ1
δ2
δ3

9.3 Per-Class Segmentation Results

Class             IoU

road
sidewalk
building
wall
fence
pole
traffic light
traffic sign
vegetation
terrain
sky
person
rider
car
truck
bus
train
motorcycle
bicycle

9.4 Full Pipeline Results

Image     Difficulty Score Level



[Add relevant scene indicators if useful.]

# 10. Discussion

10.1 U-Net Discussion

[Discuss strengths and weaknesses based on actual results.]

10.2 MiDaS Discussion

[Discuss strengths and weaknesses, especially relative depth.]

10.3 Fusion and Scene Understanding

[Explain what combining semantic and depth information adds.]

10.4 Difficulty Analysis

[Discuss the heuristic score and its interpretation.]

10.5 Dataset Limitations

[Discuss the fact that Cityscapes and KITTI are not paired.]

# 11. Limitations

[Limitation]

[Limitation]

[Limitation]

# 12. Conclusion

[Summarize what was implemented, evaluated and demonstrated.]

# 13. References

U-Net: Convolutional Networks for Biomedical Image Segmentation.

MiDaS: Towards Robust Monocular Depth Estimation.

Cityscapes Dataset.

KITTI Vision Benchmark Suite.