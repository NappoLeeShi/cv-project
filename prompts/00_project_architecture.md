# Project Architecture Specification

You are a senior Computer Vision and Machine Learning software architect.

Your task is to analyze the EXISTING project architecture and produce
a detailed technical design specification for the project.

IMPORTANT:
- Do NOT write implementation code.
- Do NOT redesign the project from scratch.
- Do NOT create a completely new folder structure.
- Do NOT arbitrarily add or remove modules.
- First understand the existing architecture.
- Only recommend structural changes if there is a strong technical reason.
- If a change is recommended, explain why before making the recommendation.

---

# 1. Project

Project name:

CV-PROJECT

Project goal:

Segmentation + Depth → Scene Understanding

The system is designed to analyze street scenes using:

- Semantic Segmentation
- Monocular Depth Estimation
- Segmentation + Depth Fusion
- Scene / Traffic Context Analysis

---

# 2. Existing Architecture

The current project structure is:
```
CV-PROJECT/
│
├── data/
│   ├── cityscapes/
│   └── kitti/
│
├── evaluation/
│   ├── depth_metrics.py
│   ├── difficulty_analysis.py
│   ├── pipeline_metrics.py
│   └── segmentation_metrics.py
│
├── models/
│   ├── midas/
│   │   ├── inference.py
│   │   └── model.py
│   │
│   └── unet/
│       ├── inference.py
│       └── model.py
│
├── notebooks/
│
├── outputs/
│   ├── analysis/
│   ├── depth/
│   └── segmentation/
│
├── preprocessing/
│   ├── cityscapes.py
│   └── kitti.py
│
├── prompts/
│
├── scene_understanding/
│   ├── analyzer.py
│   └── fusion.py
│
├── visualization/
│   ├── depth.py
│   ├── scene.py
│   └── segmentation.py
│
├── configs/
│   ├── unet.yaml
│   ├── midas.yaml
│   └── pipeline.yaml
│
├── utils/
│   ├── config.py
│   ├── seed.py
│   └── logger.py
│
├── tests/
│
├── README.md
├── requirements.txt
└── main.py
```

Treat this architecture as the baseline architecture.

---

# 3. Problem Definition

Input:

- Street images

Required AI outputs:

1. Semantic segmentation
2. Monocular depth map

The final objective is to use these outputs to perform
traffic scene / scene context analysis.

The project uses:

Cityscapes
→ semantic segmentation

KITTI Depth
→ depth estimation / depth evaluation

IMPORTANT:

Cityscapes and KITTI are separate datasets.

Do NOT assume that an image from Cityscapes and an image
from KITTI correspond to the same physical scene.

Do NOT fabricate paired segmentation-depth ground truth.

---

# 4. AI Models

## U-Net

Purpose:

Semantic segmentation.

Input:

RGB image.

Output:

Pixel-wise class predictions.

For Cityscapes, consider the appropriate semantic class mapping
and number of trainable classes.

Explain how U-Net should be configured for this project.

Cover:

- Encoder
- Convolution
- Downsampling
- Bottleneck
- Decoder
- Upsampling
- Skip connections
- Output layer

---

## MiDaS

Purpose:

Monocular depth estimation.

Input:

RGB image.

Output:

Relative depth representation.

IMPORTANT:

Do not treat MiDaS output as metric depth in meters unless
a valid scale/calibration methodology is explicitly introduced.

Prefer using the pretrained MiDaS implementation rather than
unnecessarily reimplementing the entire model.

Explain:

- Model loading
- Preprocessing
- Inference
- Output handling
- Depth normalization
- Limitations of relative depth

---

# 5. Required Functional Components

Design the technical responsibilities for:

## Data

- Cityscapes loading
- KITTI loading
- preprocessing
- transformations
- train/validation/test handling

## U-Net

- model definition
- configuration
- training/fine-tuning strategy
- checkpoint loading/saving
- inference

## MiDaS

- model loading
- preprocessing
- inference
- depth output handling

## Fusion

- segmentation + depth interface
- combining semantic and depth information
- region/object depth analysis
- spatial context analysis

## Scene Analyzer

Analyze the fused information to produce useful
traffic/scene understanding results.

Examples:

- near/far objects
- object-region depth
- road context
- scene complexity
- traffic context

The fusion should initially be interpretable and
rule-based/analytical rather than introducing an unnecessary
new neural network.

---

# 6. Evaluation

Design exactly where and how each metric should be calculated.

## Segmentation

Consider:

- Pixel Accuracy
- IoU
- mIoU
- Dice Score

Identify the primary metric and explain why.

## Depth

Consider:

- RMSE
- MAE / Absolute Error
- Abs Rel
- δ accuracy

Explain which metrics are appropriate given the
relative-depth nature of MiDaS and the available ground truth.

## Full Pipeline

Design metrics for the complete:

Image
→ Segmentation
+
Depth
→ Fusion
→ Scene Understanding

Consider:

- task-specific scene understanding correctness
- inference time
- robustness
- difficulty analysis

---

# 7. Visualization

Define the responsibility of:

visualization/segmentation.py

visualization/depth.py

visualization/scene.py

The visualization should support:

- original image
- segmentation mask
- segmentation overlay
- depth map
- segmentation + depth visualization
- prediction vs ground truth

---

# 8. Difficulty Analysis

Define how:

evaluation/difficulty_analysis.py

should analyze scene difficulty.

Possible factors:

- number of objects
- object size
- occlusion
- segmentation complexity
- depth variation
- visual complexity

Propose a reasonable:

Easy / Medium / Hard

classification methodology suitable for a university
coursework project.

Do not create an unnecessarily complicated machine learning
model for difficulty classification.

---

# 9. Configuration

Define what should be stored in:

configs/unet.yaml
configs/midas.yaml
configs/pipeline.yaml

Examples:

- dataset paths
- image size
- batch size
- learning rate
- epochs
- number of classes
- model path
- device
- random seed
- output paths

Clearly separate configuration from implementation logic.

---

# 10. Utils

Define the responsibility of:

utils/config.py
utils/seed.py
utils/logger.py

Explain what should be shared between modules and
what should NOT be placed inside utils.

---

# 11. Testing

Design tests for:

tests/

At minimum:

1. Dataset loading
2. Preprocessing
3. U-Net forward pass
4. U-Net inference
5. MiDaS inference
6. Segmentation metrics
7. Depth metrics
8. Fusion
9. Scene analyzer
10. End-to-end pipeline

For each test, specify:

- input
- expected output
- what should be validated

Do not write the test code yet.

---

# 12. Main Pipeline

Define how main.py should orchestrate the system.

The desired conceptual flow is:

Dataset
    ↓
Preprocessing
    ↓
U-Net / MiDaS
    ↓
Predictions
    ↓
Evaluation
    ↓
Fusion
    ↓
Scene Understanding
    ↓
Visualization
    ↓
Results

Clearly explain which parts are independent branches
and which parts are combined.

---

# 13. Technology Stack

Recommend the final technology stack.

Evaluate the necessity of:

- Python
- PyTorch
- OpenCV
- NumPy
- Pillow
- Matplotlib
- scikit-learn
- PyYAML
- pytest
- Git

For every technology, explain:

1. Why it is needed
2. Which module uses it
3. Whether it is required or optional

Do not add unnecessary libraries.

---

# 14. MLOps / CI-CD

Design a lightweight coursework-level CI/CD workflow.

Example:

Git Push
   ↓
Automated Tests
   ↓
Code Validation
   ↓
Model / Pipeline Evaluation
   ↓
Build / Package

Explain:

- what should be automated
- what should remain manual
- where tests are executed
- where model evaluation is executed

Do NOT design a large production cloud infrastructure.

---

# 15. Module Dependency

Create a dependency map showing which modules
can import which modules.

Avoid:

- circular dependencies
- model code depending on visualization
- evaluation depending on UI
- preprocessing depending on model inference

Prefer a clean dependency direction such as:

config/utils
      ↓
preprocessing
      ↓
models
      ↓
scene_understanding
      ↓
evaluation / visualization

Explain if another dependency direction is technically better.

---

# 16. Data Flow

Create Mermaid diagrams for:

1. Overall system
2. Segmentation branch
3. Depth branch
4. Fusion / scene understanding
5. Evaluation
6. CI/CD

---

# 17. Scope Control

The following are OUT OF SCOPE:

- Object Detection
- Object Tracking
- Lane Detection
- Instance Segmentation
- 3D Object Detection
- LiDAR-camera fusion
- Autonomous driving control
- LLM
- Chatbot
- unnecessary cloud deployment
- unnecessary neural fusion network

Do not introduce an "AI Agent".

Use:

"Fusion Module"

or

"Scene Understanding Module"

for the integration component.

---

# 18. Required Output

Produce a detailed architecture document with:

1. Project Overview
2. Functional Requirements
3. Non-functional Requirements
4. In-Scope
5. Out-of-Scope
6. Technology Stack
7. High-Level Architecture
8. Detailed Module Architecture
9. Folder Responsibility
10. Module Dependencies
11. Data Flow
12. U-Net Design
13. MiDaS Integration
14. Fusion Design
15. Scene Analyzer Design
16. Evaluation Design
17. Visualization Design
18. Configuration Design
19. Testing Strategy
20. MLOps / CI-CD
21. Development Order
22. Technical Risks
23. Final Architecture Recommendation

Use tables and Mermaid diagrams where appropriate.

Do NOT write implementation code.

The final specification must be detailed enough that another
developer or AI coding agent can implement the project
step-by-step without redesigning the architecture.