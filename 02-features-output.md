# 2. Features → Output Mapping

This document maps low-level and mid-level image features to the two target outputs.

---

## 2.1 Key image features exploited

| Feature type | What it captures | Relevant output |
|---|---|---|
| **Colour / intensity** | Road is dark grey, sky is blue, vegetation is green | Segmentation |
| **Texture** | Asphalt vs. grass vs. metal have different texture patterns | Segmentation |
| **Edges / gradients** | Object boundaries, horizon line | Segmentation + Depth discontinuities |
| **Perspective / vanishing point** | Parallel lines converge → distance cue | Depth |
| **Relative size** | Same-class objects appear smaller when far away | Depth |
| **Occlusion ordering** | Nearer objects block farther ones | Depth |
| **Shadows / shading** | Light direction reveals surface orientation | Depth |
| **Semantic priors** | "Sky is always above the horizon", "road is at bottom" | Both |

---

## 2.2 Output 1 — Semantic Segmentation

### Per-pixel features → class label

```
pixel features (colour, texture, local context)
        │
        ▼
   Encoder  ──►  high-level feature map  (semantic context)
        │
        ▼
   Decoder  ──►  per-pixel logits  (num_classes channels)
        │
        ▼
   argmax  ──►  class index ∈ {road, car, sky, building, …}
```

Key design choices:
- **Multi-scale context** – a pixel's class depends on both local appearance and global scene layout.
- **Skip connections** – preserve fine boundaries lost during downsampling.
- **CRF / refinement** – optional post-processing for sharper masks.

### Target classes (Cityscapes, 19 classes)
`road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle`

---

## 2.3 Output 2 — Depth Map

### Per-pixel features → scalar depth

```
pixel features (colour, texture, perspective cues)
        │
        ▼
   Encoder  ──►  feature representation
        │
        ├──► Depth head  ──►  log-depth / disparity  (H×W×1)
        │
        └──► (optional) Confidence map  (H×W×1)
```

Key design choices:
- **Scale-invariant loss** – predict relative depth when absolute scale is unavailable.
- **Multi-scale supervision** – auxiliary losses at intermediate decoder stages improve gradient flow.
- **Edge-aware refinement** – sharpen depth discontinuities at object boundaries.

---

## 2.4 Relationship Between Segmentation and Depth

Although semantic segmentation and depth estimation are performed using separate models (U-Net and MiDaS), both tasks rely on similar visual cues extracted from the input image.

```
Input Image
     │
 ┌───┴───┐
 │       │
 ▼       ▼
U-Net   MiDaS
 │       │
 ▼       ▼
Seg.    Depth
Mask    Map
```

### Complementary Information

- **Segmentation** answers: *What is in the scene?*
- **Depth estimation** answers: *How far is each object from the camera?*

When combined, they provide a more complete understanding of the traffic environment.

For example:

| Object | Segmentation Output | Depth Output |
|----------|----------|----------|
| Car | Car | 12 m |
| Pedestrian | Person | 6 m |
| Road | Road | Drivable surface |
| Building | Building | 40 m |

### Benefits of Combining Both Tasks

- Improve scene understanding in complex traffic environments.
- Provide both semantic and geometric information.
- Support applications such as autonomous driving, robot navigation, and intelligent transportation systems.

Although the two models are trained separately, their outputs can be integrated to obtain richer scene-level information than either task alone.

## 2.5 Feature → Output summary

| Low-level feature | Mid-level feature | High-level feature | Segmentation | Depth |
|---|---|---|---|---|
| RGB colour | Edges, corners | Object parts | ✓ | ✓ |
| Texture gradients | Surface orientation | Scene layout | ✓ | ✓ |
| — | Vanishing point | Global geometry | — | ✓ |
| — | Object boundaries | Instance extent | ✓ | ✓ |
---

## 2.6 Out of Scope

The feature analysis presented in this section focuses on common visual cues used for semantic segmentation and monocular depth estimation.

The following aspects are outside the scope of this project:

- **Video-based temporal features** such as object motion, optical flow, and trajectory information.
- **Multi-view geometry** from stereo cameras or multiple viewpoints.
- **LiDAR, Radar, or sensor-fusion features** that provide direct depth measurements.
- **Instance-level reasoning**, such as distinguishing between individual objects of the same class.
- **3D object pose estimation** and full 3D scene reconstruction.
- **Weather and illumination robustness analysis** under extreme conditions (heavy rain, fog, snow, or nighttime scenes).
- **Explainability of learned features**, since deep neural networks learn many internal representations automatically.

Therefore, this project only considers visual features extracted from a single RGB street-view image for semantic segmentation and monocular depth estimation.