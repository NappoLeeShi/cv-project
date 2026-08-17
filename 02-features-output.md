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

## 2.4 Shared feature backbone

Both tasks benefit from a **shared encoder** that learns general visual features:

```
Input image
     │
     ▼
┌────────────┐
│  Shared    │   ← pretrained on ImageNet / similar
│  Encoder   │
└──┬─────┬───┘
   │     │
   ▼     ▼
Seg     Depth
Head    Head
```

**Advantages:**
- Fewer parameters than two separate networks.
- Depth cues (edges, geometry) help segmentation; semantic cues (class identity) help depth.
- Joint training can regularise both heads.

---

## 2.5 Feature → Output summary

| Low-level feature | Mid-level feature | High-level feature | Segmentation | Depth |
|---|---|---|---|---|
| RGB colour | Edges, corners | Object parts | ✓ | ✓ |
| Texture gradients | Surface orientation | Scene layout | ✓ | ✓ |
| — | Vanishing point | Global geometry | — | ✓ |
| — | Object boundaries | Instance extent | ✓ | ✓ |
