# 1. Problem Definition

## Task
**Segmentation + Depth → Scene Understanding**

Given a single street-view image, produce two dense predictions:

| Output | Description |
|---|---|
| Semantic segmentation mask | Per-pixel class label (road, car, sky, building, pedestrian, …) |
| Depth map | Per-pixel distance from camera |

Together these form a basic **scene understanding** system: *what* is in the image and *where* it is in 3-D.

---

## Why it matters
- **Autonomous driving** – vehicles need to know drivable surfaces and surrounding objects plus their distance.
- **Robotics / navigation** – a robot must segment free space and estimate distance to obstacles.
- **AR / mixed reality** – realistic insertion of virtual objects requires both layout and depth.

---

## Input → Output diagram

```
┌─────────────┐
│  Street      │
│  image       │
│  (H×W×3)    │
└──────┬──────┘
       │
       ├──► Semantic Segmentation  (H×W)  class per pixel
       │
       └──► Depth Map              (H×W)  distance per pixel (monocular)
```

---

## Assumptions & constraints
1. **Monocular input** – single RGB image (no stereo, no LiDAR).
2. **Street scenes** – road, vehicles, sky dominate; generalisation to indoor scenes is out of scope.
3. **Single forward pass** – real-time or near-real-time inference preferred.
4. **Evaluation on standard benchmarks** – Cityscapes (segmentation), KITTI (depth).
