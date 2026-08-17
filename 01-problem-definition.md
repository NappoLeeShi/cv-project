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
┌─────────────────┐
│ Street Image    │
│ (H × W × 3)     │
└────────┬────────┘
         │
 ┌───────┴────────┐
 │                │
 ▼                ▼

U-Net           MiDaS
(Segmentation)  (Depth)

 ▼                ▼

Semantic       Depth
Mask           Map

      │
      ▼

Scene Understanding
```

---

## Assumptions & constraints
1. **Monocular input** – only a single RGB image is used..
2. **Street scenes** – road environments containing vehicles, pedestrians, buildings, and traffic infrastructure.
3. **Separate prediction models** – segmentation and depth estimation are performed using dedicated models (U-Net and MiDaS).
4. **Near real-time inference preferred** – for practical traffic-scene analysis.
5. **Evaluation on public benchmarks:**
- Cityscapes for semantic segmentation.
- KITTI for depth estimation.