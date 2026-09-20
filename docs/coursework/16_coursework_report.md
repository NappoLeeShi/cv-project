# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

> Báo cáo coursework (Step 16) — dự án **CV-PROJECT**.
> Mọi con số trong báo cáo được trích trực tiếp từ các file JSON kết quả và config của dự án:
> `outputs/analysis/unet_cityscapes_evaluation.json`, `outputs/analysis/unet_training_history.json`,
> `outputs/analysis/midas_kitti_evaluation.json`, `outputs/analysis/pipeline_evaluation.json`,
> `configs/unet.yaml`, `configs/midas.yaml`, `configs/pipeline.yaml`. Không có số liệu ước lượng.

---

# 1. Introduction

## 1.1 Background

**Scene Understanding** là lĩnh vực của computer vision nhằm giúp máy tính "hiểu" một cảnh (scene)
trong ảnh — không chỉ trả lời *"trong cảnh có đối tượng gì"* mà còn cho biết *"mỗi pixel thuộc lớp
nào"* và *"các đối tượng nằm ở khoảng cách nào"*. Hai kỹ thuật nền tảng được dùng trong dự án:

- **Semantic Segmentation** là quá trình gán một class (nhãn) cho từng pixel trong ảnh, ví dụ
  `road`, `sidewalk`, `car`, `person`, `sky`, ... Kết quả là một "mask" ngữ nghĩa cùng kích thước
  với ảnh gốc.
- **Monocular Depth Estimation** là bài toán ước lượng độ sâu của từng pixel từ **một** ảnh RGB
  duy nhất (không dùng stereo camera hay sensor). Trong dự án, mô hình MiDaS trả về
  **relative inverse depth** (độ sâu tỉ đối nghịch đảo): giá trị càng lớn nghĩa là càng **gần
  camera**, và giá trị này **không phải** khoảng cách tính bằng mét.

## 1.2 Motivation

Trong ảnh đường phố (driving scenes), ngữ nghĩa (semantic) và độ sâu (depth) bổ sung cho nhau:

- Chỉ có segmentation thì biết *"có một chiếc car"* nhưng không biết car đó gần hay xa.
- Chỉ có depth thì biết *"vùng này gần"* nhưng không biết vùng đó là `road`, `car` hay
  `pedestrian`.
- Kết hợp cả hai giúp xây dựng một **scene understanding report** hữu ích: class nào nằm ở vùng
  gần, có bao nhiêu đối tượng động (dynamic objects) trong cảnh, vùng drivable bao phủ tỉ lệ bao
  nhiêu, đối tượng động gần nhất là gì, và cảnh đó **khó** (difficult) ở mức độ nào.

Đây là nền tảng cho các hệ thống hỗ trợ lái (driver assistance) hoặc xe tự hành, nhưng dự án chỉ
dừng ở mức **phân tích cảnh**, không điều khiển phương tiện.

---

# 2. Problem Definition

Bài toán nhận **một ảnh RGB đường phố duy nhất** làm input, thực hiện đồng thời semantic
segmentation và monocular depth estimation trên **chính ảnh đó**, sau đó fuse (kết hợp) hai đầu ra
để phân tích cảnh và mức độ khó (difficulty) của cảnh.

## 2.1 Input

Một ảnh RGB duy nhất (PIL image hoặc numpy array `[H, W, 3]`), ví dụ ảnh đường phố từ
`data/pipeline/images/`. Yêu cầu quan trọng: **cùng một ảnh** được đưa vào cả hai mô hình
(gọi là *same-image contract*).

## 2.2 Output

- **Semantic Segmentation**: mask class `[H, W]` với các trainId 0..18 (Cityscapes, 19 lớp),
  class `255` biểu thị void/ignore.
- **Relative Inverse Depth**: bản đồ depth `[H, W]` — relative inverse depth, đơn vị tương đối,
  **không phải mét**; giá trị lớn hơn = gần camera hơn.
- **Fusion** (`FusionResult`): thống kê per-class (pixel count, mean/median inverse depth), vùng
  near / middle / far.
- **Scene Analysis**: báo cáo JSON gồm semantic distribution, depth distribution, traffic context
  (vehicles, pedestrians, nearest dynamic class), interpretation.
- **Difficulty Analysis**: điểm `difficulty_score` trong `[0, 1]` kèm mức **Easy / Medium / Hard**
  — đây là **heuristic**, không phải ground-truth difficulty.

---

# 3. Objectives

1. **Tái sử dụng các module đã có** của dự án (U-Net, MiDaS, fusion, analyzer, visualization,
   evaluation) mà không thiết kế lại kiến trúc.
2. **Đánh giá mô hình ở mức model-level**: U-Net trên Cityscapes val (mIoU, Pixel Accuracy,
   Mean Dice, per-class IoU) và MiDaS trên KITTI val (RMSE, MAE, AbsRel, δ1, δ2, δ3) — tách biệt,
   không trộn hai dataset.
3. **Xây dựng full-pipeline evaluation/demo**: ảnh RGB → segmentation → depth → fusion →
   scene analysis → difficulty analysis trên **cùng một ảnh**.
4. **Cung cấp công cụ CLI tái lập** (reproducible CLI) và các bài test offline chạy bằng
   mock/synthetic (không cần GPU thật hay dataset lớn).
5. **Tạo báo cáo coursework đầy đủ** (chính là file này) dựa trên số liệu thực từ JSON output.

---

# 4. Scope and Out of Scope

## 4.1 In Scope

- **Semantic Segmentation** — U-Net 19 class, huấn luyện trên Cityscapes.
- **Monocular Depth Estimation** — MiDaS DPT-Large pretrained, output relative inverse depth.
- **Feature Fusion** — fusion phân tích (rule-based) giữa segmentation và depth.
- **Scene Understanding** — báo cáo cảnh từ `scene_understanding/analyzer.py`.
- **Difficulty Analysis** — điểm khó heuristic, cấu hình được.
- **Model Evaluation** — U-Net/Cityscapes val (500 ảnh), MiDaS/KITTI val (1000 ảnh).
- **Visualization** — xuất ảnh original, segmentation, depth, fusion, overview, scene report.
- **Reproducible CLI Pipeline** — `python -m evaluation.evaluate_*`, `python -m main`.

## 4.2 Out of Scope

- **MiDaS training / fine-tuning** — MiDaS là pretrained, không huấn luyện trong dự án.
- **Object Detection and Tracking** — không bounding box, không tracking theo thời gian.
- **3D Reconstruction** — không tái tạo scene 3D.
- **Autonomous-driving control** — không điều khiển phương tiện.
- **Metric-depth prediction từ raw MiDaS output** — không quy đổi raw MiDaS thành mét.
- **Paired Cityscapes–KITTI benchmarking** — hai dataset không ghép cặp với nhau.
- **Ground-truth difficulty labeling** — difficulty chỉ là heuristic, không phải nhãn GT.

---

# 5. Dataset

## 5.1 Cityscapes

**Cityscapes** là dataset urban scene cho semantic segmentation và scene understanding, cung cấp
ảnh RGB đường phố đô thị cùng **ground-truth labels** gán nhãn từng pixel (remap về trainId
space 0..18 — 19 lớp; `255` = ignore).

Trong dự án:

- Cấu trúc: `data/cityscapes/` gồm `images/` và `labels/` (xem `preprocessing/cityscapes.py`).
- U-Net được **huấn luyện** trên Cityscapes train và **đánh giá** trên split **val**: theo
  `outputs/analysis/unet_cityscapes_evaluation.json`, số mẫu đánh giá là **500** (toàn bộ val split).
- Kích thước ảnh huấn luyện/đánh giá: `image_size: [256, 512]` (`data.image_size` trong
  `configs/unet.yaml`); normalization ImageNet mean `[0.485, 0.456, 0.406]`, std
  `[0.229, 0.224, 0.225]`.

## 5.2 KITTI

**KITTI** là bộ benchmark nổi tiếng cho autonomous driving (ảnh stereo màu, LiDAR, GPS...). Dự án
sử dụng **depth ground truth** từ KITTI raw data:

- Cấu trúc: `data/kitti/` gồm `images/` và `depth/` (depth uint16, đơn vị **millimetre**, quy đổi
  sang mét bằng `value / scale_mm` với `scale_mm=1000`; xem `preprocessing/kitti.py`).
- Depth `0` nghĩa là **invalid**; `depth_cap_m=80` đánh dấu pixel xa hơn 80 m là invalid; valid
  được truyền qua `valid_mask`.
- Đánh giá trên split **val**, số mẫu **1000** (`num_samples=1000`, `num_expected=1000`,
  `skipped=0` trong `midas_kitti_evaluation.json`).
- Loader ghép cặp ảnh–depth **theo stem của tên file** (không theo thứ tự sắp xếp) nên không thể
  lệch cặp.

## 5.3 Dataset Relationship

**Cityscapes và KITTI KHÔNG phải hai dataset được ghép cặp (paired).** Chúng có camera, hình học
chụp và hệ nhãn khác nhau. Luật bất biến của dự án:

- **Không bao giờ** ghép ảnh Cityscapes với depth map KITTI.
- **Model-level evaluation** được thực hiện **tách biệt**:
  - U-Net + Cityscapes GT → mIoU, Pixel Accuracy, Dice.
  - MiDaS + KITTI GT → RMSE, MAE, AbsRel, δ1, δ2, δ3.
- **Full pipeline** dùng **một ảnh RGB duy nhất** làm đầu vào cho cả hai mô hình (same-image
  contract) trên các ảnh street **standalone** — không cần ground-truth nào của hai dataset trên.

---

# 6. Methodology

## 6.1 Overall Architecture

```
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
```

Quá trình: một ảnh RGB → U-Net sinh segmentation mask và MiDaS sinh relative inverse depth → căn
chỉnh không gian (alignment) về đúng resolution của ảnh nguồn → fusion → scene analysis →
difficulty analysis.

## 6.2 U-Net

**U-Net** (`models/unet/model.py`) là mạng **encoder–bottleneck–decoder** (contracting path –
bottleneck – expanding path) với **skip connections** nối đặc trưng cùng cấp giữa encoder và
decoder, giúp giữ lại chi tiết không gian — rất quan trọng cho segmentation.

- Cấu hình thực tế (`configs/unet.yaml`): `num_classes=19`, `in_channels=3`, `base_channels=64`.
- Đầu ra là **logits** `[C, H, W]`; mask class = `argmax(dim=1)`.
- Ảnh resize về kích thước mô hình bằng **bilinear**; mask resize ngược về ảnh nguồn bằng
  **nearest-neighbour** (không dùng bilinear cho class ID).
- Loss: **weighted cross-entropy** (cân bằng class); optimizer **Adam**; checkpoint
  `checkpoints/unet_cityscapes.pth` (huấn luyện 20 epochs — chi tiết mục 8.1).

## 6.3 MiDaS DPT-Large

**MiDaS DPT-Large** (`models/midas/model.py`, `inference.py`) là mô hình monocular depth
estimation **pretrained** dùng **Vision Transformer (DPT)** làm backbone, loaded từ checkpoint
`checkpoints/dpt_large_384.pt` với input size `384`, chạy trong `no_grad`/eval, output được resize
bilinear về resolution nguồn.

**Semantic quan trọng về đầu ra:**

> Raw MiDaS output là **relative inverse depth**, KHÔNG phải depth metric (mét).
> Giá trị lớn hơn = vùng gần camera hơn. Dự án không bao giờ mô tả raw MiDaS là mét.
> **Median scaling** chỉ được dùng ở evaluation time (Step 14) để align với GT KITTI; trong demo
> pipeline (Step 15) bản đồ depth giữ nguyên nghĩa relative.

## 6.4 Feature Fusion

Fusion là **analytical (rule-based)**, không phải mạng nơ-ron học. Logic nằm trong
`scene_understanding/fusion.py`:

- Yêu cầu segmentation và depth **cùng kích thước không gian** (ràng buộc same-image).
- Pixel depth không hữu hạn (NaN/inf) và pixel `void (255)` bị loại khỏi mọi thống kê thông qua
  `analyzed_mask`.
- **Per-class stats**: `pixel_count`, `pixel_ratio`, `mean_depth`, `median_depth`, `min_depth`,
  `max_depth` cho mỗi class xuất hiện.
- **Depth regions**: ngưỡng được dẫn xuất từ chính depth map (default terciles
  `[0.3333, 0.6667]`): `near` = 1/3 giá trị lớn nhất, `middle`, `far`. Quy ước:
  **larger inverse depth = closer**.
- Kết quả là `FusionResult` chứa `per_class`, `region_map`, `region_summary`, `thresholds`.

## 6.5 Scene Analysis

`scene_understanding/analyzer.py` (`analyze_fusion`) chuyển `FusionResult` thành báo cáo JSON:

- `scene` — kích thước, số lớp ngữ nghĩa, `depth_convention: "inverse_relative_larger_closer"`.
- `semantic_distribution` — pixel count từng class.
- `depth_distribution` — tỉ lệ near/middle/far.
- `regions` — chi tiết per-class depth.
- `traffic_context` — vehicles, pedestrians, road, `drivable_coverage_ratio`,
  `drivable_median_depth`, `dynamic_object_count`, `nearest_dynamic_class`.
- `interpretation` — các dòng mô tả, ví dụ:
  *"The nearest dynamic class is 'car' (near relative-depth region)."*

## 6.6 Difficulty Analysis

`evaluation/difficulty_analysis.py` tính **pipeline difficulty score** — một **heuristic** minh
bạch, không phải ground-truth difficulty. Công thức đúng như cài đặt:

```
D = 0.25·U + 0.20·V + 0.20·C + 0.20·F + 0.15·O
```

trong đó:

- `U` = **segmentation uncertainty** = `1 - mean_confidence` (mean_confidence là max softmax
  probability trung bình của U-Net trên ảnh).
- `V` = **depth variation** = `std / mean` của finite inverse depth (coefficient of variation).
- `C` = **scene complexity** = số class xuất hiện / 19 (tổng class Cityscapes trainId).
- `F` = **foreground fraction** = tỉ lệ pixel thuộc vùng *near* (depth region).
- `O` = **object density** = tỉ lệ pixel thuộc các class ROI (dynamic objects, person..bicycle,
  trainId 11..18).

Mỗi indicator được clamp về `[0, 1]`, điểm số clamp về `[0, 1]` và phân loại theo ngưỡng thực tế
trong `configs/pipeline.yaml` (`difficulty.bins`):

| Điều kiện | Mức |
|---|---|
| `score <= 0.4` (bins.easy) | **Easy** |
| `score <= 0.7` (bins.medium) | **Medium** |
| else | **Hard** |

Đây là **điểm khó heuristic của pipeline** (pipeline difficulty score / scene complexity score),
**không** phải nhãn difficulty ground-truth.

---

# 7. Implementation

## 7.1 Project Structure

```
cv-project/
├── configs/            # unet.yaml, midas.yaml, pipeline.yaml
├── data/
│   ├── cityscapes/     # images/, labels/
│   ├── kitti/          # images/, depth/
│   └── pipeline/images/ # ảnh demo RGB standalone (gta5, pipeline_demo_001/002, streetest)
├── models/
│   ├── unet/           # model.py (UNet), inference.py (UNetInference)
│   └── midas/          # model.py (MiDaSModel...), inference.py (MidDepthPredictor)
├── preprocessing/      # cityscapes.py, kitti.py, transforms.py
├── scene_understanding/# fusion.py, analyzer.py, pipeline.py
├── visualization/      # segmentation.py, depth.py, fusion.py, scene.py, io.py
├── evaluation/         # evaluate_unet.py, evaluate_midas.py, evaluate_pipeline.py,
│                       # difficulty_analysis.py, segmentation_metrics.py,
│                       # depth_metrics.py, pipeline_metrics.py (placeholder)
├── utils/              # config.py, device.py, seed.py, logger.py
├── training/           # train_unet.py, trainer.py
├── tests/              # bộ test pytest (offline, synthetic)
├── checkpoints/        # unet_cityscapes.pth, dpt_large_384.pt
├── outputs/
│   ├── analysis/       # unet_*/midas_*/pipeline_evaluation.json, demo_*.png/json
│   ├── segmentation/   # ảnh original + segmentation demo
│   ├── depth/          # ảnh depth demo
│   └── visualization/  # smoke test
├── main.py             # demo CLI cho ảnh tuỳ chọn (Step 17)
└── prompts/            # các bước công việc (steps 00-17)
```

## 7.2 Data Preprocessing

- **Shared transforms** `preprocessing/transforms.py`: `ImageTransform` (resize + chuẩn hoá
  ImageNet mean/std), `DepthTransform`.
- **Cityscapes**: label raw (labelIds 0..33) được remap sang **trainId 0..18, 255 = void** qua
  bảng tra cứu chính thức; label resize bằng nearest-neighbour.
- **KITTI**: depth uint16 (mm) → mét (`/scale_mm=1000`); `0` = invalid (`valid_mask`); `depth_cap_m=80`
  giới hạn; bilinear depth resize rồi recompute valid mask để không leak vào pixel invalid.
- Quan trọng: loaders ghép cặp theo stem tên file (không theo thứ tự sắp xếp) → tránh lệch cặp.

## 7.3 Model Training and Inference

- **U-Net**: huấn luyện trên Cityscapes train qua `training/trainer.py` (`UNetTrainer`) và CLI
  `training/train_unet.py` (Step 12). Inference qua `UNetInference.predict` (eval, `no_grad`), tuỳ
  chọn trả về `(prediction, confidence)` với `return_confidence=True`.
- **MiDaS**: load `checkpoints/dpt_large_384.pt` vào backend DPT-Large (dùng workaround hub
  `torch.hub.load(MODEL_SOURCE, "DPT_Large", pretrained=False)` vì upstream export tên `DPT_Large`
  trong khi dự án chuẩn hoá `dpt_large`), inference qua `MidDepthPredictor.predict` (eval,
  `no_grad`), output relative inverse depth.

## 7.4 Evaluation

- **Model-level**:
  - `evaluation/evaluate_unet.py` — Cityscapes val (500 ảnh), dùng **một global confusion
    matrix** (bỏ qua `255`) rồi tính Pixel Accuracy, mIoU, Mean Dice, per-class IoU.
  - `evaluation/evaluate_midas.py` — KITTI val (1000 ảnh), mỗi ảnh alignment độc lập bằng
    **median scaling** `scale = median(gt_valid) / median(pred_valid)`, tính
    RMSE/MAE/AbsRel/δ1/δ2/δ3 rồi lấy trung bình per-image.
- **Pipeline-level**:
  - `evaluation/evaluate_pipeline.py` — cùng một ảnh vào U-Net và MiDaS → align → fuse →
    `analyze_fusion` → `compute_difficulty` → JSON + visualization (Step 15).

## 7.5 Visualization

Các module trong `visualization/`:

- `segmentation.py` — `colorize_segmentation` (palette cố định 19 màu Cityscapes), overlay, legend.
- `depth.py` — `colorize_depth` (colormap `turbo`); nhãn luôn ghi rõ
  *"Relative inverse depth (larger = closer)"*, không ghi mét.
- `fusion.py` — `create_fusion_overlay` (segmentation + depth), region map near/middle/far.
- `scene.py` — `create_full_visualization` (figure tổng hợp), `save_scene_report` (JSON).
- `io.py` — `save_visualization`, `save_figure`.

## 7.6 CLI and Reproducibility

```bash
# Toàn bộ test suite (offline, không cần model/dataset tải về)
.venv/bin/python -m pytest -q

# Model-level U-Net (Cityscapes val)
.venv/bin/python -m evaluation.evaluate_unet --checkpoint checkpoints/unet_cityscapes.pth

# Model-level MiDaS (KITTI val)
.venv/bin/python -m evaluation.evaluate_midas --checkpoint checkpoints/dpt_large_384.pt

# Full-pipeline smoke test (2 ảnh RGB, CUDA nếu có)
.venv/bin/python -m evaluation.evaluate_pipeline \
    --input-dir data/pipeline/images \
    --limit 2 --save-visualizations --device auto

# Demo ảnh tuỳ chọn (Step 17)
.venv/bin/python main.py --input-dir data/pipeline/images
```

Seed cố định (`system.seed: 42` trong configs) giúp tái lập; config hash được ghi vào log.

---

# 8. Experimental Setup

## 8.1 U-Net Training

Siêu tham số thực tế từ `configs/unet.yaml` (nhóm `training:`) và
`outputs/analysis/unet_training_history.json`:

| Tham số | Giá trị |
|---|---|
| Architecture | U-Net (encoder–bottleneck–decoder, skip connections) |
| `num_classes` | 19 |
| `in_channels` / `base_channels` | 3 / 64 |
| Image size (train/val) | `[256, 512]` |
| Batch size | 1 (bản gốc batch 4 OOM trên GPU 4 GB) |
| Learning rate | `0.0001` |
| Optimizer | Adam (`adam`) |
| Weight decay | `0.00001` |
| Epochs | 20 |
| Mixed precision | `false` (AMP tắt) |
| Loss | weighted cross-entropy (`weighted_ce`) |
| Seed | 42 |

**Training history** (từ `unet_training_history.json`):

- `best_epoch` = **20**, `best_val_loss` = **0.31346111369878055**.
- `best_val_miou` = **0.4262424504443624** (trùng `training_val_miou` trong evaluation JSON và
  `checkpoint_epoch: 20`).
- `training_val_pixel_accuracy` = **0.9021092620377307**.
- History gồm 19 bản ghi, epoch 2 → 20 (epoch 1 vắng).

## 8.2 U-Net Evaluation

- Split: **Cityscapes val**; số mẫu: **500**.
- `image_size`: `[256, 512]`; `ignore_index`: `255`; `num_classes`: `19`.
- Thủ tục: load `checkpoints/unet_cityscapes.pth` (epoch 20) → chạy eval/`no_grad` → gộp một
  **confusion matrix toàn cục** → tính Pixel Accuracy, mIoU, Mean Dice, per-class IoU.

## 8.3 MiDaS Evaluation

- Split: **KITTI val**; số mẫu: **1000** (`num_samples=1000`, `num_expected=1000`, `skipped=0`).
- **Median scaling (evaluation-time only)**: mỗi ảnh được align độc lập với
  `scale = median(gt_valid) / median(pred_valid)` trước khi tính lỗi;
  `mean_scale = 0.22169487541812957`. Đây là alignment để so sánh, **không** biến MiDaS thành
  depth mét.
- `depth_cap_m = 80`; depth 0 = invalid (chỉ tính trên valid pixels).
- Metadata JSON xác nhận: `relative_inverse_depth: true`, `larger_value_is_closer: true`,
  `metric_alignment: median`, `median_scaling: true`.

## 8.4 Full Pipeline Smoke Test

- Nguồn ảnh: `data/pipeline/images/` — ảnh RGB street **standalone**; JSON hiện tại đánh giá
  **2 ảnh**: `pipeline_demo_001.png`, `pipeline_demo_002.png`.
- Lệnh tham chiếu: `--limit 2 --save-visualizations --device auto`.
- Pipeline: same-image → U-Net → MiDaS → align → fuse → scene analysis → difficulty.
- **Quan trọng**: đây là **smoke test 2 ảnh** để xác minh tính khả thi (checkpoint load, CUDA,
  same-image, alignment, fusion, analyzer, difficulty, visualization, JSON). **KHÔNG** được coi là
  benchmark có ý nghĩa thống kê.

---

# 9. Results

## 9.1 U-Net Results

Từ `outputs/analysis/unet_cityscapes_evaluation.json` (Cityscapes **val**, 500 ảnh,
checkpoint epoch 20):

| Metric | Result |
|---|---|
| Pixel Accuracy | 0.9042111912537422 |
| mIoU | 0.445074137144175 |
| Mean Dice | 0.5498125290480808 |
| val mIoU ghi trong checkpoint (epoch 20) | 0.4262424504443624 |

## 9.2 MiDaS Results

Từ `outputs/analysis/midas_kitti_evaluation.json` (KITTI **val**, 1000 ảnh, median scaling):

| Metric | Result |
|---|---|
| RMSE | 4.256132507952642 |
| MAE | 3.018216943917585 |
| AbsRel (absrel) | 0.8489374433912796 |
| δ1 (delta1) | 0.16709821565801006 |
| δ2 (delta2) | 0.3275189399048214 |
| δ3 (delta3) | 0.4781034299876558 |
| mean scale (median) | 0.22169487541812957 |

Giải thích: các δ thấp và RMSE/MAE/AbsRel tương đối cao phản ánh việc chiếu relative inverse
depth của MiDaS sang thang depth tuyến tính của KITTI sau median scaling còn sai lệch lớn —
median scaling chỉ hiệu chỉnh mức tổng thể, không phải độ phi tuyến cục bộ (chi tiết Discussion).

## 9.3 Per-Class Segmentation Results

IoU từng lớp (Cityscapes val) từ `unet_cityscapes_evaluation.json`:

| Class | IoU | Class | IoU |
|---|---|---|---|
| road | 0.9522424488781998 | person | 0.5279796058947245 |
| sidewalk | 0.6792857854814197 | rider | 0.011459488791885788 |
| building | 0.8237347214169852 | car | 0.8436484341008157 |
| wall | 0.19122455834870059 | truck | 0.07598792344364184 |
| fence | 0.18809036593962372 | bus | 0.16566493200137167 |
| pole | 0.39857740293790717 | train | 0.13067776671794407 |
| traffic light | 0.21834830684174153 | motorcycle | 0.04224830903969902 |
| traffic sign | 0.500585585377887 | bicycle | 0.530411461903086 |
| vegetation | 0.8630465977246684 | | |
| terrain | 0.42901382203064126 | | |
| sky | 0.8841810888683829 | | |

Nhận xét: các lớp chiếm diện tích lớn — `road` (0.952), `sky` (0.884), `vegetation` (0.863),
`car` (0.844), `building` (0.824) — đạt IoU cao; các lớp hiếm/nhỏ — `rider` (0.011),
`motorcycle` (0.042), `truck` (0.076), `wall` (0.191), `fence` (0.188) — có IoU rất thấp.

## 9.4 Full Pipeline Results

Kết quả **smoke test** (2 ảnh) từ `outputs/analysis/pipeline_evaluation.json`:

| Image | Difficulty Score | Level |
|---|---|---|
| pipeline_demo_001.png | 0.541012 | **Medium** |
| pipeline_demo_002.png | 0.508277 | **Medium** |

**Aggregate:**

| Indicator | Value |
|---|---|
| Number of images | 2 |
| Average difficulty score | 0.524644 |
| Easy / Medium / Hard | 0 / 2 / 0 |
| Average segmentation confidence | 0.661301 |
| Average depth variation (std) | 9.042071 |

Chi tiết từng ảnh:

| Image | classes present | mean conf | mean inv. depth | std inv. depth | min | max | scene size |
|---|---|---|---|---|---|---|---|
| pipeline_demo_001.png | 19 | 0.646535 | 13.72307 | 9.546155 | 0.432997 | 45.738678 | 352×1216 |
| pipeline_demo_002.png | 18 | 0.676068 | 13.27886 | 8.537987 | 0.808746 | 32.905846 | 352×1216 |

Component difficulty từng ảnh (khớp công thức mục 6.6; đã đối chiếu lại
`0.541012 = 0.25×0.353465 + 0.20×0.695628 + 0.20×1.0 + 0.20×0.333335 + 0.15×0.312353`):

| Image | U | V | C | F | O |
|---|---|---|---|---|---|
| pipeline_demo_001.png | 0.353465 | 0.695628 | 1.0 | 0.333335 | 0.312353 |
| pipeline_demo_002.png | 0.323932 | 0.642976 | 0.947368 | 0.333335 | 0.283717 |

Đối tượng động gần nhất (nearest dynamic class) của cả 2 ảnh đều là **`car`** ở vùng **near**;
`dynamic_object_count`: 8 (ảnh 1), 7 (ảnh 2).

> ⚠️ Đây chỉ là **smoke test 2 ảnh**, không diễn giải như benchmark thống kê; phân bố
> Easy/Medium/Hard thu được chỉ mang tính minh hoạ.

---

# 10. Discussion

## 10.1 U-Net Discussion

Dựa trên kết quả thực tế (Cityscapes val, 500 ảnh):

- **Điểm mạnh**: Pixel Accuracy ≈ 0.904 và mIoU ≈ 0.445 là kết quả hợp lý cho một U-Net nhỏ
  (base 64) huấn luyện 20 epochs ở `[256, 512]` batch 1. Các lớp chiếm diện tích lớn đạt IoU cao
  (`road` 0.952, `sky` 0.884, `vegetation` 0.863, `car` 0.844, `building` 0.824).
- **Điểm yếu**: các lớp hiếm/nhỏ bị thấp: `rider` 0.011, `motorcycle` 0.042, `truck` 0.076,
  `wall` 0.191, `fence` 0.188, `traffic light` 0.218. Nguyên nhân chính là **class imbalance**,
  resolution huấn luyện thấp so với Cityscapes gốc (1024×2048), và số epochs ít. Confusion chủ
  yếu giữa các lớp hiếm với lớp nền.
- Lưu ý đo lường: mIoU trong evaluation JSON (0.4451) khác `training_val_miou` (0.4262) vì
  evaluation dùng một confusion matrix toàn cục trên suốt tập val còn training log tính trung
  bình val theo epoch — đây là hai cách aggregate hợp lệ khác nhau.

## 10.2 MiDaS Discussion

- MiDaS là mô hình **zero-shot pretrained**, không được huấn luyện trên KITTI trong dự án.
- Kết quả KITTI val (RMSE 4.256, MAE 3.018, AbsRel 0.849, δ1 0.167) cho thấy khi **median-scaled**
  sang không gian depth tuyến tính của KITTI, độ khớp còn hạn chế — hệ quả chính đáng vì MiDaS
  dự đoán **relative inverse depth** (disparity-like) có thang đo và phân bố khác, và median
  scaling chỉ căn chỉnh mức tổng thể (một hằng số) chứ không hiệu chỉnh phi tuyến cục bộ.
- **Quan trọng**: dự án không bao giờ khẳng định raw MiDaS là depth mét. Giá trị lớn hơn = gần
  camera hơn — thuận lợi cho fusion vì vùng gần tương ứng disparity lớn.
- Trong pipeline demo, bản đồ depth giữ nguyên nghĩa tương đối, chỉ dùng để phân vùng
  near/middle/far và tính depth variation.

## 10.3 Fusion and Scene Understanding

Kết hợp semantic + depth giúp trả lời các câu hỏi mà từng nguồn riêng không đủ:

- Xác định **nearest dynamic class**: cả 2 ảnh demo đều cho `car` ở vùng near.
- Đưa ra `drivable_coverage_ratio` (ảnh 1: 0.1317, ảnh 2: 0.1701) và `drivable_median_depth` —
  thông tin về vùng lòng đường và khoảng cách tương đối của nó.
- Phân bố depth near/middle/far gần như 1/3–1/3–1/3 trong smoke test (do ngưỡng tercile được dẫn
  xuất từ chính depth map của mỗi ảnh) — nghĩa là fusion phân chia theo phân phối riêng của từng
  ảnh, không phải ngưỡng cố định.
- `interpretation` bằng ngôn ngữ tự nhiên giúp trình bày kết quả trực tiếp.

## 10.4 Difficulty Analysis

- Điểm difficulty là **heuristic** tổ hợp tuyến tính của 5 indicator chuẩn hoá (công thức mục
  6.6) với trọng số cấu hình được (`0.25/0.20/0.20/0.20/0.15` từ `configs/pipeline.yaml`).
- Cả 2 ảnh demo đều rơi vào **Medium** (0.541 và 0.508) — chủ yếu do `scene_complexity` cao
  (nhiều class nhưng lệ thuộc phần lớn vào 19 class Cityscapes) và `depth_variation` cao;
  `foreground_fraction` cố định ~0.333 do tercile.
- `segmentation_uncertainty` đóng góp vừa phải (~0.32–0.35). Với 2 ảnh không thể suy luận gì về
  phân phối tổng thể của difficulty.
- Điểm này nhạy cảm với trọng số do con người chọn; chỉ có nghĩa khi được diễn giải như một
  **chỉ số phức tạp cảnh tương đối**, không phải nhãn ground-truth.

## 10.5 Dataset Limitations

- Cityscapes và KITTI **không ghép cặp**; không thể dùng GT của dataset này để đánh giá model
  của dataset kia.
- Model-level metrics của U-Net và MiDaS đến từ **hai benchmark riêng biệt** với GT khác nhau
  (semantic label vs depth map).
- Pipeline demo dùng ảnh RGB **independent, không ground-truth** → kết quả pipeline mang tính minh
  hoạ/định tính, không đo được độ chính xác tuyệt đối.
- U-Net chỉ đánh giá trên 500 ảnh val (toàn bộ Cityscapes val); MiDaS trên 1000 ảnh KITTI val.
  Generalization ngoài phân phối (domain shift) chưa được đo.

---

# 11. Limitations

1. **Model-level thấp ở lớp hiếm** — U-Net cho IoU rất thấp với `rider`, `motorcycle`, `truck`,
   `wall`, `fence` do class imbalance + resolution thấp + ít epochs.
2. **MiDaS chỉ là relative inverse depth** — không thể đưa ra khoảng cách tuyệt đối (mét); mọi
   phân tích depth đều mang tính tương đối (near/middle/far). Median scaling chỉ là alignment ở
   evaluation time.
3. **Smoke test pipeline chỉ có 2 ảnh** — không có ý nghĩa thống kê; không nên đánh giá chất
   lượng pipeline từ con số difficulty của 2 ảnh.
4. **Difficulty chỉ là heuristic** — không phải difficulty ground-truth; trọng số/ngưỡng do con
   người chọn, cần hiệu chỉnh theo tác vụ cụ thể.
5. **Không có ground-truth cho pipeline** — pipeline-level không được đo bằng metric khách quan
   (ảnh demo độc lập); các chỉ số đều suy diễn từ output của mô hình.
6. **Chi phí tính toán** — U-Net chạy ở `[256, 512]`/batch 1 (do VRAM ~4 GB), không dùng AMP;
   MiDaS DPT-Large chạy ở `input_size=384` không GPU 4 GB khi cùng lúc với U-Net bị giới hạn bởi
   resolution inference (xem `_safe_unet_inference_size`).
7. **CI/CD không được triển khai** — mọi kiểm tra đều chạy thủ công bằng pytest cục bộ; không có
   pipeline CI chạy tự động.

---

# 12. Conclusion

Dự án đã xây dựng và đánh giá một **full-pipeline scene understanding** cho ảnh đường phố bằng
cách kết hợp **U-Net** (semantic segmentation, 19 lớp, huấn luyện trên Cityscapes) và
**MiDaS DPT-Large** (pretrained, relative inverse depth) trên **cùng một ảnh RGB**:

- U-Net đạt **Pixel Accuracy ≈ 0.9042**, **mIoU ≈ 0.4451** trên 500 ảnh Cityscapes val; các lớp
  phổ biến (road, building, vegetation, sky, car) đạt IoU cao, lớp hiếm còn yếu.
- MiDaS được đánh giá trên 1000 ảnh KITTI val với median scaling (RMSE 4.256, MAE 3.018,
  AbsRel 0.849, δ1 0.167), luôn được mô tả là **relative inverse depth** chứ không phải mét.
- Pipeline **same-image**: ảnh → segmentation + depth → fusion (rule-based) → scene report →
  difficulty heuristic (D = 0.25U + 0.20V + 0.20C + 0.20F + 0.15O). Smoke test trên 2 ảnh cho cả
  hai đều xếp **Medium**; đối tượng động gần nhất là `car` (near).
- Toàn bộ quy trình tái lập được qua CLI, kèm bộ test offline
  (`pytest -q` → **456 passed, 1 skipped**), JSON kết quả và visualization đầy đủ.

Kết quả chứng minh tính khả thi của việc kết hợp ngữ nghĩa và depth tỉ đối cho scene understanding
trên ảnh đường phố, đồng thời phơi bày các giới hạn rõ ràng (lớp hiếm, tính tương đối của depth,
difficulty heuristic, smoke test 2 ảnh) — là căn cứ cho các bước mở rộng sau này.

---

# 13. References

1. **U-Net** — O. Ronneberger, P. Fischer, T. Brox, *U-Net: Convolutional Networks for Biomedical
   Image Segmentation*, MICCAI 2015.
2. **MiDaS** — R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, V. Koltun, *Towards Robust
   Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer*, IEEE TPAMI
   2022.
3. **DPT (MiDaS DPT-Large backbone)** — R. Ranftl, A. Bochkovskiy, V. Koltun, *Vision Transformers
   for Dense Prediction*, ICCV 2021.
4. **Cityscapes** — M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
   U. Franke, S. Roth, B. Schiele, *The Cityscapes Dataset for Semantic Urban Scene Understanding*,
   CVPR 2016.
5. **KITTI** — A. Geiger, P. Lenz, R. Urtasun, *Are we ready for Autonomous Driving? The KITTI
   Vision Benchmark Suite*, CVPR 2012.

> Ghi chú: các tham chiếu ở trên là tài liệu gốc nổi tiếng của từng phương pháp/dataset; dự án
> không trích dẫn DOI hay số trang, chỉ nêu tên công trình tương ứng.

---

*Báo cáo được tạo từ dữ liệu thực của dự án. Mọi con số đều khớp với:*
`outputs/analysis/unet_cityscapes_evaluation.json`, `outputs/analysis/unet_training_history.json`,
`outputs/analysis/midas_kitti_evaluation.json`, `outputs/analysis/pipeline_evaluation.json`,
`configs/unet.yaml`, `configs/midas.yaml`, `configs/pipeline.yaml`.