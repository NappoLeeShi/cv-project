# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

## Coursework Report — Step 06: Evaluation

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Tài liệu này mô tả **cách project đánh giá** hệ thống ở hai cấp: **model-level** (U-Net trên Cityscapes, MiDaS trên KITTI) và **pipeline-level** (full pipeline trên cùng một ảnh RGB).
> Nguồn sự thật: code (`evaluation/segmentation_metrics.py`, `evaluation/depth_metrics.py`, `evaluation/evaluate_*.py`), các file kết quả `outputs/analysis/*.json`, `configs/*.yaml`, `tests/` và các doc 00–05.
> Mọi con số trong tài liệu này đều **đọc từ JSON hiện tại của repository** — không ước lượng, không bịa kết quả.

---

## 1. REQUIREMENT

### 1.1 Evaluation Objectives

Đánh giá là cần thiết vì:

- **Model-level**: phải biết mỗi model học có tốt không — U-Net phân đoạn đúng tới đâu, MiDaS ước lượng độ sâu chính xác ra sao so với ground truth riêng của nó;
- **Pipeline-level**: phải biết **cả hệ thống ghép lại** có hoạt động đúng không — cùng ảnh đi qua U-Net + MiDaS → Fusion → Scene Analysis → Difficulty có chạy trơn tru và cho báo cáo cảnh hợp lý không;
- Đánh giá tạo **dữ liệu để trình bày** (metric, con số, hình ảnh) và là **cổng kiểm tra quy hồi** khi thêm bước mới.

### 1.2 Evaluation Levels

#### Model-Level Evaluation

Mỗi model được đánh giá **độc lập** với ground truth phù hợp của riêng nó:

- **U-Net** ↔ Cityscapes: ảnh RGB + segmentation label → so khớp phân đoạn → segmentation metrics.
- **MiDaS** ↔ KITTI: ảnh RGB + depth map (mét) → so khớp sau alignment → depth metrics.

#### Pipeline-Level Evaluation

Pipeline được đánh giá theo **hành vi của toàn hệ thống** trên cùng một ảnh RGB, **không cần ground-truth ghép cặp**:

```text
Một ảnh RGB
→ U-Net (segmentation)
→ MiDaS (relative inverse depth)
→ Fusion
→ Scene Analysis
→ Difficulty Analysis
→ JSON + visualization
```

**Hai cấp không được lẫn:** metric model trả lời câu hỏi *"model có giỏi không so với GT?"*, còn pipeline trả lời *"hệ thống ghép lại có chạy đúng và cho hiểu cảnh gì?"*. Một U-Net mIoU cao chưa chứng minh pipeline tốt; pipeline chạy đẹp cũng không chứng minh model giỏi.

---

## 2. EVALUATION DESIGN

### 2.1 Overall Evaluation Flow

```text
[Model-level]

Cityscapes (RGB + semantic GT)
  → U-Net
  → segmentation prediction (trainId 0..18)
  → Segmentation metrics (Pixel Acc / mIoU / Dice)

KITTI (RGB + depth GT metric)
  → MiDaS
  → relative inverse depth
  → median scaling alignment (evaluation-time)
  → Depth metrics (RMSE / MAE / AbsRel / δ1..δ3)

[Pipeline-level]

Custom RGB image (standalone)
  → U-Net + MiDaS (cùng một ảnh)
  → spatial alignment (nearest cho class ID, bilinear cho depth)
  → Fusion (rule-based)
  → Scene Analysis (JSON report)
  → Difficulty Analysis (heuristic: easy/medium/hard)
```

### 2.2 Dataset Separation

- **Cityscapes** dùng để đánh giá semantic segmentation của U-Net.
- **KITTI** dùng để đánh giá depth của MiDaS.
- Chúng là **hai dataset riêng biệt, KHÔNG ghép cặp với nhau.**
- Một ảnh Cityscapes **không bao giờ** được ghép với một depth map của KITTI.
- Pipeline dùng **một ảnh RGB duy nhất** cho cả hai model và **không cần** GT ghép chéo giữa hai dataset.

---

## 3. U-NET EVALUATION

### 3.1 Evaluation Input

Các giá trị để U-Net evaluation (verify từ `outputs/analysis/unet_cityscapes_evaluation.json`):

| Item | Value |
|---|---|
| Dataset | Cityscapes |
| Split | `val` |
| Number of samples | **500** (đúng 500 ảnh val chính thức) |
| Image size | `[256, 512]` (scale-thấp so với native 1024×2048) |
| Checkpoint | `checkpoints/unet_cityscapes.pth` (best epoch 20) |
| num_classes | 19 (trainId) |
| Ignore index | `255` (Cityscapes `gtFine` void/ignore) |

### 3.2 Evaluation Procedure

Thực hiện trong `evaluation/evaluate_unet.py` (`evaluate_dataset`):

1. **Load checkpoint** qua `torch.load(weights_only=True)`, yêu cầu dict chứa `model_state_dict` — nếu format không đúng sẽ báo lỗi.
2. **Cross-check task:** `num_classes`/`ignore_index` trong checkpoint metadata phải khớp config; nếu lệch (ví dụ checkpoint 5 class mà config 19) sẽ **raise**, không chấm điểm lặng lẽ.
3. **Evaluation mode:** `model.eval()`, toàn bộ forward nằm trong `torch.no_grad()` — model **không bao giờ được train/cập nhật** ở bước này.
4. **Inference:** logits → `argmax(dim=1)` → prediction integer.
5. **Ground-truth handling:** so với label trainId `0..18`; pixel `255` được loại khỏi mọi tính toán.
6. **Confusion matrix:** mỗi batch cộng dồn `confusion_matrix(...)` vào **một global matrix** `(19, 19)`.
7. **Metric calculation:** từ global matrix tính `pixel_accuracy`, `iou_per_class`, `miou`, `dice_per_class`, `mean_dice`, `class_accuracy` qua `aggregate_from_confusion_matrix`.
8. **Output:** JSON tại `outputs/analysis/unet_cityscapes_evaluation.json` (per-class theo tên class, kèm metadata checkpoint).

### 3.3 Confusion Matrix

Trong semantic segmentation, confusion matrix là bảng **`num_classes × num_classes`** với:

- **Rows = ground-truth class** (class thật);
- **Columns = predicted class** (class model dự đoán);
- Mỗi ô `CM[i][j]` = số pixel có GT là class `i` nhưng được dự đoán là class `j`.

Từ đó suy ra cho từng class:

- **True Positive (TP)** = `CM[i][i]` — pixel nhận đúng;
- **False Positive (FP)** = tổng cột `i` trừ TP — pixel không phải `i` lại được dự đoán là `i`;
- **False Negative (FN)** = tổng hàng `i` trừ TP — pixel là `i` nhưng bị dự đoán class khác.

Pixel GT == `255` **bị loại hoàn toàn trước khi xây matrix**, nên `255` không bao giờ thành một class. **Global matrix** (cộng dồn mọi ảnh) hữu dụng vì: chỉ một lần đếm pixel, không cần giữ toàn bộ tensor H×W của 500 ảnh trong RAM, và cho số liệu đủ lớn để metric ổn định.

### 3.4 Metrics

Các metric đúng code trong `evaluation/segmentation_metrics.py`:

| Metric | Formula (đúng code) | Meaning | Cao hơn = tốt hơn |
|---|---|---|---|
| **Pixel Accuracy** | `correct_valid_pixels / total_valid_pixels` (valid = GT ≠ 255; nếu 0 valid trả về `0.0`) | Tỷ lệ pixel nhận đúng trong số pixel hợp lệ | ✔ |
| **Per-class IoU** | `IoU_c = TP_c / (TP_c + FP_c + FN_c)` | Chồng lấn giữa dự đoán và GT cho class `c` | ✔ |
| **Mean IoU (mIoU)** | `mean(IoU_c)` trên các class có `TP+FP+FN > 0` | Trung bình cộng IoU các class **có dữ liệu** | ✔ |
| **Per-class Dice** | `Dice_c = 2·TP_c / (2·TP_c + FP_c + FN_c)` | F1-like, tương đương `2·intersection/(pred+GT)` | ✔ |
| **Mean Dice** | `mean(Dice_c)` trên các class có `TP+FP+FN > 0` | Trung bình Dice | ✔ |

**Giải thích mIoU cẩn thận:** mIoU là **trung bình cộng (arithmetic mean) của IoU từng class**, KHÔNG phải "độ chính xác trung bình" pixel. Class nhỏ (rider) có thể kéo mIoU xuống dù Pixel Accuracy cao, vì mỗi class góp một phần **bằng nhau** vào mIoU bất kể diện tích.

**Quy ước class undefined** (`TP+FP+FN = 0`): IoU/Dice của class đó là `0.0` và **bị loại khỏi mean** — một class không xuất hiện không thể tự gán `IoU=1` để bơm mIoU. Nếu mọi class đều undefined, mean = `0.0`.

### 3.5 U-Net Results

Đọc từ `outputs/analysis/unet_cityscapes_evaluation.json` (500 ảnh Cityscapes val):

| Metric | Value |
|---|---|
| Pixel Accuracy | **0.9042** |
| Mean IoU | **0.4451** |
| Mean Dice | **0.5498** |
| (checkpoint) training_val_miou | 0.4262 |
| (checkpoint) training_val_pixel_accuracy | 0.9021 |

**Per-class IoU / Dice / Accuracy:**

| Class | IoU | Dice | Class Acc | Class | IoU | Dice | Class Acc |
|---|---|---|---|---|---|---|---|
| road | 0.9522 | 0.9755 | 0.9760 | person | 0.5280 | 0.6911 | 0.7730 |
| sidewalk | 0.6793 | 0.8090 | 0.7810 | rider | 0.0115 | 0.0227 | 0.0116 |
| building | 0.8237 | 0.9033 | 0.9331 | car | 0.8436 | 0.9152 | 0.9375 |
| wall | 0.1912 | 0.3211 | 0.2649 | truck | 0.0760 | 0.1412 | 0.0820 |
| fence | 0.1881 | 0.3166 | 0.2915 | bus | 0.1657 | 0.2842 | 0.1973 |
| pole | 0.3986 | 0.5700 | 0.4713 | train | 0.1307 | 0.2311 | 0.5600 |
| traffic light | 0.2183 | 0.3584 | 0.2235 | motorcycle | 0.0422 | 0.0811 | 0.0450 |
| traffic sign | 0.5006 | 0.6672 | 0.5685 | bicycle | 0.5304 | 0.6932 | 0.6327 |
| vegetation | 0.8630 | 0.9265 | 0.9183 | sky | 0.8842 | 0.9385 | 0.9742 |
| terrain | 0.4290 | 0.6004 | 0.6495 | | | | |

(Trình bày độ chính xác gốc giữ trong JSON; đây là bản dễ đọc tròn 4 chữ số.)

### 3.6 Per-Class Analysis

- **Mạnh:** road (0.952), sky (0.884), vegetation (0.863), car (0.844), building (0.824) — đây là các class **lớn, phổ biến, cấu trúc tương đối đồng nhất** trong cảnh đường phố.
- **Yếu:** rider (0.011), motorcycle (0.042), truck (0.076), wall (0.191), fence (0.188), traffic light (0.218) — các class **hiếm/nhỏ, hình dạng mảnh**, dễ bị nuốt bởi class lân cận.
- **Đáng chú ý:** train có IoU thấp (0.131) nhưng Class Accuracy khá cao (0.560) — tức nhiều pixel train nhận đúng nhưng vùng bao phủ dự đoán nhỏ hẹp so với GT (FP/FN cân bằng kém).
- *(Nhận định trên là **interpretation** chung về đặc điểm class lớn/nhỏ; repository không cung cấp phân tích nhân-quả chính thức nào thêm.)*

---

## 4. MIDAS EVALUATION

### 4.1 Evaluation Input

Từ `outputs/analysis/midas_kitti_evaluation.json`:

| Item | Value |
|---|---|
| Dataset | KITTI |
| Split | `val` |
| Number of samples | **1000** (num_expected 1000, skipped 0) |
| Checkpoint/model | `checkpoints/dpt_large_384.pt` — MiDaS DPT-Large pretrained |
| Depth cap | `depth_cap_m: 80` (pixel xa hơn 80 m bị gắn cờ invalid) |
| Invalid depth handling | Depth `0`/không hữu hạn bị loại khỏi metric; chỉ pixel `GT > 0` + finite + prediction finite mới tính |
| Input | `preprocessing.input_size: 384` (MiDaS transform resize 384) |

### 4.2 MiDaS Output Semantics

**MiDaS sinh ra RELATIVE INVERSE DEPTH, không phải depth metric tính bằng mét.**

- Giá trị **lớn hơn → gần camera hơn** (larger inverse-depth value = closer);
- Giá trị **nhỏ hơn → xa hơn** (smaller = farther);
- Không có thang mét (`is_metric=False`).
- Raw MiDaS prediction **không bao giờ được mô tả là mét**.

### 4.3 Evaluation Procedure

Thực hiện trong `evaluation/evaluate_midas.py` (`evaluate_dataset`):

1. **Load model:** build backend DPT-Large (workaround tên `DPT_Large` trên hub) + load weights cục bộ qua `_load_weights` (validate strict: missing/unexpected/mismatched → `MiDaSError`).
2. **Inference:** `MidDepthPredictor` (input 384, bilinear resize về resolution nguồn), `model.eval()`, `no_grad` — không có bất kỳ cập nhật tham số nào.
3. **Dataset:** `KittiDepthDataset` với `image_size=None` (giữ resolution native để so trực tiếp với GT depth map), `stem_key` chuẩn hóa tên `_sync_image_` / `_sync_groundtruth_depth_`.
4. **Valid-depth mask:** chỉ pixel `GT finite & GT > 0 & pred finite` (và `pred > 0` cho median scaling/threshold) mới được tính.
5. **Median scaling:** mỗi ảnh được **median-align độc lập** trước khi tính metric.
6. **Metric calculation:** `evaluate_depth(pred, gt, valid, align="median")` cho RMSE/MAE/AbsRel/δ1..δ3.
7. **Aggregation:** per-image metrics **trung bình cộng** trên toàn dataset; `mean_scale` = trung bình scale các ảnh.
8. **Output:** JSON tại `outputs/analysis/midas_kitti_evaluation.json`.

### 4.4 Median Scaling

**Phần bắt buộc — giải thích đơn giản:**

MiDaS dự đoán depth **tương đối** (inverse depth), không có thang mét mà KITTI GT yêu cầu. Để so sánh được, ta căn chỉnh prediction về thang GT **tại thời điểm đánh giá**:

```text
scale = median(GT_valid) / median(pred_valid)
scaled_prediction = prediction × scale
```

- Dùng **median** (không bị lệch bởi outlier) thay vì mean;
- Thực hiện **riêng cho từng ảnh**;
- Code thật: `depth_metrics.median_scale` — `aligned_prediction = prediction * scale`, trả bản sao **mới**, không sửa prediction/GT gốc.

**Quan trọng — KHÔNG được nói sai:**
- Median scaling **không thay đổi trọng số** của model MiDaS;
- **Không phải training**, không có backward/update nào;
- Đây chỉ là **evaluation-time alignment** (bước căn chỉnh ở lúc đánh giá) giúp đưa prediction về cùng thang đo mét với GT.

### 4.5 Depth Metrics

Các metric đúng code trong `evaluation/depth_metrics.py`:

| Metric | Formula (đúng code) | Meaning | Tốt khi |
|---|---|---|---|
| **RMSE** | `sqrt(mean((pred−gt)²))` | Sai số root-mean-square; phạt mạnh sai sót lớn | càng thấp |
| **MAE** | `mean(|pred−gt|)` | Sai số tuyệt đối trung bình | càng thấp |
| **AbsRel** | `mean(|pred−gt| / gt)` | Sai số tương đối chuẩn hóa theo GT | càng thấp |
| **δ1 / δ2 / δ3** | `mean(max(pred/gt, gt/pred) < t)` với `t = 1.25, 1.25², 1.25³` | Tỷ lệ pixel có tỉ số dự đoán/GT nằm trong ngưỡng t | càng cao |

Đơn vị RMSE/MAE là **mét** (sau alignment); AbsRel và δ là tỉ lệ (0..1).

### 4.6 MiDaS Results

Từ `outputs/analysis/midas_kitti_evaluation.json` (1000 ảnh KITTI val, align median):

| Metric | Value |
|---|---|
| RMSE | **4.2561 m** |
| MAE | **3.0182 m** |
| AbsRel | **0.8489** |
| δ1 | **0.1671** |
| δ2 | **0.3275** |
| δ3 | **0.4781** |
| Mean scale | **0.2217** |

### 4.7 MiDaS Result Interpretation

- AbsRel ≈ 0.85 và δ1 ≈ 0.17 là **kết quả yếu theo chuẩn depth evaluation** — DPT-Large pretrained không align sẵn với phân phối KITTI; phần lớn lỗi đến từ việc model dự đoán trên toàn bộ ảnh cảnh (crop_margin 0) trong khi các benchmark KITTI thường đánh giá trên vùng center-crop/valid mask gần xe.
- Các giá trị này phản ánh **quality của prediction sau alignment**, không phải "MiDaS tự sản sinh mét".
- MiDaS **không được train/cập nhật bằng KITTI** — toàn bộ điều chỉnh chỉ là median scale ở lúc đánh giá.
- *(Đây là interpretation dựa trên design của evaluation trong repo; repository không đưa ra benchmark khác để so sánh.)*

---

## 5. FULL PIPELINE EVALUATION

### 5.1 Pipeline Evaluation Goal

Chỉ có metric model là **chưa đủ**: phải kiểm tra cả hệ thống ghép lại có hành xử đúng không:

- thực thi trên **cùng một ảnh** (same-image contract);
- output được **căn chỉnh không gian** đúng (seg nearest, depth bilinear về resolution ảnh);
- **Fusion** chạy và cho thống kê per-class + region;
- **Scene Analysis** tạo report JSON hợp lệ;
- **Difficulty Analysis** tính score trong `[0,1]`;
- **visualization/output generation** sinh đủ file.

### 5.2 Evaluation Procedure

Thực hiện trong `evaluation/evaluate_pipeline.py` (`evaluate_single_image`):

```text
mở ảnh RGB (data/pipeline/images, ext .png/.jpg/.jpeg/.bmp/.tiff)
→ cùng image object đưa vào U-Net (predict, return_confidence=True) và MiDaS (predict)
→ segmentation = argmax class map; mean_confidence = mean(max softmax prob, finite)
→ depth = relative inverse depth
→ spatial alignment: segmentation resize NEAREST, depth resize BILINEAR về kích thước ảnh
→ fusion = fuse(segmentation, depth)
→ scene_report = analyze_fusion(fr, cfg)
→ difficulty = compute_difficulty(fr, scene_report, mean_confidence, weights, bins)
→ lưu JSON per-image + aggregate + tùy chọn visualization
```

U-Net inference resolution được **giới hạn** ở config nhỏ nhất (`[256,512]`) để không làm cạn VRAM khi cạnh tranh với MiDaS trên GPU 4 GB; wrapper vẫn resize predictions về resolution nguồn.

### 5.3 Same-Image Contract

**Cùng một object ảnh RGB được truyền vào cả hai model** (`seg_predictor.predict(np_image)` và `depth_predictor.predict(np_image)` với cùng biến `np_image`). Đây là yêu cầu trung tâm của project — không bao giờ trộn dataset; hai output sau đó nằm trong cùng hệ tọa độ không gian để Fusion ghép được.

### 5.4 Pipeline Results

Đọc từ `outputs/analysis/pipeline_evaluation.json`:

> **Đây là smoke test với 2 ảnh — KHÔNG phải benchmark có ý nghĩa thống kê.** Được nêu rõ như vậy; không trình bày 2 ảnh như một con số pháp lý cho chất lượng pipeline.

| Image | Difficulty | Level | Classes present | Mean confidence | Nearest dynamic |
|---|---|---|---|---|---|
| `pipeline_demo_001.png` | 0.541012 | **medium** | 19/19 | 0.6465 | car (near) |
| `pipeline_demo_002.png` | 0.508277 | **medium** | 18/19 | 0.6761 | car (near) |

**Aggregate:** `num_images = 2`, `average_difficulty_score = 0.5246`, `easy_count = 0`, `medium_count = 2`, `hard_count = 0`, `average_segmentation_confidence = 0.6613`, `average_depth_variation = 9.0421`.

Các chỉ số pipeline khác (demo_001): kích thước ảnh 352×1216, `analyzed_pixels = 428032`, `depth_convention = inverse_relative_larger_closer`, depth thresholds `[7.756, 16.862]`, `drivable_coverage_ratio = 0.1317`, `drivable_median_depth = 18.446`, `dynamic_object_count = 8`.

### 5.5 Difficulty Analysis

- **Score:** tổng có trọng số của 5 chỉ số (chuẩn hóa/clamp về [0,1]):
  `D = 0.25·U + 0.20·V + 0.20·C + 0.20·F + 0.15·O`
  (`U` = segmentation uncertainty `1 − mean_confidence`; `V` = depth variation coef. of variation; `C` = scene complexity = số class hiện diện/19; `F` = foreground fraction = near-region ratio; `O` = object density = ROI-class ratio).
- **Ngưỡng thật trong `configs/pipeline.yaml` (`difficulty.bins`):** `score ≤ 0.4` → **easy**; `score ≤ 0.7` → **medium**; còn lại → **hard**.
- **Quan trọng:** difficulty là **heuristic** (scene complexity **score**) — **không phải ground-truth difficulty** của ảnh.

---

## 6. RESULTS COMPARISON

### 6.1 Model-Level Summary

| Model | Dataset | Task | Main Metrics |
|---|---|---|---|
| **U-Net** | Cityscapes | Semantic Segmentation | Pixel Acc **0.9042** · mIoU **0.4451** · Mean Dice **0.5498** (500 val ảnh) |
| **MiDaS** | KITTI | Monocular Depth | AbsRel **0.8489** · RMSE **4.2561** · δ1 **0.1671** (1000 val ảnh, median-aligned) |

### 6.2 Pipeline-Level Summary

- Số ảnh test: **2** (smoke);
- Phân phối difficulty: **2 medium** (không easy, không hard);
- Scene-analysis outputs: per-class depth stats, depth distribution ~1/3 mỗi vùng near/middle/far, traffic context (vehicles/pedestrians/road, nearest dynamic object);
- Nearest dynamic object (cả 2 ảnh): **car** ở vùng **near** (relative depth);
- Khác: mean confidence 0.65–0.68, toàn bộ 19 class hiện diện ở ít nhất 1 ảnh (motorcycle chỉ ở demo_001).

### 6.3 What the Results Mean

- **U-Net làm tốt:** class lớn/cấu trúc (road/sky/vegetation/car/building) → pipeline có nền tảng phân đoạn đối tượng chính chắc chắn.
- **U-Net kém:** class nhỏ/hiếm (rider/motorcycle/truck/wall/fence) → mIoU tổng bị kéo xuống; đây là hạn chế đã biết của training ít epoch/class mất cân bằng.
- **MiDaS:** dù DPT-Large pretrained, số liệu sau median-alignment trên full-ảnh KITTI còn yếu — chỉ nên hiểu là chất lượng prediction trên setup hiện tại, không phải khẳng định về model preprint.
- **Pipeline:** smoke test chứng minh **hệ thống chạy đúng** (same-image, fusion, scene, difficulty hoạt động, ra JSON/hình). **Không thể** kết luận chất lượng pipeline từ 2 ảnh.
- **Không kết luận được:** benchmark pipeline với N lớn, so sánh công bằng với phương pháp khác, hay bất kỳ claim độ chính xác tuyệt đối của pipeline.

---

## 7. VISUAL EVALUATION

Dựa trên các module `visualization/*` và các file đã có trong `outputs/` (đã verify tồn tại — **không bịa hình**):

### 7.1 Segmentation Visualization

`visualization/segmentation.py` ánh xạ trainId `0..18` sang **palette cố định 19 màu** (`CITYSCAPES_TRAINID_COLORS`, màu void = đen), vẽ segmentation màu hoặc overlay lên ảnh gốc. File có thật: `outputs/segmentation/pipeline_demo_001_segmentation.png`, `_002_…`.

### 7.2 Depth Visualization

`visualization/depth.py` chỉ “đọc” depth, chuẩn hóa min-max trên **bản sao** cho hiển thị (raw prediction không bao giờ bị sửa), dùng colormap **turbo**, nhãn màu: *"Relative inverse depth (larger = closer)"* — gần = giá trị màu cao. File có thật: `outputs/depth/pipeline_demo_001_depth.png`.

### 7.3 Fusion Visualization

`visualization/fusion.py` pha segmentation overlay + depth colormap lên ảnh RGB với alpha từ `configs/pipeline.yaml` (`alpha_segmentation: 0.5`, `alpha_depth: 0.35`) → `outputs/analysis/pipeline_demo_001_fusion.png`.

### 7.4 Scene Overview

`visualization/scene.py` dựng **overview figure tổng hợp** từ scene report (`create_full_visualization`: ảnh gốc, segmentation, depth, region map, bảng tóm tắt) + `save_scene_report` ghi JSON → `outputs/analysis/pipeline_demo_001_overview.png` và `pipeline_demo_001_scene_report.json`. Ngoài ra `outputs/custom/` còn có demo nhiều ảnh (pipeline_demo_001/002, streetest, gta5) với đầy đủ original/segmentation/depth/fusion/overview + scene_report.

---

## 8. EVALUATION LIMITATIONS

**Factual limitations (có thật trong repo):**

- Cityscapes và KITTI là **hai dataset riêng biệt** — không có cặp GT chéo.
- **Pipeline không có** paired ground-truth benchmark: không có GT depth/segmentation cho ảnh pipeline để chấm tự động.
- **Pipeline smoke test hiện tại chỉ 2 ảnh** (`pipeline_demo_001/002`), ít hơn nhiều so với yêu cầu thống kê.
- **MiDaS output là relative inverse depth** — mọi con số "mét" chỉ có sau evaluation-time alignment.
- **Median scaling là evaluation-time alignment** — không phải training, không đổi weight.
- **U-Net performance không đều giữa các class** — mIoU bị class nhỏ/hiếm kéo thấp.
- Giới hạn tài nguyên GPU (~4 GB) buộc U-Net inference chạy `[256,512]` trong pipeline (ghi rõ trong `evaluation/evaluate_pipeline.py`).

**Interpretation (giải thích của người viết doc, không phải repo):** các số U-Net/MiDaS nên được đọc như benchmark riêng của từng task, không tổng hợp thành một con số chất lượng pipeline; muốn kết luận pipeline cần đánh giá trên nhiều ảnh hơn.

---

## 9. AI-ASSISTED DEVELOPMENT

### 9.1 Prompt

Các bước đánh giá được thực hiện theo task brief trong `prompts/`:

- `prompts/06_segmentation_evaluation.md` — Step 06: **segmentation metrics** (chỉ là metric, không MiDaS/fusion/training);
- `prompts/08_depth_evaluation.md` — depth metrics/alignment;
- `prompts/13_evaluate_unet.md`, `prompts/14_evaluate_midas.md`, `prompts/15_pipeline_evaluation.md` — các lần chạy đánh giá sinh ra 3 file JSON.

*(Prompt Step 06 quy định rõ: focus ONLY segmentation metrics; không MiDaS, không depth, không train.)*

### 9.2 Generated/Modified Evaluation Scripts

| File | Trách nhiệm |
|---|---|
| `evaluation/segmentation_metrics.py` | Pixel Acc / IoU / mIoU / Dice từ global confusion matrix |
| `evaluation/depth_metrics.py` | RMSE / MAE / AbsRel / δ1..3 + `median_scale` |
| `evaluation/evaluate_unet.py` | Chạy U-Net checkpoint trên Cityscapes val |
| `evaluation/evaluate_midas.py` | Chạy MiDaS DPT-Large trên KITTI val |
| `evaluation/evaluate_pipeline.py` | Full pipeline trên ảnh standalone |
| `evaluation/difficulty_analysis.py` | Difficulty score heuristic |
| `evaluation/pipeline_metrics.py` | Helpers aggregate pipeline JSON |

### 9.3 Testing

Test liên quan đánh giá (synthetic, offline, không dataset/GPU):

```text
.venv/bin/python -m pytest \
    tests/test_segmentation_metrics.py tests/test_depth_metrics.py \
    tests/test_evaluate_unet.py tests/test_evaluate_midas.py \
    tests/test_evaluate_pipeline.py -q
133 passed in 2.37s        # (đo tại thời điểm viết doc)
```

Toàn bộ regression suite hiện tại (đã xác minh): **456 passed, 1 skipped** (skip CUDA).

### 9.4 Validation

- Các test metric sử dụng **ví dụ tính tay** (perfect prediction, ignore index `255`, class bị thiếu không bơm IoU, shape mismatch báo lỗi) — như prompt Step 06 yêu cầu;
- Kết quả JSON được sinh bởi chính các script đánh giá (metadata checkpoint + con số) — đối chiếu chéo giữa JSON/script/config khi viết doc.

---

## 10. DISCUSSION

### 10.1 Strengths

- **Hai cấp đánh giá tách bạch:** model metric dùng GT riêng từng dataset; pipeline đánh giá hành vi — không lẫn lộn.
- **Đúng chuẩn kỹ thuật:** ignore pixel `255` bị loại; class undefined không tự thành 1; global confusion matrix (không cần giữ tensor lớn); median scaling evaluation-time rõ ràng.
- **Tái sử dụng metric:** trainer (Step 12) gọi `evaluate_segmentation`, eval scripts gọi `aggregate_from_confusion_matrix` — không duplicate công thức.
- **Same-image contract + alignment policy** được enforce trong pipeline (nearest class ID / bilinear depth).
- **JSON có metadata đầy đủ** (checkpoint, split, num_samples, conventions) giúp tái lập và trình bày.

### 10.2 Weaknesses

- **Số liệu MiDaS thấp** (AbsRel 0.85, δ1 0.17) — chưa align tốt với KITTI full-ảnh; đánh giá hiện tại chưa phải chuẩn benchmark tối ưu.
- **Pipeline smoke 2 ảnh** — chưa đủ để nói gì về chất lượng thống kê.
- **Không có so sánh** với baseline/state-of-the-art nào được lưu trong repo.

### 10.3 Interpretation

U-Net (mIoU 0.445/val) phân đoạn tốt các class chủ đạo, kém class nhỏ — đủ để pipeline hoạt động như một demo; MiDaS cho depth tương đối với lỗi lớn sau alignment, phù hợp để minh họa khái niệm hơn là đo đạc chính xác; pipeline chứng minh được tính khả thi về mặt build (end-to-end chạy), chưa xác nhận chất lượng định lượng. **Không overclaim.**

---

## 11. CONCLUSION

Tóm tắt dễ trình bày miệng:

- **U-Net được đánh giá độc lập trên Cityscapes** (500 ảnh val): Pixel Acc 0.9042, mIoU 0.4451 — class lớn tốt, class nhỏ/hiếm yếu.
- **MiDaS được đánh giá độc lập trên KITTI** (1000 ảnh val): AbsRel 0.8489, RMSE 4.2561 (sau median alignment) — MiDaS output là **relative inverse depth**, không phải mét; median scaling chỉ là **evaluation-time alignment**, không phải training.
- **Cityscapes và KITTI không bao giờ được ghép cặp** — mỗi model đánh giá với GT riêng của dataset mình.
- **Full pipeline đánh giá bằng cùng một ảnh RGB** cho cả hai model (same-image contract) — không cần GT ghép chéo.
- **Fusion kết hợp thông tin semantic + relative depth** (rule-based, per-class stats + vùng near/middle/far).
- **Model-level metrics và pipeline-level behavior trả lời hai câu hỏi khác nhau:** *"model học tốt không?"* vs *"hệ thống ghép lại có chạy đúng và hiểu cảnh gì?"* — không lẫn lộn hai cấp này.

---

## 12. REFERENCES

- `prompts/06_segmentation_evaluation.md` — task brief Step 06 (segmentation metrics).
- `prompts/08_depth_evaluation.md`, `prompts/13_evaluate_unet.md`, `prompts/14_evaluate_midas.md`, `prompts/15_pipeline_evaluation.md` — các bước đánh giá.
- `evaluation/segmentation_metrics.py`, `evaluation/depth_metrics.py` — metric implementation.
- `evaluation/evaluate_unet.py`, `evaluation/evaluate_midas.py`, `evaluation/evaluate_pipeline.py`, `evaluation/difficulty_analysis.py`, `evaluation/pipeline_metrics.py` — evaluation scripts.
- `outputs/analysis/unet_cityscapes_evaluation.json`, `outputs/analysis/midas_kitti_evaluation.json`, `outputs/analysis/pipeline_evaluation.json` — kết quả (nguồn sự thật cho mọi con số).
- `configs/unet.yaml`, `configs/midas.yaml`, `configs/pipeline.yaml` — cấu hình (image_size, input_size 384, depth_cap_m 80, weights/bins difficulty).
- `visualization/{segmentation,depth,fusion,scene,io}.py` — visual evaluation.
- `docs/coursework/00..05` — các doc trước (thống nhất thuật ngữ, kiến trúc, build flow).