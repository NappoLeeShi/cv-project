# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

> Báo cáo coursework môn Thị giác máy tính — dự án **CV-PROJECT**.
> Tất cả số liệu bên dưới được trích trực tiếp từ các file JSON kết quả và cấu hình (config) của dự án, không phải số liệu ước lượng.

---

# 1. Introduction

## 1.1 Background

**Scene Understanding** là lĩnh vực của computer vision nhằm làm cho máy tính "hiểu" một cảnh (scene) trong ảnh, không chỉ nhận diện *có đối tượng nào* mà còn *mỗi pixel thuộc lớp gì* và *các đối tượng nằm ở đâu*. Hai kỹ thuật nền tảng được dùng trong dự án:

- **Semantic Segmentation** là quá trình gán một class (nhãn) cho từng pixel trong ảnh. Ví dụ pixel thuộc `road`, `car`, `person`, `sky`, ... Kết quả là một "mask" cùng kích thước với ảnh gốc.
- **Monocular Depth Estimation** là bài toán ước lượng độ sâu của mỗi pixel từ **một** ảnh RGB duy nhất (không dùng stereo camera hay sensor). MiDaS trả về **relative inverse depth** (độ sâu tỉ đối nghịch đảo), tức giá trị lớn hơn nghĩa là **gần camera hơn**, và giá trị này **không** phải khoảng cách tính bằng mét.

## 1.2 Motivation

Trong ảnh đường phố (driving scenes), ngữ nghĩa (semantic) và độ sâu (depth) bổ sung cho nhau:

- Chỉ có segmentation thì biết *"có một chiếc car"* nhưng không biết car đó gần hay xa.
- Chỉ có depth thì biết *"vùng này gần"* nhưng không biết vùng đó là `road`, `car` hay `pedestrian`.
- Kết hợp cả hai giúp xây dựng một **scene understanding report** hữu ích: class nào nằm ở vùng gần, có bao nhiêu đối tượng động (dynamic objects) trong cảnh, vùng drivable (lòng đường) bao phủ tỉ lệ nào, đối tượng gần nhất là gì...

Đây là nền tảng cho các hệ thống hỗ trợ lái (driver assistance) hoặc xe tự hành, nhưng dự án chỉ dừng ở mức phân tích cảnh, **không** điều khiển phương tiện.

---

# 2. Problem Definition

Bài toán nhận **một ảnh RGB đường phố duy nhất** làm input, thực hiện đồng thời semantic segmentation và monocular depth estimation trên **chính ảnh đó**, sau đó fuse (kết hợp) hai đầu ra để phân tích cảnh và mức độ khó (difficulty) của cảnh.

## 2.1 Input

Một ảnh RGB duy nhất (PIL image hoặc numpy array `[H, W, 3]`), ví dụ ảnh đường phố từ `data/pipeline/images/`. Yêu cầu quan trọng: **cùng một ảnh** được đưa vào cả hai mô hình (gọi là *same-image contract*).

## 2.2 Output

- **Semantic Segmentation**: mask class `[H, W]` với các trainId 0..18 (Cityscapes, 19 lớp), 255 = ignore/void.
- **Relative Inverse Depth**: bản đồ depth `[H, W]` (relative inverse depth, đơn vị tương đối — **không phải mét**).
- **Fusion** (`FusionResult`): thống kê per-class (pixel count, mean/median inverse depth), vùng near/middle/far.
- **Scene Analysis**: báo cáo JSON gồm semantic distribution, depth distribution, traffic context (vehicles, pedestrians, nearest dynamic class), interpretation.
- **Difficulty Analysis**: điểm `difficulty_score` trong `[0,1]` kèm mức **Easy / Medium / Hard** (đây là heuristic, **không** phải ground-truth difficulty).

---

# 3. Objectives

1. **Tái sử dụng các module đã có** của dự án (U-Net, MiDaS, fusion, analyzer, visualization, evaluation) mà không thiết kế lại kiến trúc.
2. **Xây dựng pipeline đánh giá đầy đủ** (full-pipeline evaluation): ảnh RGB → segmentation → depth → fusion → scene analysis → difficulty analysis trên **cùng một ảnh**.
3. **Đánh giá mô hình ở mức model-level**: U-Net trên Cityscapes val (mIoU, Pixel Accuracy, Mean Dice, per-class IoU) và MiDaS trên KITTI val (RMSE, MAE, AbsRel, δ1, δ2, δ3) — tách biệt, không trộn hai dataset.
4. **Cung cấp công cụ CLI tái lập** (reproducible CLI) và các bài test offline chạy bằng mock/synthetic.
5. **Tạo báo cáo đầy đủ** (chính là file này) dựa trên số liệu thực từ JSON output.

---

# 4. Scope and Out of Scope

## 4.1 In Scope

- **Semantic Segmentation** — U-Net checkpoint đã huấn luyện trên Cityscapes.
- **Monocular Depth Estimation** — MiDaS DPT-Large pretrained.
- **Feature Fusion** — fusion phân tích (rule-based) giữa segmentation và relative depth.
- **Scene Understanding** — báo cáo cảnh từ `scene_understanding/analyzer.py`.
- **Difficulty Analysis** — điểm khó heuristic, cấu hình được.
- **Model Evaluation** — đánh giá model-level (U-Net/Cityscapes, MiDaS/KITTI) và pipeline-level.
- **Visualization** — xuất ảnh segmentation, depth, fusion, overview.
- **Reproducible CLI Pipeline** — `python -m evaluation.evaluate_pipeline`, `main.py`, v.v.

## 4.2 Out of Scope

- **MiDaS training/fine-tuning** — MiDaS là pretrained, không huấn luyện.
- **Object Detection and Tracking** — không nhận diện/dự phóng bounding box, không theo dõi đối tượng theo thời gian.
- **3D Reconstruction** — không tái tạo scene 3D.
- **Autonomous-driving control** — không điều khiển phương tiện.
- **Metric-depth prediction from raw MiDaS output** — không quy đổi GDP raw MiDaS thành mét.
- **Paired Cityscapes–KITTI benchmarking** — hai dataset không được ghép cặp.
- **Ground-truth difficulty labeling** — difficulty là heuristic, không phải nhãn ground-truth.

---

# 5. Dataset

## 5.1 Cityscapes

**Cityscapes** là dataset urban scene dùng cho semantic segmentation và scene understanding. Dataset cung cấp ảnh RGB đường phố đô thị cùng **ground-truth labels** gán nhãn từng pixel (sau đó được remap về trainId space 0..18, 19 lớp; 255 = ignore).

Trong dự án:

- Cấu trúc: `data/cityscapes/` gồm `images/` và `labels/` (xem `preprocessing/cityscapes.py`).
- U-Net được **huấn luyện** trên Cityscapes và được **đánh giá** trên split **val**. Theo file `outputs/analysis/unet_cityscapes_evaluation.json`, số mẫu đánh giá là **500** — tương ứng toàn bộ validation split của Cityscapes.
- Kích thước ảnh sử dụng: `[256, 512]` (`data.image_size` trong `configs/unet.yaml`).

## 5.2 KITTI

**KITTI** là bộ benchmark nổi tiếng cho autonomous driving, gồm ảnh stereo màu, LiDAR, GPS, v.v. Dự án sử dụng **depth ground truth** từ KITTI raw data.

Trong dự án:

- Cấu trúc: `data/kitti/` gồm `images/` và `depth/` (depth là uint16, đơn vị millimetre, quy đổi sang mét bằng `value / 1000`; xem `preprocessing/kitti.py`).
- Depth 0 nghĩa là invalid; `depth_cap_m=80` đánh dấu pixel xa hơn 80 m là invalid.
- Số mẫu: **1000 ảnh RGB** và **1000 depth GT** (`num_samples=1000`, `num_expected=1000`, `skipped=0` trong `midas_kitti_evaluation.json`), đánh giá trên split **val**.

## 5.3 Dataset Relationship

**Cityscapes và KITTI KHÔNG phải là hai dataset được ghép cặp (paired).** Chúng có camera, hình học chụp và hệ nhãn khác nhau. Luật bất biến của dự án:

- **Không bao giờ** ghép ảnh Cityscapes với depth map KITTI.
- **Model-level evaluation** được thực hiện tách biệt:
  - U-Net + Cityscapes GT → mIoU, Pixel Accuracy, Dice.
  - MiDaS + KITTI GT → RMSE, MAE, AbsRel, δ1, δ2, δ3.
- **Full pipeline** dùng **một ảnh RGB duy nhất** làm đầu vào cho cả hai mô hình (same-image contract), không cần ground-truth.

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

Quá trình: một ảnh RGB → U-Net (segmentation mask) và MiDaS (relative inverse depth) → căn chỉnh không gian (alignment) về đúng resolution của ảnh → fusion → scene analysis → difficulty analysis.

## 6.2 U-Net

**U-Net** (`models/unet/model.py`) là mạng **encoder–bottleneck–decoder** truyền thống:

- **Encoder** (contracting path) trích xuất đặc trưng, giảm dần kích thước không gian.
- **Bottleneck** giữ đặc trưng ngữ nghĩa mức cao nhất.
- **Decoder** (expanding path) tăng dần kích thước về resolution gốc.
- **Skip connections** nối đặc trưng cùng cấp giữa encoder và decoder, giúp giữ lại chi tiết không gian (quan trọng cho segmentation).
- Đầu ra là **logits** `[C, H, W]`; `argmax(dim=1)` cho mask class. Cấu hình: `num_classes=19`, `in_channels=3`, `base_channels=64`.
- Ảnh được resize về kích thước mô hình bằng bilinear; mask được resize ngược về ảnh nguồn bằng **nearest-neighbour** (không dùng bilinear cho class ID).

**Training**: dùng Adam optimizer, loss **weighted cross-entropy** (cân bằng class), có augmentation (hflip, random scale-crop, color jitter). Chi tiết siêu tham số thực tế ở mục 8.1. (Chi tiết công thức mất mát nằm trong `training/`; trong khuôn khổ báo cáo này không đi sâu vào toàn bộ code.)

## 6.3 MiDaS DPT-Large

**MiDaS DPT-Large** (`models/midas/model.py`, `inference.py`) là mô hình monocular depth estimation pretrained, dùng **Vision Transformer (DPT)** làm backbone.

Điều quan trọng về ngữ nghĩa đầu ra:

> **Raw MiDaS output là relative inverse depth**, không phải depth metric (mét). Giá trị lớn hơn = vùng gần camera hơn. MiDaS không bao giờ được mô tả là đầu ra depth theo mét. Median scaling chỉ được áp dụng ở bước đánh giá (evaluation-time alignment), không làm thay đổi bản chất depth tương đối của bản đồ demo.

Trong dự án, MiDaS được load từ checkpoint `checkpoints/dpt_large_384.pt`, input size `384`, chạy trong `no_grad`/eval mode, và đầu ra được resize bilinear về resolution nguồn.

## 6.4 Feature Fusion

Fusion là **analytical (rule-based)**, không phải mạng nơ-ron fusion. Logic nằm trong `scene_understanding/fusion.py`:

- Yêu cầu segmentation và depth **cùng kích thước không gian** (shape match) — đây là ràng buộc same-image.
- Pixels depth không hữu hạn (NaN/inf) và pixel void (`255`) bị loại khỏi mọi thống kê (thông qua `analyzed_mask`).
- **Per-class stats**: với mỗi class xuất hiện, tính `pixel_count`, `pixel_ratio`, `mean_depth`, `median_depth`, `min_depth`, `max_depth`.
- **Depth regions**: ngưỡng được dẫn xuất từ chính depth map (default terciles `[0.3333, 0.6667]`): `near` = 1/3 giá trị lớn nhất, `middle`, `far` = 1/3 nhỏ nhất. Quy ước: **larger inverse depth = closer**.

Kết quả là `FusionResult` chứa `per_class`, `region_map`, `region_summary`, `thresholds`.

## 6.5 Scene Analysis

`scene_understanding/analyzer.py` chuyển `FusionResult` thành báo cáo JSON:

- `scene` — kích thước, số lớp ngữ nghĩa, depth convention.
- `semantic_distribution` — pixel count từng class.
- `depth_distribution` — tỉ lệ near/middle/far.
- `regions` — chi tiết per-class depth.
- `traffic_context` — vehicles, pedestrians, road, `drivable_coverage_ratio`, `drivable_median_depth`, `dynamic_object_count`, `nearest_dynamic_class`.
- `interpretation` — các dòng mô tả bằng tiếng Anh, ví dụ *"The nearest dynamic class is 'car' (near relative-depth region)."*

## 6.6 Difficulty Analysis

`evaluation/difficulty_analysis.py` tính **pipeline difficulty score** (heuristic). Công thức đúng như cài đặt:

```
D = 0.25·U + 0.20·V + 0.20·C + 0.20·F + 0.15·O
```

trong đó:

- `U` = **segmentation uncertainty** = `1 - mean_confidence` (mean confidence là max softmax probability trung bình của U-Net trên ảnh).
- `V` = **depth variation** = `std / mean` của finite inverse depth (coefficient of variation).
- `C` = **scene complexity** = số class xuất hiện / 19 (tổng class Cityscapes trainId).
- `F` = **foreground fraction** = tỉ lệ pixel thuộc vùng *near* (depth region).
- `O` = **object density** = tỉ lệ pixel thuộc các class ROI (dynamic objects, person..bicycle, trainId 11..18).

Điểm số được kẹp (clamp) về `[0, 1]` và phân loại theo ngưỡng thực tế trong `configs/pipeline.yaml`:

| Ngưỡng | Mức |
|---|---|
| `score <= 0.4` | **Easy** |
| `score <= 0.7` | **Medium** |
| else | **Hard** |

Đây là **điểm khó heuristic của pipeline** (pipeline difficulty score / scene complexity score), **không** phải nhãn difficulty ground-truth.

---

# 7. Implementation

## 7.1 Project Structure

Tóm tắt cấu trúc dự án (chỉ nêu các phần quan trọng; không chép toàn bộ code):

```
cv-project/
├── configs/            # unet.yaml, midas.yaml, pipeline.yaml
├── data/
│   ├── cityscapes/     # images/, labels/
│   ├── kitti/          # images/, depth/
│   └── pipeline/images/# ảnh demo (2 ảnh) cho full pipeline
├── models/
│   ├── unet/           # model.py, inference.py
│   └── midas/          # model.py, inference.py
├── preprocessing/      # cityscapes.py, kitti.py, transforms.py
├── scene_understanding/# fusion.py, analyzer.py, pipeline.py
├── visualization/      # segmentation.py, depth.py, fusion.py, scene.py, io.py
├── evaluation/         # evaluate_unet.py, evaluate_midas.py,
│                       # evaluate_pipeline.py, difficulty_analysis.py,
│                       # segmentation_metrics.py, depth_metrics.py, pipeline_metrics.py
├── utils/              # config.py, device.py, seed.py, logger.py
├── training/           # logic huấn luyện U-Net
├── tests/              # bộ test pytest (offline, synthetic)
├── checkpoints/        # unet_cityscapes.pth, dpt_large_384.pt
├── outputs/
│   ├── analysis/       # *evaluation.json, pipeline_evaluation.json
│   ├── segmentation/   # ảnh segmentation demo
│   ├── depth/          # ảnh depth demo
│   └── visualization/  # smoke test
└── main.py             # demo CLI cho 1 ảnh
```

## 7.2 Data Preprocessing

- **Shared transform** `preprocessing/transforms.py`: `ImageTransform` (resize, chuẩn hóa ImageNet mean/std: `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`).
- Cityscapes: label raw (labelIds 0..33) được remap sang **trainId 0..18, 255 = void** qua bảng tra cứu; label resize bằng nearest-neighbour.
- KITTI: depth uint16 (mm) → mét (`/1000`); `0` = invalid; `depth_cap_m=80` giới hạn.
- Quan trọng: dataset loader **ghép cặp theo stem của tên file**, không theo thứ tự sắp xếp, nên không thể xảy ra lệch cặp.

## 7.3 Model Training and Inference

- **U-Net**: huấn luyện trên Cityscapes train (epochs, batch size, v.v. ở mục 8.1). Inference qua `UNetInference.predict` (eval mode, `no_grad`), tùy chọn trả về `(prediction, confidence)`.
- **MiDaS**: load `checkpoints/dpt_large_384.pt` vào backend DPT-Large (dùng cùng workaround hub `DPT_Large` như `evaluate_midas.py`), inference qua `MidDepthPredictor.predict` (eval mode, `no_grad`), đầu ra relative inverse depth.

## 7.4 Evaluation

- **Model-level**:
  - `evaluation/evaluate_unet.py` — Cityscapes val, confusion matrix global (ignore 255), tính Pixel Accuracy / mIoU / Mean Dice / per-class IoU.
  - `evaluation/evaluate_midas.py` — KITTI val, per-image **median scaling** `scale = median(gt_valid) / median(pred_valid)`, sau đó tính RMSE/MAE/AbsRel/δ1/δ2/δ3 và lấy trung bình.
- **Pipeline-level**:
  - `evaluation/evaluate_pipeline.py` — đưa cùng một ảnh vào U-Net và MiDaS, align khích thước, fuse, analyze scene, tính difficulty, xuất JSON + visualization.

## 7.5 Visualization

Các module trong `visualization/`:

- `segmentation.py` — `colorize_segmentation` (palette cố định 19 màu Cityscapes), overlay.
- `depth.py` — `colorize_depth` (colormap `turbo`), nhãn luôn ghi rõ *"Relative inverse depth (larger = closer)"*, không ghi mét.
- `fusion.py` — overlay segmentation + depth, và region map near/middle/far.
- `scene.py` — `create_full_visualization` (figure tổng hợp), `save_scene_report` (JSON).

## 7.6 CLI and Reproducibility

```bash
# Chạy toàn bộ test (offline, không cần model/dataset)
.venv/bin/python -m pytest -q

# Pipeline smoke test trên ảnh thật (2 ảnh, CUDA nếu có)
.venv/bin/python -m evaluation.evaluate_pipeline \
    --input-dir data/pipeline/images \
    --limit 2 --save-visualizations --device auto

# Đánh giá model-level U-Net (Cityscapes val)
.venv/bin/python -m evaluation.evaluate_unet --checkpoint checkpoints/unet_cityscapes.pth

# Đánh giá model-level MiDaS (KITTI val)
.venv/bin/python -m evaluation.evaluate_midas --checkpoint checkpoints/dpt_large_384.pt
```

Seed cố định (`system.seed: 42`) giúp kết quả tái lập được.

---

# 8. Experimental Setup

## 8.1 U-Net Training

Siêu tham số thực tế từ `configs/unet.yaml` (nhóm `training`) và `outputs/analysis/unet_training_history.json`:

| Tham số | Giá trị |
|---|---|
| Architecture | U-Net (encoder–decoder, skip connections) |
| `num_classes` | 19 |
| `in_channels` / `base_channels` | 3 / 64 |
| Image size (train) | `[256, 512]` |
| Batch size | 1 |
| Learning rate | `0.0001` |
| Optimizer | Adam (`adam`) |
| Weight decay | `0.00001` |
| Epochs | 20 |
| Mixed precision | `false` |
| Seed | 42 |
| Loss | weighted cross-entropy |

**Training history** (từ JSON):

- `best_epoch` = **20**, `best_val_loss` = **0.31346111369878055**.
- `best_val_miou` = **0.4262424504443624**.
- `best_epoch` trùng với giá trị ghi trong checkpoint (`checkpoint_epoch: 20`) — checkpoint cuối/val-mIoU tốt nhất là epoch 20.
- Pixel accuracy cuối epoch huấn luyện: `0.9021092620377307`.

## 8.2 U-Net Evaluation

- Split: **Cityscapes val**.
- Số mẫu: **500** (full val split).
- `image_size`: `[256, 512]`, `ignore_index`: `255`, `num_classes`: `19`.
- Thủ tục: load checkpoint, chạy eval/no_grad, gộp **một** confusion matrix toàn cục rồi tính metrics (xem `evaluation/evaluate_unet.py`).

## 8.3 MiDaS Evaluation

- Split: **KITTI val**; số mẫu: **1000** (`num_samples=1000`, `num_expected=1000`, `skipped=0`).
- **Median scaling** (evaluation-time only): mỗi ảnh được align độc lập với `scale = median(gt_valid)/median(pred_valid)` trước khi tính lỗi; `mean_scale = 0.22169487541812957`. Đây chỉ là alignment để so sánh, **không** biến MiDaS thành depth mét.
- `depth_cap_m = 80`, depth 0 = invalid.
- MiDaS output vẫn là **relative inverse depth** (`relative_inverse_depth: true`, `larger_value_is_closer: true`).

## 8.4 Full Pipeline Smoke Test

- Nguồn ảnh: thư mục `data/pipeline/images/` chứa **2 ảnh RGB đường phố** (pipeline_demo_001.png, pipeline_demo_002.png).
- Lệnh: `--limit 2 --save-visualizations --device auto`.
- Kích thước ảnh: `352×1216` (2 ảnh; `analyzed_pixels = 428032`).
- **Quan trọng**: đây chỉ là **smoke test** gồm 2 ảnh để kiểm tra tính khả thi của pipeline (checkpoint load, CUDA, same-image, fusion, analyzer, difficulty, visualization, JSON). **KHÔNG** được coi là benchmark có ý nghĩa thống kê.

---

# 9. Results

## 9.1 U-Net Results

Từ `outputs/analysis/unet_cityscapes_evaluation.json` (Cityscapes **val**, 500 ảnh, checkpoint epoch 20):

| Metric | Result |
|---|---|
| Pixel Accuracy | 0.9042111912537422 |
| mIoU | 0.445074137144175 |
| Mean Dice | 0.5498125290480808 |
| (val mIoU ghi trong checkpoint epoch 20) | 0.4262424504443624 |

## 9.2 MiDaS Results

Từ `outputs/analysis/midas_kitti_evaluation.json` (KITTI **val**, 1000 ảnh, median scaling):

| Metric | Result |
|---|---|
| RMSE | 4.256132507952642 |
| MAE | 3.018216943917585 |
| AbsRel | 0.8489374433912796 |
| δ1 (delta1) | 0.16709821565801006 |
| δ2 (delta2) | 0.3275189399048214 |
| δ3 (delta3) | 0.4781034299876558 |
| mean scale (median) | 0.22169487541812957 |

Ghi chú: các giá trị δ thấp, RMSE/MAE/AbsRel tương đối cao là dấu hiệu cho thấy relative inverse depth của MiDaS khi đưa về thang depth tuyến tính của KITTI có sai lệch lớn — chi tiết ở Discussion.

## 9.3 Per-Class Segmentation Results

IoU từng lớp (Cityscapes val) từ `unet_cityscapes_evaluation.json`:

| Class | IoU | | Class | IoU |
|---|---|---|---|---|
| road | 0.9522424488781998 | | person | 0.5279796058947245 |
| sidewalk | 0.6792857854814197 | | rider | 0.011459488791885788 |
| building | 0.8237347214169852 | | car | 0.8436484341008157 |
| wall | 0.19122455834870059 | | truck | 0.07598792344364184 |
| fence | 0.18809036593962372 | | bus | 0.16566493200137167 |
| pole | 0.39857740293790717 | | train | 0.13067776671794407 |
| traffic light | 0.21834830684174153 | | motorcycle | 0.04224830903969902 |
| traffic sign | 0.500585585377887 | | bicycle | 0.530411461903086 |
| vegetation | 0.8630465977246684 | | | |
| terrain | 0.42901382203064126 | | | |
| sky | 0.8841810888683829 | | | |

Nhận xét ngắn: các lớp chiếm diện tích lớn như `road`, `building`, `vegetation`, `sky`, `car` đạt IoU cao; các lớp hiếm/nhỏ như `rider`, `motorcycle`, `truck`, `wall`, `fence` có IoU rất thấp.

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

Component difficulty của từng ảnh (đúng theo công thức mục 6.6):

| Image | U (uncertainty) | V (depth var) | C (complexity) | F (foreground) | O (object density) |
|---|---|---|---|---|---|
| pipeline_demo_001.png | 0.353465 | 0.695628 | 1.0 | 0.333335 | 0.312353 |
| pipeline_demo_002.png | 0.323932 | 0.642976 | 0.947368 | 0.333335 | 0.283717 |

Đối tượng động gần nhất (nearest dynamic class) cho cả 2 ảnh đều là **`car`** ở vùng **near**. `dynamic_object_count`: 8 (ảnh 1), 7 (ảnh 2).

> ⚠️ **Lưu ý danh nghĩa:** đây là smoke test với 2 ảnh; không diễn giải như benchmark thống kê. Phân bố Easy/Medium/Hard thu được chỉ mang tính minh họa.

---

# 10. Discussion

## 10.1 U-Net Discussion

Dựa trên kết quả thực tế:

- **Điểm mạnh**: Pixel Accuracy ~0.90 và mIoU ~0.445 (val) là kết quả hợp lý cho U-Net nhỏ (base 64, 256×512) huấn luyện 20 epochs. Các lớp chiếm diện tích lớn (road 0.952, vegetation 0.863, sky 0.884, building 0.824, car 0.844) đạt IoU cao.
- **Điểm yếu**: các lớp hiếm/nhỏ bị thấp: rider 0.011, motorcycle 0.042, truck 0.076, wall 0.191, fence 0.188, traffic light 0.218. Nguyên nhân: dữ liệu không cân bằng (class imbalance), độ phân giải huấn luyện thấp `[256,512]` so với bản gốc 1024×2048, và ít epochs. Confusion chủ yếu giữa các lớp ít xuất hiện và lớp nền.

## 10.2 MiDaS Discussion

- MiDaS là mô hình **zero-shot pretrained**, không được huấn luyện trên KITTI trong dự án này.
- Kết quả KITTI val (RMSE 4.26, MAE 3.02, AbsRel 0.85, δ1 0.167) cho thấy khi **median-scaled sang không gian depth tuyến tính của KITTI**, độ khớp còn hạn chế. Đây là hệ quả chính đáng vì MiDaS dự đoán **relative inverse depth** (disparity-like), có thang đo và phân bố khác depth metric; median scaling chỉ căn chỉnh mức tổng thể chứ không hiệu chỉnh phi tuyến.
- **Quan trọng**: dự án không bao giờ khẳng định raw MiDaS là depth mét. Giá trị lớn hơn = gần camera hơn (thuận lợi cho fusion vì gần tương ứng disparity lớn).
- Trong pipeline demo, bản đồ depth vẫn được giữ nguyên nghĩa tương đối và chỉ dùng để phân vùng near/middle/far và tính variation.

## 10.3 Fusion and Scene Understanding

Kết hợp semantic + depth giúp trả lời các câu hỏi mà từng nguồn riêng không đủ:

- Xác định **nearest dynamic class** (cả 2 ảnh demo đều là `car` ở vùng near).
- Đưa ra `drivable_coverage_ratio` (ảnh 1: 0.1317, ảnh 2: 0.1701) và `drivable_median_depth` — thông tin về vùng lòng đường.
- Phân bố depth near/middle/far gần như 1/3–1/3–1/3 trong smoke test (do ngưỡng tercile dẫn xuất từ chính depth map), tức fusion phân chia theo phân phối của riêng từng ảnh.
- Báo cáo `interpretation` bằng ngôn ngữ tự nhiên giúp trình bày kết quả trực tiếp.

## 10.4 Difficulty Analysis

- Điểm difficulty là **heuristic** tổ hợp tuyến tính của 5 chỉ số chuẩn hóa (công thức mục 6.6), với trọng số cấu hình được (`0.25/0.20/0.20/0.20/0.15`).
- Cả 2 ảnh demo đều rơi vào **Medium** (0.541 và 0.508) — chủ yếu do `scene_complexity` cao (gần như đủ 19 lớp) và `depth_variation` cao; `foreground_fraction` cố định ~0.333 do tercile.
- `segmentation_uncertainty` đóng góp vừa phải (~0.32–0.35). Điểm này nhạy cảm với các chỉ số khác nhau; với 2 ảnh không thể nói gì về phân phối tổng thể.

## 10.5 Dataset Limitations

- Cityscapes và KITTI **không ghép cặp**; không thể dùng GT của dataset này cho model của dataset kia.
- Model-level metrics của U-Net và MiDaS đến từ hai benchmark riêng biệt, với dữ liệu GT khác nhau (semantic label vs depth map).
- Pipeline demo dùng ảnh RGB độc lập, **không** có ground-truth để kiểm chứng khách quan — vì vậy kết quả pipeline mang tính minh họa/định tính.
- U-Net chỉ đánh giá trên 500 ảnh val (toàn bộ Cityscapes val); MiDaS trên 1000 ảnh KITTI val. Đây là các split chuẩn, nhưng generalization ngoài phân phối (domain shift) chưa được đo.

---

# 11. Limitations

1. **Model-level thấp ở lớp hiếm** — U-Net cho IoU rất thấp với `rider`, `motorcycle`, `truck`, `wall`, `fence`; không cân bằng được do ít mẫu và resolution thấp.
2. **MiDaS chỉ là relative inverse depth** — không thể đưa ra khoảng cách tuyệt đối (mét); mọi phân tích depth đều mang tính tương đối (near/middle/far).
3. **Smoke test pipeline chỉ có 2 ảnh** — không có ý nghĩa thống kê; không nên đánh giá chất lượng pipeline từ con số difficulty của 2 ảnh.
4. **Difficulty chỉ là heuristic** — không được hiểu là difficulty ground-truth; trọng số/ngưỡng do con người chọn, cần được hiệu chỉnh theo tác vụ cụ thể.
5. **Không có ground-truth cho pipeline** — pipeline-level không được đo bằng metric khách quan (vì ảnh demo độc lập), chỉ có các chỉ số suy diễn từ mô hình.
6. **Chi phí tính toán** — U-Net chạy ở `[256,512]`/batch 1 (do VRAM ~4 GB), không dùng AMP; hạn chế khả năng nâng cao độ chính xác bằng resolution lớn hơn.

---

# 12. Conclusion

Dự án đã xây dựng và đánh giá một **full-pipeline scene understanding** cho ảnh đường phố bằng cách kết hợp **U-Net** (semantic segmentation, huấn luyện trên Cityscapes) và **MiDaS DPT-Large** (pretrained, relative inverse depth) trên **cùng một ảnh RGB**:

- U-Net đạt **Pixel Accuracy ≈ 0.9042**, **mIoU ≈ 0.4451** trên 500 ảnh Cityscapes val; các lớp phổ biến (road, building, vegetation, sky, car) đạt IoU cao, lớp hiếm còn yếu.
- MiDaS được đánh giá trên 1000 ảnh KITTI val với median scaling (RMSE 4.256, MAE 3.018, AbsRel 0.849, δ1 0.167), luôn được mô tả là **relative inverse depth** chứ không phải mét.
- Pipeline **same-image**: ảnh → segmentation + depth → fusion (rule-based) → scene report → difficulty heuristic. Smoke test trên 2 ảnh cho cả hai đều xếp **Medium**, đối tượng động gần nhất là `car` (near).
- Toàn bộ quy trình tái lập được qua CLI, kèm bộ test offline (428 passed, 1 skipped), JSON kết quả và visualization đầy đủ.

Kết quả chứng minh tính khả thi của việc kết hợp ngữ nghĩa và depth tỉ đối cho scene understanding trên ảnh đường phố, đồng thời phơi bày các giới hạn rõ ràng (lớp hiếm, tính chất tương đối của depth, heuristic difficulty và giới hạn của smoke test) — là căn cứ cho các bước mở rộng sau này.

---

# 13. References

1. **U-Net** — O. Ronneberger, P. Fischer, T. Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
2. **MiDaS** — R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, V. Koltun, *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer*, IEEE TPAMI 2022.
3. **DPT (MiDaS DPT-Large backbone)** — R. Ranftl, A. Bochkovskiy, V. Koltun, *Vision Transformers for Dense Prediction*, ICCV 2021.
4. **Cityscapes** — M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, B. Schiele, *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016.
5. **KITTI** — A. Geiger, P. Lenz, R. Urtasun, *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012.

---

*Tài liệu này được tạo tự động từ dữ liệu thực của dự án. Mọi con số đều khớp với các file:* `outputs/analysis/unet_cityscapes_evaluation.json`, `outputs/analysis/unet_training_history.json`, `outputs/analysis/midas_kitti_evaluation.json`, `outputs/analysis/pipeline_evaluation.json`, `configs/unet.yaml`, `configs/midas.yaml`, `configs/pipeline.yaml`.