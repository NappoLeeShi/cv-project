# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

*Coursework Report — Step 00: Project Architecture*

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Nội dung được viết dựa trên Step 00 (Project Architecture) và nguồn thực tế của repository (prompts, README, configs, models, scene_understanding, evaluation, preprocessing). Không có số liệu kết quả nào được bịa ra cho bước này.

---

## 1. Project Overview

Dự án **CV-PROJECT** xây dựng một hệ thống **Scene Understanding** (hiểu cảnh) cho ảnh đường phố (driving scenes). Mục tiêu tổng quát là **Segmentation + Depth → Scene Understanding**: từ một ảnh RGB duy nhất, hệ thống thực hiện đồng thời hai tác vụ, sau đó kết hợp (fusion) hai đầu ra để phân tích bối cảnh giao thông và mức độ khó của cảnh.

Hệ thống dùng bốn thành phần kỹ thuật chính:

- **Semantic Segmentation** — gán một class (nhãn) cho từng pixel, ví dụ `road`, `car`, `person`, `sky`; kết quả là một mask ngữ nghĩa cùng kích thước với ảnh gốc.
- **Monocular Depth Estimation** — ước lượng độ sâu của cảnh từ **một** ảnh RGB duy nhất (không cần stereo camera hoặc LiDAR); MiDaS trả về **relative inverse depth**.
- **Segmentation + Depth Fusion** — kết hợp thông tin ngữ nghĩa và thông tin độ sâu để hiểu cảnh đầy đủ hơn.
- **Scene / Traffic Context Analysis** — dùng thông tin đã fuse để mô tả bối cảnh: đối tượng động, vùng drivable, đối tượng gần nhất…

**Vấn đề cần giải quyết:** trong một cảnh đường phố, mỗi loại thông tin riêng lẻ đều chưa đủ. Chỉ biết một pixel thuộc class `car` thì chưa biết chiếc xe gần hay xa; chỉ biết một vùng gần camera thì chưa biết vùng đó là `road`, `car` hay `person`. Ý tưởng tổng thể của dự án là kết hợp hai nguồn thông tin bổ trợ này bằng hai mô hình chuyên biệt:

- **U-Net** — đảm nhiệm **Semantic Segmentation**: kiến trúc encoder–decoder với skip connections, nhận ảnh RGB và trả về mask class theo **19 lớp Cityscapes**.
- **MiDaS** — mô hình **Monocular Depth Estimation** pretrained (biến thể DPT-Large), nhận ảnh RGB và trả về bản đồ **Relative Inverse Depth** (`larger value = closer`). MiDaS dùng nguyên bản pretrained, không được huấn luyện lại.

Về kiến trúc: nhánh U-Net và nhánh MiDaS chạy **độc lập trên cùng một ảnh RGB** đầu vào, sau đó được hợp nhất tại một **Fusion Module** thuần phân tích (rule-based), không phải mạng nơ-ron. Cityscapes và KITTI là hai dataset riêng biệt và **không được ghép cặp** với nhau (xem Mục 7).

---

## 2. Problem Definition

### 2.1 Input

Input của hệ thống là **một ảnh RGB đường phố duy nhất** (PIL image hoặc numpy array `[H, W, 3]`), ví dụ ảnh trong `data/pipeline/images/`. Yêu cầu quan trọng — **same-image contract**: chính ảnh đó được đưa vào **cả** U-Net **và** MiDaS. Hai model không bao giờ nhận ảnh từ hai nguồn khác nhau trong cùng một lần chạy.

### 2.2 Expected Outputs

Với mỗi ảnh đầu vào, hệ thống tạo ra bốn nhóm đầu ra:

1. **Semantic Segmentation mask** — mảng `[H, W]` chứa trainId `0..18` (19 lớp Cityscapes); `255` = ignore/void.
2. **Relative Inverse Depth map** — mảng `[H, W]` float (độ sâu tỉ đối nghịch đảo; giá trị lớn hơn = gần camera hơn; **không phải metric depth theo mét**).
3. **Fusion / Scene Understanding report** — báo cáo JSON: phân bố semantic, phân bố depth `near`/`middle`/`far`, `traffic_context`, phần diễn giải bằng tiếng Anh.
4. **Difficulty Analysis** — điểm `difficulty_score` trong `[0, 1]` kèm mức **Easy / Medium / Hard** (đây là heuristic, không phải ground-truth difficulty).

### 2.3 What “Scene Understanding” Means Here

Trong dự án này, **Scene Understanding** không dừng lại ở việc gán nhãn pixel hay ước lượng độ sâu. Đó là bước **tổng hợp** để trả lời các câu hỏi mang tính bối cảnh: class nào đang xuất hiện, class nào ở vùng gần/xa, vùng drivable (`road`, `sidewalk`) bao phủ tỉ lệ bao nhiêu, có bao nhiêu dynamic object, đối tượng động gần nhất là gì. Kết quả được mô tả bằng ngôn ngữ tự nhiên và lưu dưới dạng JSON dễ đọc, dễ trích xuất.

---

## 3. System Architecture

Hệ thống có hai nhánh độc lập cùng xuất phát từ một nguồn ảnh duy nhất, sau đó được hợp nhất. Luồng xử lý mức cao đúng như trong `scene_understanding/pipeline.py` và `main.py`:

```
                     RGB Image
                         |
              +----------+----------+
              |                     |
              v                     v
           U-Net                 MiDaS
       Semantic           Relative Inverse
     Segmentation              Depth
              |                     |
              |                     |
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

Diễn giải luồng hoạt động:

1. **Bước 1:** một ảnh RGB được đưa **đồng thời** vào U-Net và MiDaS.
2. **Bước 2:** U-Net trả về segmentation mask; MiDaS trả về bản đồ relative inverse depth.
3. **Bước 3:** hai đầu ra được căn chỉnh về cùng kích thước với ảnh nguồn (mask dùng **nearest-neighbour**, depth dùng **bilinear**) rồi đưa vào Fusion.
4. **Bước 4:** Fusion tính thống kê per-class và các vùng `near`/`middle`/`far`; Scene Analyzer chuyển thành báo cáo cảnh.
5. **Bước 5:** Difficulty Analysis tính điểm khó heuristic từ các indicator của cảnh.

Bảng ánh xạ từng bước của pipeline tới module thực hiện trong repository:

| Pipeline step | Module (file) | Thành phần chính |
|---|---|---|
| Orchestration | `main.py`, `scene_understanding/pipeline.py` | `SceneUnderstandingPipeline.run()` |
| Segmentation branch | `models/unet/inference.py` | `UNetInference.predict()` |
| Depth branch | `models/midas/inference.py` | `MidDepthPredictor.predict()` |
| Fusion | `scene_understanding/fusion.py` | `fuse()` / `FusionResult` |
| Scene analysis | `scene_understanding/analyzer.py` | `analyze_fusion()` |
| Difficulty | `evaluation/difficulty_analysis.py` | `compute_difficulty()` |
| Visualization | `visualization/*` | Ảnh segmentation, depth, fusion, overview |

**Quan trọng (dữ liệu):** cùng **một ảnh RGB** được cung cấp cho **cả U-Net và MiDaS** (same-image contract). Cityscapes và KITTI là hai dataset **riêng biệt** và **không được ghép cặp** với nhau — không bao giờ lấy một ảnh Cityscapes ghép với một depth map KITTI để tính toán chung. Chi tiết thiết kế dataset ở Mục 7.

---

## 4. U-Net Role

**U-Net** là mô hình đảm nhiệm nhánh **Semantic Segmentation**: với mỗi pixel của ảnh đầu vào, mô hình quyết định pixel đó thuộc class nào trong **19 lớp Cityscapes** (trainId `0..18`). Đây là bài toán pixel-wise classification, nên đầu ra là một mask class cùng kích thước không gian với ảnh gốc.

### 4.1 Cấu hình trong dự án

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `num_classes` | `19` | Số lớp trainId Cityscapes |
| `in_channels` | `3` | Ảnh đầu vào RGB |
| `base_channels` | `64` | Số kênh cơ sở của encoder |
| `image_size` (train) | `[256, 512]` | Kích thước ảnh huấn luyện `[H, W]` |
| Checkpoint | `checkpoints/unet_cityscapes.pth` | Trọng số đã huấn luyện trên Cityscapes |

Danh sách 19 lớp: `road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle`.

### 4.2 Vai trò của các thành phần kiến trúc

| Thành phần | Vai trò trong dự án |
|---|---|
| **Encoder** (contracting path) | Trích xuất đặc trưng; giảm dần kích thước không gian qua downsampling (ví dụ `MaxPool2d`). |
| **Convolution** | `DoubleConv` (Conv 3×3 + BatchNorm + ReLU) trích đặc trưng ở từng tầng. |
| **Bottleneck** | Giữ đặc trưng ngữ nghĩa sâu nhất, độ phân giải thấp nhất. |
| **Decoder** (expanding path) | Tăng dần kích thước không gian qua upsampling (ví dụ `ConvTranspose2d`) về resolution gốc. |
| **Skip Connection** | Nối đặc trưng cùng cấp giữa encoder và decoder, giữ lại chi tiết không gian — rất quan trọng cho segmentation. |
| **Output layer** | Conv 1×1 → logits `[C, H, W]`; `argmax(dim=1)` → mask class. |

### 4.3 Điểm kiến trúc cần nhớ

Sau khi dự đoán, mask class được resize ngược về resolution của ảnh nguồn bằng **nearest-neighbour** (không dùng bilinear cho class ID). Inference chạy ở chế độ `eval` + `no_grad`. Trong phạm vi Step 00 chỉ trình bày vai trò kiến trúc; quá trình huấn luyện và siêu tham số chi tiết thuộc các bước sau.

---

## 5. MiDaS Role

**MiDaS** đảm nhiệm nhánh **Monocular Depth Estimation**: từ một ảnh RGB duy nhất, ước lượng độ sâu tương đối của từng pixel. Dự án sử dụng biến thể **DPT-Large** (Vision Transformer backbone) từ kho pretrained `intel-isl/MiDaS`, không huấn luyện lại.

### 5.1 Cấu hình và quá trình inference

| Mục | Giá trị / mô tả |
|---|---|
| Model | MiDaS **DPT-Large**, pretrained (`configs/midas.yaml` → `variant: dpt_large`) |
| Weights | `checkpoints/dpt_large_384.pt` |
| Input size | `384` (vuông), dùng official MiDaS transform |
| Inference | `MidDepthPredictor.predict()` trong `models/midas/inference.py`, chế độ `eval` + `no_grad` |
| Output handling | Resize bilinear về resolution nguồn; đầu ra là bản đồ depth float `[H, W]` |

### 5.2 Ý nghĩa của Relative Inverse Depth

- Output của MiDaS là **relative inverse depth** (dạng disparity-like), **không phải metric depth**: giá trị **lớn hơn = gần camera hơn**.
- Không có thang đo theo mét; `metric_scale` luôn là `None`. Dự án **không** quy đổi raw MiDaS output sang mét trừ khi có phương pháp scale/calibration được giới thiệu tường minh (một số bước đánh giá sau chỉ áp dụng alignment tại thời điểm evaluation, không biến MiDaS thành depth mét).
- Depth normalization / visualization chỉ dùng để hiển thị, không làm thay đổi bản chất tương đối của giá trị depth.

**Vì sao MiDaS bổ trợ cho segmentation?** Segmentation cho biết vùng nào thuộc class nào nhưng không cho biết khoảng cách. MiDaS bổ sung thông tin độ gần/xa, giúp scene understanding trả lời được câu hỏi *đối tượng nào đang ở gần camera*, *vùng drivable có nằm phía trước không*, v.v.

---

## 6. Fusion / Scene Understanding

**Fusion** là module hợp nhất hai đầu ra của U-Net và MiDaS. Đây là bước **bắt buộc** vì mỗi nguồn thông tin riêng lẻ đều không đủ để “hiểu cảnh”:

### 6.1 Vì sao segmentation một mình chưa đủ?

Segmentation biết *vùng này là gì* nhưng không biết *nó ở đâu theo chiều sâu*. Hai cảnh có cùng bố cục ngữ nghĩa có thể rất khác nhau về mức độ nguy hiểm nếu đối tượng ở sát xe hoặc ở rất xa.

### 6.2 Vì sao depth một mình chưa đủ?

Depth biết *vùng này gần hay xa* nhưng không biết *vùng đó là gì*: một vùng gần có thể là `road` (an toàn) hoặc `person` (cần chú ý). Depth tương đối của MiDaS cũng không có giá trị tuyệt đối theo mét.

### 6.3 Cách hai đầu ra được kết hợp

- Hai đầu vào phải **cùng kích thước không gian** (`shape match`) — ràng buộc này trực tiếp phản ánh *same-image contract*.
- Pixel depth không hữu hạn (NaN/inf) và pixel void (`255`) bị loại khỏi mọi thống kê (qua `analyzed_mask`).
- **Per-class stats**: với mỗi class xuất hiện, tính `pixel_count`, `pixel_ratio`, `mean_depth`, `median_depth`, `min_depth`, `max_depth`.
- **Depth regions**: ngưỡng được dẫn xuất từ chính depth map (mặc định terciles `[0.3333, 0.6667]`): `near` = 1/3 giá trị lớn nhất, `middle`, `far` = 1/3 nhỏ nhất. Quy ước **larger inverse depth = closer**.
- Kết quả là `FusionResult` gồm `per_class`, `region_map`, `region_summary`, `thresholds`.

**Scene Analyzer** (`scene_understanding/analyzer.py`) chuyển `FusionResult` thành báo cáo JSON: `scene`, `semantic_distribution`, `depth_distribution`, `regions`, `traffic_context` (vehicles, pedestrians, road, `drivable_coverage_ratio`, `drivable_median_depth`, `dynamic_object_count`, `nearest_dynamic_class`) và `interpretation` (các câu mô tả bằng tiếng Anh).

### 6.4 Fusion là rule-based, không phải mạng học

Fusion được thiết kế **analytical / rule-based** (`scene_understanding/fusion.py`) thay vì một **neural fusion network**. Nhờ đó kết quả dễ kiểm tra, dễ giải thích khi trình bày, phù hợp quy mô coursework và không cần thêm dữ liệu ghép cặp để huấn luyện.

---

## 7. Dataset Design

Dự án thiết kế dữ liệu theo ba vai trò tách biệt. Điều kiện tiên quyết: các loader **ghép cặp theo stem của tên file** (không theo thứ tự list), nên không thể xảy ra lệch cặp.

### 7.1 Cityscapes → U-Net (Semantic Segmentation)

| Mục | Mô tả |
|---|---|
| Vai trò | Huấn luyện và đánh giá model-level của U-Net |
| Layout | `data/cityscapes/` gồm `images/` và `labels/` (`preprocessing/cityscapes.py`) |
| Label | `labelIds` (0..33) remap về trainId `0..18`, `255` = ignore |
| Resize | Label chỉ dùng **nearest-neighbour**; có `require_labels` + `missing_policy` |
| Split | `train` / `val` / `test` |

### 7.2 KITTI → MiDaS (Depth Evaluation)

| Mục | Mô tả |
|---|---|
| Vai trò | Đánh giá model-level của MiDaS (depth estimation / depth evaluation) |
| Layout | `data/kitti/` gồm `images/` và `depth/` (`preprocessing/kitti.py`) |
| Depth | PNG 16-bit, đơn vị millimetre → mét (`/ 1000`); `0` = invalid |
| Giới hạn | `depth_cap_m = 80` đánh dấu pixel xa hơn 80 m là invalid |
| Split | `splits/{train,val,test}.txt`; mặc định `val` |

### 7.3 Custom RGB Images → Full Pipeline Demo

| Mục | Mô tả |
|---|---|
| Vai trò | Demo full pipeline trên ảnh đường phố tự chọn, không cần ground truth |
| Đầu vào | Ảnh trong `data/pipeline/images/` (png/jpg/jpeg/bmp/webp/tiff) |
| Cách chạy | `main.py --input-dir data/pipeline/images` (hoặc `evaluation.evaluate_pipeline`) |
| Kết quả | `outputs/custom/`: original, segmentation, depth, fusion, overview, scene report JSON |
| Checkpoint | `checkpoints/unet_cityscapes.pth` + `checkpoints/dpt_large_384.pt` (inference-only) |

### 7.4 Mối quan hệ giữa các dataset

**Cityscapes và KITTI là hai dataset riêng biệt và KHÔNG được ghép cặp (NOT paired).** Chúng có camera, hình học chụp và hệ nhãn khác nhau. Quy tắc bất biến của dự án:

- Không bao giờ ghép ảnh Cityscapes với depth map KITTI cho một phép tính chung.
- Đánh giá model-level tách biệt: U-Net trên Cityscapes GT; MiDaS trên KITTI GT.
- Full pipeline dùng **một ảnh RGB duy nhất** làm đầu vào cho cả hai mô hình (same-image contract), không cần ground-truth.

---

## 8. Evaluation Design

Step 00 chỉ trình bày **kiến trúc đánh giá** (đánh giá ở đâu, tính metric gì theo thiết kế). Các giá trị số thực tế thuộc các bước sau của coursework và **không được đưa vào tài liệu này**. Dự án thiết kế ba cấp đánh giá:

### 8.1 Model-level — U-Net (Cityscapes val)

- Script: `evaluation/evaluate_unet.py`; dữ liệu: split `val` của Cityscapes.
- Metric thiết kế: **Pixel Accuracy**, **IoU per-class**, **mIoU**, **Dice Score**; class ignore (`255`) bị loại.
- Cấu hình liên quan: `num_classes = 19`, kích thước ảnh `[256, 512]` trong `configs/unet.yaml`.

### 8.2 Model-level — MiDaS (KITTI val)

- Script: `evaluation/evaluate_midas.py`; dữ liệu: split `val` của KITTI depth.
- Metric thiết kế: **RMSE**, **MAE / Absolute Error**, **Abs Rel**, **δ accuracy (δ1/δ2/δ3)**.
- Vì MiDaS là relative inverse depth, ở bước đánh giá áp dụng **median scaling** (`scale = median(gt_valid) / median(pred_valid)`) chỉ để align khi so sánh; MiDaS vẫn là depth tương đối, không phải mét. `depth_cap_m = 80`.

### 8.3 Full-Pipeline Evaluation

- Script: `evaluation/evaluate_pipeline.py`; đầu vào: các ảnh RGB độc lập (không cần ground truth).
- Quy trình: **cùng một ảnh** → segmentation + depth → fusion → scene analysis → difficulty.
- Sản phẩm: JSON per-image + aggregate (`pipeline_metrics.py`, `difficulty_analysis.py`), visualization.
- Không tính metric cần ground truth cho pipeline; difficulty là **heuristic** (`evaluation/difficulty_analysis.py`): tổ hợp tuyến tính có trọng số của 5 indicator chuẩn hóa và phân loại **Easy / Medium / Hard** theo ngưỡng trong `configs/pipeline.yaml`.

> **Lưu ý:** đây chỉ là thiết kế kiến trúc đánh giá cho Step 00. Mọi con số kết quả (mIoU, RMSE, difficulty score…) thuộc các bước sau, không nêu ở đây.

---

## 9. Scope

### 9.1 In Scope

Các khả năng được triển khai trong dự án:

- **Semantic Segmentation** — U-Net huấn luyện trên Cityscapes, checkpoint có sẵn.
- **Monocular Depth Estimation** — MiDaS DPT-Large pretrained (inference-only).
- **Fusion** — module kết hợp rule-based giữa segmentation và relative depth.
- **Scene Understanding** — báo cáo cảnh từ `scene_understanding/analyzer.py`.
- **Difficulty Analysis** — điểm khó heuristic, cấu hình được.
- **Evaluation** — model-level (U-Net/Cityscapes, MiDaS/KITTI) và pipeline-level.
- **Visualization** — ảnh segmentation, depth, fusion, overview.
- **Reproducible CLI & tests** — `main.py`, `evaluation.evaluate_pipeline`, bộ test pytest offline.

### 9.2 Out of Scope

Những khả năng dưới đây **cố tình KHÔNG được triển khai** trong dự án, để giữ ranh giới rõ ràng và đúng quy mô coursework:

| Mục | Trạng thái trong dự án |
|---|---|
| Object Detection / Object Tracking | Không dự phóng bounding box, không theo dõi đối tượng theo thời gian |
| Lane Detection | Không phát hiện làn đường |
| Instance Segmentation | Chỉ có semantic segmentation, không tách instance |
| 3D Object Detection / 3D Reconstruction | Không dựng scene 3D |
| LiDAR–camera fusion | Không dùng LiDAR; depth chỉ đến từ monocular estimation |
| Autonomous driving control | Chỉ phân tích cảnh, không điều khiển phương tiện |
| Metric-depth prediction từ raw MiDaS output | Không quy đổi raw output thành mét (chỉ depth tương đối) |
| Paired Cityscapes–KITTI benchmarking | Hai dataset không ghép cặp; không tạo paired ground truth |
| Learned neural fusion network | Fusion là rule-based, không thêm mạng nơ-ron fusion |
| LLM / Chatbot / AI Agent | Không tích hợp; module tích hợp gọi là **Fusion Module** / **Scene Understanding Module** |
| Unnecessary cloud deployment | Không triển khai hạ tầng cloud phức tạp |

Việc liệt kê Out of Scope ngay từ Step 00 giúp xác định biên của dự án: hệ thống dừng ở mức **phân tích cảnh phục vụ hiểu cảnh đường phố**, không mở rộng sang điều khiển, theo dõi, hay tái tạo 3D.

---

## 10. Project Structure

Cấu trúc cây rút gọn chỉ giữ các thư mục/file quan trọng (đúng repository thực tế):

```
cv-project/
├── configs/               # unet.yaml, midas.yaml, pipeline.yaml
├── data/                  # cityscapes/, kitti/, pipeline/images/
├── models/
│   ├── unet/              # model.py (UNet), inference.py (UNetInference)
│   └── midas/             # model.py (MiDaS DPT-Large), inference.py (MidDepthPredictor)
├── preprocessing/         # cityscapes.py, kitti.py, transforms.py, errors.py
├── scene_understanding/   # fusion.py, analyzer.py, pipeline.py
├── visualization/         # segmentation.py, depth.py, fusion.py, scene.py, io.py
├── evaluation/            # evaluate_*.py, các metrics, difficulty_analysis.py
├── training/              # U-Net training (trainer.py, train_unet.py)
├── utils/                 # config.py, device.py, seed.py, logger.py
├── tests/                 # pytest suite (offline, synthetic fixtures)
├── checkpoints/           # unet_cityscapes.pth, dpt_large_384.pt
├── outputs/               # analysis/, segmentation/, depth/, custom/, visualization/
├── docs/                  # documentation, coursework report
├── main.py                # demo entry point
└── requirements.txt
```

Trách nhiệm của từng thư mục chính:

| Thư mục | Trách nhiệm |
|---|---|
| `configs/` | Cấu hình YAML tách rời logic: `unet`, `midas`, `pipeline` (dataset paths, image size, model params, output paths, seed…) |
| `models/` | Định nghĩa và inference của hai mô hình: `unet/` (U-Net) và `midas/` (MiDaS DPT-Large) |
| `preprocessing/` | Dataset loaders Cityscapes/KITTI và các transform dùng chung; ghép cặp theo stem |
| `scene_understanding/` | Fusion (rule-based), scene analyzer và pipeline orchestration |
| `visualization/` | Xuất ảnh segmentation, depth, fusion, overview và scene report JSON |
| `evaluation/` | Model-level (U-Net, MiDaS) và pipeline-level evaluation, metrics, difficulty analysis |
| `training/` | Logic huấn luyện U-Net (trainer, entry point `train_unet.py`) |
| `utils/` | Config loading, device selection, seed, logger — dùng chung toàn dự án |
| `tests/` | Test pytest offline với synthetic fixtures, không cần dataset/model |
| `checkpoints/` | Trọng số U-Net (`unet_cityscapes.pth`) và MiDaS (`dpt_large_384.pt`) |
| `outputs/` | Kết quả chạy: analysis JSON, ảnh segmentation/depth, custom demo, visualization |

---

## 11. Technology Stack

Công nghệ được liệt kê chỉ dựa trên nội dung thực tế của repository (`requirements.txt` và các module tương ứng).

| Công nghệ | Được dùng cho | Trạng thái |
|---|---|---|
| Python | Toàn bộ mã nguồn, script CLI, test | Required (runtime) |
| PyTorch (`torch`) | Models (U-Net, MiDaS), inference, training | Required |
| NumPy | Xử lý mảng: fusion stats, preprocessing, metrics | Required |
| Pillow (`PIL`) | Đọc/ghi ảnh, dataset loaders, visualization | Required |
| Matplotlib | Biểu đồ hình ảnh trong `visualization/*` | Required |
| PyYAML (`yaml`) | Đọc config trong `utils/config.py` | Required |
| pytest | Bộ test offline trong `tests/` | Required (development) |
| Git | Quản lý phiên bản (repository là git repo) | Required (development) |
| `torchvision`, `opencv-python`, `scikit-learn`, `tqdm` | Có trong `requirements.txt` nhưng không được import trực tiếp trong module của dự án | Declared dependency |

Nguyên tắc chọn stack: chỉ giữ các thư viện thực sự cần thiết cho hai mô hình, pipeline scene understanding và evaluation — không thêm thư viện không dùng tới.

---

## 12. Architecture Summary

Tóm tắt kiến trúc từ A đến Z, có thể dùng để trình bày trực tiếp với giảng viên:

- **Input:** một ảnh RGB đường phố duy nhất.
- **Cùng một ảnh** được đưa đồng thời vào **U-Net** (→ segmentation mask, 19 lớp Cityscapes) và **MiDaS** (→ relative inverse depth, larger = closer).
- Hai đầu ra được căn chỉnh về cùng kích thước rồi đi vào **Fusion Module** (rule-based): tính thống kê per-class và phân vùng `near`/`middle`/`far`.
- **Scene Understanding Module** sinh báo cáo JSON về bối cảnh giao thông (đối tượng động, vùng drivable, đối tượng gần nhất…).
- **Difficulty Analysis** tính điểm khó heuristic trong `[0, 1]` và gán mức **Easy / Medium / Hard**.
- **Dữ liệu tách biệt:** Cityscapes dùng cho U-Net, KITTI dùng cho MiDaS evaluation, ảnh custom dùng cho full-pipeline demo; **không ghép cặp Cityscapes–KITTI**.
- **Ranh giới:** MiDaS giữ nguyên pretrained (không retrain); fusion không phải mạng học; pipeline inference-only; không điều khiển phương tiện.

Nói ngắn gọn khi trình bày: *“Hệ thống nhận một ảnh đường phố, đưa cùng ảnh đó vào U-Net để phân đoạn ngữ nghĩa và vào MiDaS để ước lượng độ sâu tương đối, sau đó kết hợp hai kết quả trong một fusion rule-based để tạo báo cáo hiểu cảnh và đánh giá độ khó của cảnh.”* Việc tách hai nhánh rồi mới fuse giúp hệ thống dễ giải thích, dễ kiểm tra và phù hợp quy mô coursework.

---

## References

1. U-Net — O. Ronneberger, P. Fischer, T. Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
2. MiDaS — R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, V. Koltun, *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer*, IEEE TPAMI 2022.
3. DPT (MiDaS DPT-Large backbone) — R. Ranftl, A. Bochkovskiy, V. Koltun, *Vision Transformers for Dense Prediction*, ICCV 2021.
4. Cityscapes — M. Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016.
5. KITTI — A. Geiger, P. Lenz, R. Urtasun, *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012.

---

*Tài liệu này là một phần của coursework report. Chỉ bao gồm nội dung Step 00 (Project Architecture); các kết quả thử nghiệm thuộc các bước sau.*