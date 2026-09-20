# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

*(Coursework Report — Final consolidated document)*

## Abstract

Dự án xây dựng một pipeline **Scene Understanding** cho ảnh street-scene RGB đơn lẻ: từ **một
ảnh** duy nhất, hệ thống trích xuất đồng thời **semantic segmentation** (trả lời "vật gì ở đâu")
và **monocular depth** (trả lời "gần/xa như thế nào"), sau đó kết hợp hai nguồn thông tin này bằng
**fusion rule-based** để phân tích ngữ cảnh cảnh (traffic context) và tính điểm **difficulty** một
cách minh bạch.

- **U-Net** (19 lớp Cityscapes trainId) đảm nhận semantic segmentation; được huấn luyện từ đầu
  trên Cityscapes.
- **MiDaS DPT-Large** (pretrained) đảm nhận depth estimation; đầu ra là **relative inverse depth**
  (giá trị lớn = gần camera), **không phải** depth mét và mô hình này **không được huấn luyện lại**
  trong dự án.
- **Cityscapes** và **KITTI** là hai dataset riêng biệt, được dùng độc lập cho hai phép đánh giá
  model-level tương ứng (không ghép cặp).
- Fusion và difficulty là **rule-based / analytical**, không có mạng fusion học, không có LLM/AI
  agent, không điều khiển phương tiện.

Kết quả đánh giá: U-Net đạt Pixel Accuracy **0.9042**, mIoU **0.4451**, Mean Dice **0.5498** trên
500 ảnh Cityscapes val; MiDaS đạt RMSE **4.2561 m**, MAE **3.0182 m**, AbsRel **0.8489** trên 1000
ảnh KITTI val (đi kèm các giới hạn về bản chất relative depth). Pipeline toàn phần được kiểm
tra bằng **2-image smoke test** (không phải benchmark thống kê), và **custom-image demo** minh họa
định tính đầu cuối. Toàn bộ dự án được kiểm tra hồi quy bằng pytest (**456 passed, 1 skipped**).

---

# 1. REQUIREMENT

## 1.1 Problem Definition

Trong các ứng dụng thị giác máy tính cho phương tiện hoặc giám sát giao thông, một hệ thống chỉ
có "nhận diện loại vật thể" là chưa đủ; cần biết thêm **vị trí gần/xa** của vật thể và **độ phức
tạp của cảnh** để hỗ trợ quyết định. Hai hướng tiếp cận phổ biến nhất là:

- **Semantic segmentation**: phân loại từng pixel theo 19 lớp ngữ nghĩa Cityscapes (road, vehicle,
  person, …) → trả lời **"What is this?"**.
- **Monocular depth estimation**: ước lượng độ sâu từ một ảnh RGB đơn → trả lời **"How near/far is
  it?"**.

Vấn đề: làm thế nào xây dựng một hệ thống kết hợp cả hai nguồn thông tin trên **cùng một ảnh RGB**,
thực hiện fusion rồi đưa ra một báo cáo cảnh (scene report) có cấu trúc cùng điểm difficulty — và
làm được điều đó với kiến trúc minh bạch, tái lập được và kiểm tra được.

## 1.2 Input and Output

**Input:** một ảnh RGB street-scene (từ Cityscapes trong evaluation, hoặc ảnh custom bất kỳ trong
demo).

**Output,** qua 5 khâu:

| Giai đoạn | Output |
|---|---|
| U-Net | Semantic segmentation mask, 19 lớp trainId (0..18), ignore = 255 |
| MiDaS | Relative inverse depth map (giá trị lớn = gần) |
| Fusion | Bản đồ vùng near/middle/far + thống kê depth theo lớp |
| Scene Understanding | Scene report: semantic distribution, depth distribution, traffic context, interpretation |
| Difficulty Analysis | Điểm difficulty [0,1] + level easy/medium/hard |

Kèm theo: visualization (original, segmentation, depth, fusion, overview) và JSON scene report.

## 1.3 Project Objectives

1. Xây dựng pipeline end-to-end: **1 ảnh RGB → U-Net segmentation + MiDaS depth → fusion → scene
   analysis → difficulty analysis** (same-image contract).
2. Huấn luyện và đánh giá model-level U-Net (Cityscapes) và đánh giá model-level MiDaS (KITTI) một
   cách định lượng.
3. Kiểm tra pipeline bằng smoke test và minh họa bằng custom-image demo (inference-only).
4. Đảm bảo chất lượng code bằng pytest regression (**456 passed, 1 skipped**).

## 1.4 Scope

- Semantic segmentation bằng U-Net (mô hình tự huấn luyện).
- Depth bằng MiDaS DPT-Large pretrained (không huấn luyện lại).
- Fusion rule-based, scene understanding, difficulty heuristic.
- Evaluation model-level (Cityscapes/KITTI) + pipeline smoke test + custom demo.

## 1.5 Out of Scope

- Không thiết kế kiến trúc fusion học.
- Không huấn luyện/điều chỉnh MiDaS.
- Không điều khiển phương tiện, không tạo ảnh hưởng tới hành vi hệ thống thật.
- Không cài đặt CI/CD (dùng pytest cục bộ).
- Không coi difficulty heuristic là ground-truth khó khăn của con người.
- Không ghép cặp Cityscapes–KITTI hay tạo paired data.

---

# 2. PURPOSE & TECHNOLOGY SURVEY

## 2.1 Purpose

Dự án minh họa cách kết hợp hai tác vụ thị giác bổ trợ (segmentation + depth) trên một ảnh RGB để
tạo ra hiểu biết cảnh có cấu trúc, đồng thời cung cấp một quy trình phát triển có kiểm tra.
Thiết kế ưu tiên **sự minh bạch và tái lập** thay vì độ chính xác tối đa.

## 2.2 Semantic Segmentation

Segmentation gán **nhãn lớp cho từng pixel** thay vì chỉ khoanh vùng như detection. Điều này cho
phép:

- biết chính xác pixel nào thuộc road, sidewalk, vegetation, vehicle, person…;
- tính tỷ lệ diện tích che phủ của từng lớp (object density, drivable coverage);
- phân biệt vùng "lòng đường/đi bộ" — đầu vào hữu ích cho phân tích giao thông.

## 2.3 Monocular Depth Estimation

Monocular depth ước lượng độ sâu chỉ từ **một ảnh RGB**. Trong dự án MiDaS DPT-Large cho ra
**relative inverse depth** (disparity-like): giá trị lớn = gần. Điểm mạnh: không cần thiết bị
stereo/LiDAR, chạy được trên ảnh đơn bất kỳ. Điểm yếu: **không phải depth mét tuyệt đối** — chỉ
dùng để phân hạng gần/xa tương đối.

## 2.4 U-Net

**U-Net** là kiến trúc encoder–decoder nổi tiếng cho segmentation:

- **Encoder** hạ mẫu (MaxPool2d) và tăng số kênh đặc trưng.
- **Bottleneck** nén thông tin ở độ phân giải thấp nhất.
- **Decoder** khôi phục kích thước (ConvTranspose2d).
- **Skip connections** nối đặc trưng đúng vị trí từ encoder sang decoder → giữ lại chi tiết biên
  vật thể.

Vì dự án cần một mô hình segmentation **tự huấn luyện, đơn giản, dễ cài đặt** trên phần cứng học
tập hạn chế (GPU ~4 GB), U-Net là lựa chọn phù hợp (kích thước nhỏ hơn các backbone transformer,
dịch chuyển kích thước linh hoạt).

## 2.5 MiDaS

**MiDaS** là bộ pretrained cho **zero-shot monocular depth**: train trên nhiều dataset để tổng quát
tốt trên ảnh lạ. Dự án dùng **biến thể DPT-Large** (DPT = Dense Prediction Transformer) vì:

- chất lượng depth tốt hơn các biến thể nhỏ;
- có sẵn weights pretrained qua `torch.hub` / checkpoint cục bộ;
- không cần huấn luyện lại → phù hợp phạm vi dự án.

MiDaS output là **relative inverse depth**; trong dự án **không có bất kỳ training/update
weights/bias** nào lên MiDaS.

## 2.6 Why U-Net + MiDaS?

Hai mô hình bổ trợ cho nhau trên **cùng một ảnh RGB**:

| Câu hỏi | Mô hình trả lời |
|---|---|
| "What is this?" (vật gì, lớp nào) | U-Net segmentation |
| "How near/far is it?" (gần/xa thế nào) | MiDaS relative inverse depth |

Kết hợp lại cho phép trả lời: **"What is where, and how close is it?"** — ví dụ: có một `car` ở
vùng `near`, `person` ở `middle`, đường (road) nằm ở đâu và gần tới mức nào. Đây chính là nền tảng
của fusion + scene understanding.

---

# 3. DATA

## 3.1 Cityscapes

- **Mục đích:** huấn luyện + đánh giá model-level **U-Net segmentation** (19 lớp semantic).
- **Nhãn:** `gtFine labelIds` (0..33) được ánh xạ về **trainId 0..18** qua bảng lookup 256 phần tử
  (`_TRAIN_ID_LOOKUP` trong `preprocessing/cityscapes.py`); các pixel void/không-quan-tâm → **255
  (ignore)**.
- **19 lớp:** road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation,
  terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle.
- **Split:** dùng `train` cho huấn luyện, `val` (500 ảnh) cho đánh giá model-level.
- Ảnh RGB PNG gốc resolution 1024×2048; trong dự án ảnh được xử lý ở `[256, 512]` (train) do giới
  hạn VRAM.

## 3.2 KITTI

- **Mục đích:** đánh giá model-level **MiDaS** (depth evaluation); KITTI **không dùng để train**.
- **Depth GT:** file PNG **16-bit (I;16), đơn vị millimetre** → chia `/scale_mm = 1000` ra mét.
- **Valid depth:** pixel depth `> 0` được coi hợp lệ; `depth_cap_m = 80` đánh dấu các pixel vượt
  giới hạn là invalid; ngoài ra loại pixel GT không-finite, pred không-finite.
- Khoảng 1000 ảnh + 1000 depth (Eigen-style, ~1216×352), depth **sparse** (~76.8% pixel invalid
  trong mẫu test).
- KITTI chỉ dùng cho **evaluation** của MiDaS — tách biệt hoàn toàn khỏi dữ liệu huấn luyện U-Net.

## 3.3 Dataset Separation

**VERY IMPORTANT:** Cityscapes và KITTI là **hai dataset riêng biệt, KHÔNG ghép cặp** với nhau.

- Khác camera, khác hình học, khác hệ nhãn, khác resolution.
- Không có image–depth pair nào chung giữa hai dataset.
- Hai nhánh model đánh giá trên **ground-truth riêng của từng dataset**: U-Net vs Cityscapes GT,
  MiDaS vs KITTI GT.
- Full pipeline dùng **một ảnh RGB** cho cả hai model (same-image contract) và **không cần GT**.

Việc tách biệt là **cố ý** để tránh hiểu nhầm "một ảnh Cityscapes có một depth GT KITTI tương
ứng".

## 3.4 Data Split

| Dataset | Split | Số lượng | Vai trò |
|---|---|---|---|
| Cityscapes | train | (fine, đầy đủ 2,975 theo spec; khóa huấn luyện thực tế dùng setup dự án) | Train U-Net |
| Cityscapes | val | **500 ảnh** | Evaluation U-Net |
| KITTI | val | **1000 ảnh** | Evaluation MiDaS |

(Repo không lưu dataset thật; các pipeline chạy offline bằng fixture tổng hợp trong test. Các
con số 500 / 1000 là `num_samples` ghi trong JSON kết quả evaluation.)

## 3.5 Data Format

- Cityscapes ảnh: PNG RGB; labels: PNG `gtFine_labelIds`.
- KITTI ảnh: PNG RGB; depth: PNG 16-bit.
- Custom demo ảnh: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp` (`SUPPORTED_EXTENSIONS` trong
  `evaluation/custom_demo.py`).

## 3.6 PNG vs JPG

- **Labels và depth phải dùng PNG lossless**: labels lưu giá trị class ID nguyên (JPEG nén mất dữ
  liệu sẽ làm sai ID); depth PNG 16-bit giữ chính xác millimetre.
- Ảnh RGB thuần cho hiển thị có thể chấp nhận JPG; dự án vẫn ưu tiên PNG theo nguồn chuẩn.

## 3.7 Data Visualization

Dự án tạo nhiều loại visualization thực (script `visualization/*`):

- `original`, `segmentation` (19 màu cố định `CITYSCAPES_TRAINID_COLORS`), `depth` (colormap
  `turbo`), `fusion overlay` (alpha_segmentation 0.5, alpha_depth 0.35), `overview` (≥ 4 panel).
- Với pipeline demo: `outputs/analysis/pipeline_demo_001_{fusion,overview}.png`,
  `outputs/analysis/pipeline_demo_002_{fusion,overview}.png`.
- Custom demo: `outputs/custom/<stem>_{original,segmentation,depth,fusion,overview}.png` +
  `<stem>_scene_report.json`.
- Depth luôn dùng nhãn **"Relative inverse depth (larger = closer)"**, không ghi "meters".

---

# 4. METHODOLOGY

## 4.1 Overall Architecture

```
                      RGB Image (cùng một ảnh)
                                 |
            +--------------------+--------------------+
            |                                         |
            v                                         v
          U-Net                                    MiDaS DPT-Large
            |                                         |
            v                                         v
   Semantic Mask (19 class)              Relative Inverse Depth
            |                                         |
            +------------------+----------------------+
                               |
                               v
                         Fusion (rule-based)
                               |
                               v
                     Scene Understanding
                               |
                               v
                      Difficulty Analysis
                               |
                               v
                    Visualization / JSON report
```

**Điểm mấu chốt:** cùng một ảnh RGB được truyền **độc lập** cho cả U-Net và MiDaS (same-image
contract), không có chia sẻ ảnh/resize/preprocessing khác nguồn giữa hai nhánh.

## 4.2 U-Net Architecture

- **Encoder:** 4 tầng `Down = MaxPool2d(2) → DoubleConv`, channels 3→64→128→256→512 (mỗi lần giảm
  nửa kích thước ảnh, nhân đôi channels).
- **Bottleneck:** 512→**1024** channels ở resolution thấp nhất.
- **Decoder:** 4 tầng `Up = ConvTranspose2d(stride=2) → concat skip → DoubleConv`
  (1024→512→256→128→64).
- **Skip connections:** concat đặc trưng encoder `e0..e3` cùng vị trí; nếu lệch kích thước thì
  bilinear-align trước khi concat.
- **Output:** `Conv2d(64, num_classes=19, kernel_size=1)` → logits `[B,19,H,W]` (không softmax
  trong model); inference dùng `argmax`.
- **DoubleConv:** Conv 3×3 pad 1 → BatchNorm → ReLU (×2).
- **Inference:** resolution `[512,1024]` (config) hoặc giới hạn an toàn theo VRAM trong pipeline.
- `UNet.from_config` đọc từ `configs/unet.yaml` (`num_classes: 19`, `in_channels: 3`,
  `base_channels: 64`).

## 4.3 MiDaS Architecture

- **Model:** DPT-Large (Dense Prediction Transformer — ViT backbone) pretrained, tải qua
  `torch.hub.load("intel-isl/MiDaS", ...)` hoặc checkpoint cục bộ `checkpoints/dpt_large_384.pt`.
- **Inference:** ảnh RGB → `input_size 384` (vuông, giữ aspect ratio) → forward trong `no_grad()`
  → clip percentile `[0.05, 0.995]` → bilinear resize về resolution ảnh gốc.
- **Output:** **relative inverse depth** — giá trị **lớn = gần**, nhỏ = xa. Không phải depth mét.
- Kiến trúc DPT **không được cài lại** — chỉ wrap qua `MiDaSModel` / `MidDepthPredictor`, validate
  weights.

## 4.4 Fusion

Fusion là **rule-based / analytical** (không phải mạng học, không LLM):

- Segmentation resize **nearest** (giữ nguyên class ID không tạo ID phân số), depth resize
  **bilinear**, về cùng resolution ảnh gốc.
- Chia depth thành 3 vùng theo **tercile** (`region_quantiles = [0.3333, 0.6667]`):
  `near` / `middle` / `far`.
- Tạo `FusionResult`: ánh xạ pixel theo lớp semantic kết hợp vùng depth; cho mỗi lớp tính
  `pixel_count`, `pixel_ratio`, `mean/median/min/max_depth`.
- Cấu hình: `roi_classes 11..18` (person..bicycle), `drivable_classes [0,1]` (road, sidewalk).

## 4.5 Scene Understanding

Từ `FusionResult`, analyzer (`scene_understanding/analyzer.py`) sinh scene report:

- `semantic_distribution`, `depth_distribution`, per-region depth stats.
- **Traffic context:** danh sách vehicle/pedestrian classes kèm `median_depth` và `proximity`
  (near/middle/far); `nearest_dynamic_class`, `dynamic_object_count`,
  `drivable_coverage_ratio`, `drivable_median_depth`.
- `interpretation`: các câu tự nhiên sinh từ kết quả (không dùng LLM).
- Cấu hình analyzer: `near_quantile 0.35`, `obstacle_frac 0.5`, `person_near_road_margin 0.1`.

## 4.6 Difficulty Analysis

Điểm difficulty là **heuristic rule-based**, **không phải** ground-truth difficulty.

**Công thức:**

```
D = 0.25·U + 0.20·V + 0.20·C + 0.20·F + 0.15·O
```

Trong đó (mỗi indicator chuẩn hóa, clamp về [0,1]):

| Ký hiệu | Indicator | Định nghĩa (theo repo) |
|---|---|---|
| U | segmentation uncertainty | 1 − mean U-Net softmax confidence |
| V | depth variation | coefficient of variation (std/mean) của inverse depth |
| C | scene complexity | số lớp hiện diện / 19 |
| F | foreground fraction | tỷ lệ pixel thuộc vùng near |
| O | object density | tỷ lệ pixel lớp động (person..bicycle) |

**Ngưỡng** (`difficulty.bins` trong `configs/pipeline.yaml`):

| Level | Điều kiện |
|---|---|
| easy | D < 0.4 |
| medium | 0.4 ≤ D < 0.7 |
| hard | D ≥ 0.7 |

Trọng số/ngưỡng đồng nhất giữa config và code (`DEFAULT_WEIGHTS`, `DEFAULT_BINS` trong
`evaluation/difficulty_analysis.py`).

---

# 5. IMPLEMENTATION

## 5.1 Project Structure

```
cv-project/
├── main.py                      # CLI composition root (--image / --input-dir)
├── configs/                     # unet.yaml, midas.yaml, pipeline.yaml
├── checkpoints/                 # unet_cityscapes.pth, dpt_large_384.pt
├── data/
│   ├── cityscapes/              # split images/semantic labels
│   ├── kitti/                   # split images/depth
│   └── pipeline/images/         # custom demo images
├── preprocessing/               # cityscapes.py, kitti.py, transforms.py
├── models/
│   ├── unet/                    # model.py (UNet), inference.py
│   └── midas/                   # model.py (DPT-Large), inference.py
├── training/                    # trainer.py, train_unet.py
├── scene_understanding/         # fusion.py, analyzer.py, pipeline.py
├── evaluation/                  # evaluate_unet/midas/pipeline.py,
│                                # segmentation_metrics.py, depth_metrics.py,
│                                # difficulty_analysis.py, custom_demo.py
├── visualization/               # io.py, segmentation.py, depth.py, fusion.py, scene.py
├── utils/                       # config.py, device.py, seed.py, logger.py
├── tests/                       # pytest suite (offline, synthetic)
├── prompts/                     # 00..17 step prompts
├── outputs/
│   ├── analysis/                # evaluation JSON + pipeline figures
│   ├── visualization/           # single-image outputs
│   └── custom/                  # custom demo outputs
└── docs/coursework/             # step reports 00..17 + final report
```

Thứ tự phụ thuộc được ép buộc: `utils/` → `preprocessing/` → `models/` →
`scene_understanding/` → `evaluation/`·`visualization/` → `main.py`.

## 5.2 Preprocessing

- `ImageTransform` normalize: ảnh `/255` rồi ImageNet normalize (mean `[0.485,0.456,0.406]`,
  std `[0.229,0.224,0.225]`).
- Cityscapes: ánh xạ `labelIds → trainId` qua lookup, pixel void → 255; resize label bằng
  **nearest** (không bilinear).
- KITTI: depth PNG mm `/1000` → cap → resize bilinear → valid mask `depth > 0`.
- Loader ghép cặp ảnh/label hoặc ảnh/depth theo **filename stem** (không theo thứ tự list), tránh
  ghép sai.

## 5.3 U-Net Implementation

- `models/unet/model.py`: `UNet`, `DoubleConv`, `Down`, `Up`, `from_config`.
- `models/unet/inference.py`: `UNetInference` với `predict(image)`, `predict_batch`,
  `return_confidence` (max softmax), checkpoint loader validate
  missing/unexpected/mismatch → `InferenceError`.

## 5.4 U-Net Training

Huấn luyện U-Net từ đầu trên Cityscapes (config `training` block, khớp `configs/unet.yaml`):

| Hyperparameter | Giá trị |
|---|---|
| epochs | 20 |
| batch_size | 1 (batch 4 OOM ~4 GB GPU; 512×1024 + batch 1 ≈ 3 GiB; [256,512] + batch 1 < 1 GiB) |
| learning_rate | 0.0001 |
| weight_decay | 1e-05 |
| optimizer | Adam |
| loss | CrossEntropyLoss(ignore_index=255) |
| image_size | [256, 512] (H×W) |
| seed | 42 |
| mixed_precision | False |

Checkpoint tốt nhất theo **best val mIoU** → `checkpoints/unet_cityscapes.pth` (epoch 20).
Lịch sử train: `outputs/analysis/unet_training_history.json` (best_val_miou 0.4262, best_val_loss
0.3135 tại epoch 20).

## 5.5 MiDaS Implementation

- `models/midas/model.py`: `MiDaSModel` wrap DPT-Large, validate weights; `models/midas/inference.py`:
  `MidDepthPredictor` với preprocess (resize→384, ImageNet normalize, PrepareForNet), forward trong
  `no_grad()`, clip percentile `[0.05,0.995]`, bilinear resize về resolution gốc.
- Dùng checkpoint `checkpoints/dpt_large_384.pt` (~1.37 GB); không tải tự động.

## 5.6 Evaluation Modules

- `evaluation/segmentation_metrics.py`: Pixel Accuracy, per-class IoU, mIoU, Dice, class accuracy —
  từ confusion matrix **toàn cục 19×19** cộng dồn; pixel 255 loại trừ trước khi xây matrix.
- `evaluation/depth_metrics.py`: RMSE, MAE, AbsRel, δ1/δ2/δ3, `median_scale`.
- `evaluation/evaluate_unet.py` → `outputs/analysis/unet_cityscapes_evaluation.json`.
- `evaluation/evaluate_midas.py` → `outputs/analysis/midas_kitti_evaluation.json`.
- `evaluation/evaluate_pipeline.py` → `outputs/analysis/pipeline_evaluation.json`.
- `evaluation/custom_demo.py` → `outputs/custom/` (custom demo).

## 5.7 Fusion and Scene Analysis

- `scene_understanding/fusion.py`: `fuse(segmentation, depth)`, `FusionResult`, region map theo
  tercile.
- `scene_understanding/analyzer.py`: `analyze_fusion` → scene report JSON.
- `scene_understanding/pipeline.py`: `SceneUnderstandingPipeline` (dependency injection, bất kỳ
  object có `predict(image)`), `PipelineResult`, `PipelineError`; đảm bảo **same-image contract**
  (kiểm tra identity `seg.last_input is depth.last_input`).

## 5.8 Visualization

- `visualization/io.py`: `save_visualization`, `save_figure(dpi=150)`.
- `visualization/segmentation.py`: `colorize_segmentation`, `create_segmentation_overlay`,
  `create_segmentation_legend`.
- `visualization/depth.py`: `normalize_depth_for_visualization`, `colorize_depth(turbo)`,
  `create_depth_figure` — nhãn **"Relative inverse depth (larger = closer)"**.
- `visualization/fusion.py`: `create_fusion_overlay`, `create_region_visualization`.
- `visualization/scene.py`: `create_scene_summary`, `save_scene_report` (JSON UTF-8, indent 2),
  `create_full_visualization`.
- Output dirs: `outputs/visualization` (single image), `outputs/custom` (demo), `outputs/analysis`
  (evaluation figures).

## 5.9 Custom Image Demo

Lệnh đã xác minh (folder-based, `evaluation/custom_demo.py` + `main.py`):

```bash
python main.py --input-dir data/pipeline/images          # xử lý tất cả ảnh trong thư mục
python main.py --input-dir data/pipeline/images --limit 1  # dừng sau 1 ảnh (test nhanh)
python main.py --input-dir data/pipeline/images --device auto
```

- Tự động phát hiện ảnh hợp lệ (đuôi `.png/.jpg/.jpeg/.bmp/.tiff/.webp`), sort, `--limit` cắt.
- Đọc **một** ảnh RGB dùng chung cho U-Net + MiDaS.
- Output `outputs/custom/<stem>_{original,segmentation,depth,fusion,overview}.png` + JSON.
- `--output-dir` tùy chọn; `--no-visualization` chỉ ghi JSON.
- Checkpoint mặc định `checkpoints/unet_cityscapes.pth` / `dpt_large_384.pt`; thiếu → lỗi rõ tên
  file; không tải mạng.
- Grayscale → chuyển RGB + cảnh báo; ảnh hỏng → `CustomDemoError` kèm tên file.

## 5.10 Configuration and Utilities

- `utils/config.py`: `load_config(name, overrides, use_cache)`, `get(cfg,"a.b.c",default)`,
  `resolve_path`, `config_path`; deep-copy/deep-merge, cache không bị poisoning; YAML `safe_load`.
- `utils/seed.py`: `set_seed(42)` — random, PYTHONHASHSEED, NumPy, PyTorch CPU+CUDA, cuDNN
  deterministic.
- `utils/device.py`: `resolve_device("auto"|"cpu"|"cuda")`.
- `utils/logger.py`: root logger `cvproject`, SHA-256 config fingerprint, RotatingFileHandler
  (logs → `outputs/logs/pipeline.log`, 1 MB × 3).

---

# 6. SYSTEM BUILD FLOW

## 6.1 Development Workflow

Dự án phát triển theo các bước nhỏ, mỗi bước một prompt (00..17): spec → implement + review →
tests → report. Chi tiết trong §9.

## 6.2 U-Net Build Flow

```
Cityscapes preprocessing → configs/unet.yaml → UNet.from_config
  → training/train_unet.py (Adam lr 1e-4, CE ignore 255, best val mIoU)
  → checkpoints/unet_cityscapes.pth
  → models/unet/inference.py (UNetInference)
```

## 6.3 MiDaS Build Flow

```
checkpoints/dpt_large_384.pt (pretrained DPT-Large)
  → models/midas/model.py (MiDaSModel)
  → models/midas/inference.py (MidDepthPredictor)
  → KITTI val evaluation (evaluate_midas, median scaling chỉ ở evaluation time)
```

## 6.4 Full Pipeline Flow

```
main.py / evaluate_pipeline / custom_demo
  → same RGB image → UNetInference + MidDepthPredictor
  → alignment (seg nearest, depth bilinear)
  → scene_understanding/fusion.py → analyzer.py → difficulty_analysis.py
  → visualization/* → outputs/
```

## 6.5 Testing

- Framework: **pytest**; toàn bộ test **offline, synthetic** (tensor ngẫu nhiên, ảnh
  `np.random.randint`), không tải dataset/weights/internet; test CUDA `skipif` khi không có GPU.
- Full regression hiện tại: **456 passed, 1 skipped** (1 skip là nhánh CUDA của U-Net inference —
  không phải failure).
- Các bộ test theo bước: config/seed/logger/device, preprocessing, U-Net model, U-Net inference,
  metrics, MiDaS, fusion/analyzer, visualization, pipeline, custom demo, … đều đạt.

## 6.6 CI/CD

**CI/CD không được triển khai trong repository hiện tại.** Không có `.github/workflows`, không có
`gitlab-ci`, không có workflow tự động. Mọi kiểm tra chạy thủ công qua lệnh:

```bash
pytest -q
```

Lý do: mô hình lớn (U-Net ~373 MB, MiDaS ~1.37 GB), dataset nặng, evaluation đắt — không phù hợp
CI thông thường. pytest là cơ chế testing/regression hiện tại.

---

# 7. EXPERIMENTAL SETUP

## 7.1 U-Net Evaluation Setup

- **Validation dataset:** Cityscapes val, `num_samples = 500`.
- **Checkpoint:** `checkpoints/unet_cityscapes.pth` (best epoch 20).
- **Cấu hình:** `num_classes 19`, `ignore_index 255`, image_size `[256, 512]`, `torch.load(
  weights_only=True)`; cross-check num_classes/ignore_index với config; confusion matrix toàn cục
  19×19.
- **Procedure:** forward eval/no_grad toàn bộ 500 ảnh → pixel accuracy, per-class IoU, mIoU, Dice.

## 7.2 MiDaS Evaluation Setup

- **Validation dataset:** KITTI val, `num_samples = 1000`, `num_expected = 1000`, `skipped = 0`.
- **Model:** pretrained DPT-Large (`dpt_large_384.pt`), đầu ra relative inverse depth.
- **Alignment:** **median scaling** ở **evaluation time**:
  `scale = median(GT_valid) / median(pred_valid); aligned = pred × scale`. Đây là alignment
  hằng số, **không** cập nhật weights/bias MiDaS; mean scale 0.2217.
- **Metrics:** RMSE, MAE (đơn vị mét sau alignment), AbsRel, δ1/δ2/δ3; `depth_cap_m = 80`.
- Chỉ tính trên pixel GT hợp lệ (`GT > 0`, finite) và pred finite.

## 7.3 Full Pipeline Setup

- Step 15 dùng **2 ảnh demo** (`pipeline_demo_001.png`, `pipeline_demo_002.png`) trong
  `data/pipeline/images/`, chạy cùng pipeline thật với checkpoint thật, `--limit 2`.
- Đây là **2-image smoke test / small demonstration**, **không phải benchmark** và **không có ý
  nghĩa thống kê**.

## 7.4 Custom Image Demonstration

- Step 17: ảnh RGB bất kỳ đặt vào `data/pipeline/images/`, chạy
  `python main.py --input-dir data/pipeline/images`.
- **Chỉ là inference/demo định tính** — không có GT cho ảnh custom nên **không tính accuracy**;
  không nên nhầm với đánh giá model như Step 13/14.

---

# 8. RESULTS

(Số liệu đọc trực tiếp từ các file JSON trong `outputs/analysis/`.)

## 8.1 U-Net Results

Evaluation trên 500 ảnh Cityscapes val, checkpoint epoch 20:

| Metric | Giá trị |
|---|---|
| Pixel Accuracy | **0.9042** |
| mIoU | **0.4451** |
| Mean Dice | **0.5498** |
| Checkpoint epoch | 20 |
| (training val_mIoU – log) | 0.4262 |
| (training val Pixel Acc – log) | 0.9021 |

## 8.2 U-Net Per-Class Results

Per-class IoU / Dice (từ `unet_cityscapes_evaluation.json`):

| Class | IoU | Dice | Class | IoU | Dice |
|---|---|---|---|---|---|
| road | 0.9522 | 0.9755 | sky | 0.8842 | 0.9385 |
| sidewalk | 0.6793 | 0.8090 | person | 0.5280 | 0.6911 |
| building | 0.8237 | 0.9033 | rider | 0.0115 | 0.0227 |
| wall | 0.1912 | 0.3211 | car | 0.8436 | 0.9152 |
| fence | 0.1881 | 0.3166 | truck | 0.0760 | 0.1412 |
| pole | 0.3986 | 0.5700 | bus | 0.1657 | 0.2842 |
| traffic light | 0.2183 | 0.3584 | train | 0.1307 | 0.2311 |
| traffic sign | 0.5006 | 0.6672 | motorcycle | 0.0422 | 0.0811 |
| vegetation | 0.8630 | 0.9265 | bicycle | 0.5304 | 0.6932 |
| terrain | 0.4290 | 0.6004 | | | |

**Nhận xét:** lớp chiếm diện tích lớn đạt cao (road 0.9522, sky 0.8842, vegetation 0.8630, car
0.8436, building 0.8237); lớp hiếm/nhỏ rất thấp (rider 0.0115, motorcycle 0.0422, truck 0.0760).

## 8.3 MiDaS Results

Evaluation trên 1000 ảnh KITTI val (relative inverse depth, median scaling ở evaluation time):

| Metric | Giá trị |
|---|---|
| Samples | 1000 / 1000 (skipped 0) |
| RMSE | **4.2561 m** |
| MAE | **3.0182 m** |
| AbsRel | **0.8489** |
| δ1 | **0.1671** |
| δ2 | **0.3275** |
| δ3 | **0.4781** |
| Mean scale (median) | **0.2217** |
| relative_inverse_depth | True |

## 8.4 Full Pipeline Results

**2-image smoke test** (từ `pipeline_evaluation.json`) — không phải benchmark:

| Ảnh | Difficulty score | Level | Seg mean conf | Depth std |
|---|---|---|---|---|
| pipeline_demo_001 | 0.541012 | medium | 0.6465 | 9.5462 |
| pipeline_demo_002 | 0.508277 | medium | 0.6761 | 8.5380 |

Aggregate:

| Chỉ số | Giá trị |
|---|---|
| Num images | 2 |
| Average difficulty score | 0.5246 |
| Easy / Medium / Hard | 0 / 2 / 0 |
| Average segmentation confidence | 0.6613 |
| Average depth variation | 9.0421 |

Cảnh báo đúng trọng tâm: kết quả này **dựa trên 2 ảnh** và **không có ý nghĩa thống kê**.

## 8.5 Visualization Results

Các artifact thực đã tồn tại trong repo:

- `outputs/analysis/pipeline_demo_001_fusion.png`, `pipeline_demo_001_overview.png` (+ bản 002).
- `outputs/analysis/pipeline_demo_001_scene_report.json` (scene + difficulty + depth conventions).
- `outputs/custom/*_{original,segmentation,depth,fusion,overview}.png` +
  `*_scene_report.json` (custom demo, 4 ảnh mỗi loại).
- `outputs/visualization/smoke/` (smoke test single pipeline).
- Depth figure luôn chú thích **"Relative inverse depth (larger = closer)"**.

## 8.6 Results Summary

| Tầng | Cơ chế | Nguồn GT | Số mẫu | Metric chính | Giá trị |
|---|---|---|---|---|---|
| U-Net (segmentation) | trained | Cityscapes val | 500 | mIoU | 0.4451 |
| U-Net | trained | Cityscapes val | 500 | Pixel Acc | 0.9042 |
| MiDaS (depth) | pretrained | KITTI val | 1000 | AbsRel | 0.8489 |
| MiDaS | pretrained | KITTI val | 1000 | RMSE | 4.2561 m |
| Pipeline | smoke test | không GT | 2 | difficulty | 0.5246 (trung bình) |
| Custom demo | inference | không GT | 4 | qualitative | — |

**Lưu ý:** không so sánh trực tiếp mIoU (segmentation) với AbsRel (depth) như thể cùng một loại
metric; chúng đo hai tác vụ khác nhau trên hai dataset riêng biệt.

---

# 9. AI-ASSISTED DEVELOPMENT

## 9.1 Development Workflow

Dự án được phát triển với sự hỗ trợ của OpenCode/AI-assisted theo mô hình: **requirement → prompt
→ implementation (AI) + review (student) → tests → validation → report**. Sinh viên là người quyết
định cuối cùng; dự án không tuyên bố AI tự động hoàn toàn. Mỗi bước kết thúc bằng kiểm tra thực
tế (import/smoke/JSON cross-check) và tài liệu `docs/coursework/`.

## 9.2 Prompt Structure

18 prompt `prompts/00_project_architecture.md` … `prompts/17_custom_image_fusion.md`, mỗi prompt gồm:

- objective và phạm vi + anti-scope rõ ràng ("Do NOT implement…");
- yêu cầu đọc đặc tả hiện có;
- test tối thiểu và lệnh chạy;
- validation (import test, pytest, cross-check JSON);
- report và "STOP after Step N".

Bảng topic (source material):

| Prompt | Chủ đề | Prompt | Chủ đề |
|---|---|---|---|
| 00 | Project architecture | 09 | Fusion & scene understanding |
| 01 | Architecture specification | 10 | Visualization |
| 02 | Config & utils | 11 | Demo pipeline (main.py) |
| 03 | Dataset & preprocessing | 12 | U-Net training |
| 04 | U-Net model | 13 | U-Net evaluation |
| 05 | U-Net inference | 14 | MiDaS evaluation |
| 06 | Segmentation evaluation | 15 | Pipeline evaluation |
| 07 | MiDaS | 16 | Coursework report |
| 08 | Depth evaluation | 17 | Custom image fusion demo |

## 9.3 Generated / Modified Scripts

Ánh xạ step → file chính (đã kiểm tra trong repo):

| Step | File tạo ra / sửa |
|---|---|
| 02 | `utils/*`, `configs/*.yaml` |
| 03 | `preprocessing/{cityscapes,kitti,transforms,errors}.py` |
| 04 | `models/unet/model.py` |
| 05 | `models/unet/inference.py` |
| 06 | `evaluation/segmentation_metrics.py` |
| 07 | `models/midas/{model,inference}.py` |
| 08 | `evaluation/depth_metrics.py` |
| 09 | `scene_understanding/{fusion,analyzer}.py` |
| 10 | `visualization/*` |
| 11 | `scene_understanding/pipeline.py`, `main.py` |
| 12 | `training/{trainer,train_unet}.py` |
| 13 | `evaluation/evaluate_unet.py` |
| 14 | `evaluation/evaluate_midas.py` |
| 15 | `evaluation/{evaluate_pipeline,difficulty_analysis,pipeline_metrics}.py` |
| 16 | `docs/report/coursework_report.md` |
| 17 | `evaluation/custom_demo.py`, demo CLI trong `main.py` |

## 9.4 Testing and Validation

- Toàn bộ bước dùng pytest; tiêu chuẩn validate: import test, focused test suite, JSON cross-check
  (số liệu report khớp file output).
- Full regression: **456 passed, 1 skipped** (pytest 21s ~).
- Không test nào tải dataset/weights thật; MiDaS checkpoint 1.37 GB chỉ dùng ở bước demo thủ công.

## 9.5 Version History

Lịch sử Git thực tế (một số commit gom nhiều step):

| Commit | Ghi chú |
|---|---|
| `a9d3f25` | docs 00–04 |
| `94bdcc5` | docs step 5 |
| `f0f3897` | docs step 6–7 |
| `9d18d9d` | docs step 8–9 |
| `4e8689c` | docs step 10–11 |
| `970c3e4` | docs step 12 |
| `927e99e` | docs step 13–14 |
| `827ab02` | docs step 15–16 |
| `47678b0` | implementation chính Steps 12–17 (unet/midas evaluation, pipeline fusion, custom demo, report) |
| `a9a55ba` | implementation 70% processing (Steps 02–11) |
| `d4a21f6` | unet module |

**Trung thực về batch:** code Steps 02–11 chủ yếu nằm trong một commit lớn `a9a55ba`, Steps 12–17
trong `47678b0`; không phải mỗi step một commit.

## 9.6 Outputs

- Mỗi bước sinh ra `docs/coursework/*.md` mô tả thực trạng + con số đã cross-check.
- Outputs chạy thật nằm trong `outputs/analysis/`, `outputs/custom/`, `outputs/visualization/`.
- Báo cáo cuối cùng này là tổng hợp (không copy nguyên văn Step 16).

---

# 10. DISCUSSION

## 10.1 Overall Interpretation

Dự án cho thấy tính khả thi của việc kết hợp U-Net + MiDaS trên **cùng một ảnh** để tạo scene
understanding có cấu trúc, minh bạch và kiểm tra được.

## 10.2 U-Net Discussion

mIoU **0.4451** và Pixel Accuracy **0.9042** trên Cityscapes val là kết quả hợp lý cho một U-Net
nhỏ (base 64) train 20 epochs ở `[256,512]` — **không nên tuyên bố quá mức**. Các lớp phổ biến
(road/sky/vegetation/car/building) đạt tốt; lớp hiếm/nhỏ yếu (rider 0.0115, motorcycle 0.0422,
truck 0.0760) do class imbalance, resolution thấp, thời lượng train ngắn.

## 10.3 MiDaS Discussion

AbsRel 0.8489, δ1 0.1671 trên KITTI val phản ánh đúng giới hạn của **relative inverse depth**:
MiDaS chưa được train trên KITTI, full-frame eval, và output không có scale tuyệt đối; median
scaling chỉ căn chỉnh mức tổng thể (hằng số) chứ không sửa sai số phi tuyến. Vì vậy dự án **không
bao giờ** mô tả MiDaS raw là depth mét.

## 10.4 Fusion / Scene Understanding Discussion

Kết hợp semantic + depth mang lại giá trị mà từng nguồn riêng không có: trả lời "đối tượng động
nào gần nhất, ở vùng nào", tính `drivable_coverage_ratio`, phân bố depth theo lớp, và các câu
interpretation ngôn ngữ tự nhiên. Fusion rule-based giữ tính **giải thích được, quyết định**,
debug dễ; nhược điểm là quy tắc thủ công có thể không tổng quát trên ảnh lạ.

## 10.5 Difficulty Analysis Discussion

Điểm difficulty là **heuristic** (tổ hợp tuyến tính 5 indicator với trọng số config) — hữu ích để
so sánh tương đối giữa các cảnh, nhưng **không phải** difficulty ground-truth của con người và
chưa được hiệu chỉnh trên dữ liệu có nhãn khó.

## 10.6 Strengths

- Same-image contract nhất quán; hai nhánh model tách biệt, đánh giá đúng GT riêng từng dataset.
- Kiến trúc minh bạch, modular, dependency layering rõ; config hóa mọi chỉ số quan trọng.
- Regression mạnh với pytest offline (456 passed, 1 skipped); test deterministic, không tải mạng.
- Demo end-to-end dùng được (`--input-dir`), không cần GT/annotation.

## 10.7 Weaknesses

- U-Net yếu trên lớp hiếm; resolution huấn luyện thấp do VRAM.
- MiDaS là relative depth → không có thông tin khoảng cách tuyệt đối.
- Difficulty heuristic phụ thuộc trọng số thủ công.
- Kết quả pipeline chưa có đối chứng định lượng đủ (smoke test 2 ảnh).

## 10.8 Limitations

- Không có ground-truth cho ảnh custom → không thể khẳng định độ chính xác pipeline trên ảnh lạ.
- Cityscapes–KITTI không ghép cặp nên không có paired GT để đánh giá fusion đúng sai.
- Fusion không phải mạng học → không học được quan hệ phức tạp nếu cần.
- Chưa có CI/CD tự động.

## 10.9 What the Results Do and Do Not Prove

- **Quantitative:** đánh giá U-Net trên 500 ảnh Cityscapes (mIoU 0.4451) và MiDaS trên 1000 ảnh
  KITTI (AbsRel 0.8489) là định lượng trên GT có thật.
- **Smoke test:** kết quả pipeline dựa trên **2 ảnh** → chỉ là smoke test, **không** có ý nghĩa
  thống kê, **không** gọi là benchmark.
- **Qualitative:** custom-image demo là inference minh họa định tính, không phải đánh giá accuracy.
- Không tuyên bố khả năng tổng quát hóa ngoài phân phối dữ liệu.

---

# 11. CONCLUSION

## 11.1 Project Summary

Dự án xây dựng thành công pipeline **Scene Understanding**: một ảnh RGB → U-Net semantic
segmentation (19 lớp) + MiDaS DPT-Large relative inverse depth → fusion rule-based → scene analysis
→ difficulty analysis → visualization + JSON. Toàn bộ quy trình tái lập được, có config, có test.

## 11.2 Main Achievements

- U-Net tự huấn luyện (Cityscapes) đạt mIoU 0.4451 / Pixel Acc 0.9042 trên 500 ảnh val.
- MiDaS pretrained đánh giá trên 1000 ảnh KITTI (RMSE 4.2561 m, AbsRel 0.8489), tôn trọng bản
  chất relative inverse depth.
- Pipeline 2 ảnh smoke test: demo_001 0.541012 (medium), demo_002 0.508277 (medium).
- Custom demo end-to-end với CLI `--input-dir`; regression 456 passed / 1 skipped.

## 11.3 Main Experimental Findings

- Segmentation tốt ở lớp lớn, yếu ở lớp hiếm — đề xuất cải thiện class reweighting/resolution.
- Relative depth hữu ích cho phân hạng gần/xa nhưng không đủ cho ứng dụng cần khoảng cách tuyệt
  đối.
- Fusion + scene understanding sinh ra thông tin ngữ cảnh có cấu trúc từ hai nguồn đơn lẻ.

## 11.4 Practical Meaning

Sinh viên có thể thuyết trình: "từ một ảnh RGB, pipeline trả lời vật gì ở vùng gần/xa nào, đối
tượng động gần nhất là gì, đường có diện tích bao nhiêu, cảnh khó dễ ra sao" — toàn bộ bằng mô-đun
minh bạch và con số thực, có kiểm tra hồi quy.

## 11.5 Future Work

*(Chỉ là đề xuất tương lai, chưa được triển khai.)*

- Nâng resolution/batch train U-Net, thêm data augmentation và class weighting cho lớp hiếm.
- Fine-tune hoặc dùng variant nhỏ hơn của MiDaS; nghiên cứu đo lường bias trên ảnh lạ.
- Đánh giá fusion bằng paired data có sẵn hoặc so sánh A/B định lượng.
- Xây dựng benchmark pipeline với tập ảnh lớn hơn và metric pipeline-level.
- Tích hợp CI/CD (build, lint, pytest) — hiện tại chỉ chạy manual.

---

# 12. REFERENCES

1. **Repository documentation:** `docs/coursework/00_project_architecture.md` …
   `docs/coursework/17_custom_image_fusion.md`; `README.md`.
2. **Project prompts:** `prompts/00_project_architecture.md` … `prompts/17_custom_image_fusion.md`.
3. **Source code:** `models/unet/model.py`, `models/midas/{model,inference}.py`,
   `scene_understanding/{fusion,analyzer,pipeline}.py`, `evaluation/*.py`,
   `preprocessing/*.py`, `visualization/*.py`, `utils/*.py`, `main.py`.
4. **Configs:** `configs/unet.yaml`, `configs/midas.yaml`, `configs/pipeline.yaml`.
5. **Results JSON:** `outputs/analysis/unet_cityscapes_evaluation.json`,
   `outputs/analysis/unet_training_history.json`, `outputs/analysis/midas_kitti_evaluation.json`,
   `outputs/analysis/pipeline_evaluation.json`, `outputs/custom/*_scene_report.json`.
6. **U-Net** — O. Ronneberger, P. Fischer, T. Brox, *U-Net: Convolutional Networks for Biomedical
   Image Segmentation*, MICCAI 2015.
7. **MiDaS** — R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, V. Koltun, *Towards Robust
   Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer*, IEEE TPAMI
   2022.
8. **DPT** — R. Ranftl, A. Bochkovskiy, V. Koltun, *Vision Transformers for Dense Prediction*,
   ICCV 2021.
9. **Cityscapes** — M. Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene
   Understanding*, CVPR 2016.
10. **KITTI** — A. Geiger, P. Lenz, R. Urtasun, *Are we ready for Autonomous Driving? The KITTI
    Vision Benchmark Suite*, CVPR 2012.

> Ghi chú: các tham chiếu 6–10 là tài liệu gốc nổi tiếng của từng phương pháp/dataset; dự án nêu
> tên công trình tương ứng, không chèn DOI hoặc citation tự chế.