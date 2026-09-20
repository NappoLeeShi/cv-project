# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

## Coursework Report — Step 04: U-Net Semantic Segmentation Model

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Tài liệu này mô tả **tầng model**: trọng tâm là **U-Net** (Step 04 — mô hình semantic segmentation), và song song là **MiDaS** (monocular depth) ở mức thông tin được repository hỗ trợ.
> Nguồn sự thật: `prompts/04_unet.md`, `prompts/01_architecture_specification.md`, code thực tế (`models/unet/model.py`, `models/midas/*`, `configs/*.yaml`, `tests/test_unet.py`) và `docs/report/coursework_report.md`.
> Step 04 chỉ triển khai **U-Net model + unit tests**; inference, training, metrics, evaluation lần lượt thuộc các bước sau.

---

## 1. REQUIREMENT

### 1.1 Problem

Dự án cần **hai khả năng cảm nhận cảnh** trên một ảnh RGB đường phố:

- **Semantic Segmentation** — biết *vùng nào là gì* (road, person, car, …);
- **Monocular Depth Estimation** — biết *vùng nào gần/xa*.

Hai đầu ra này sau đó sẽ được kết hợp (Fusion) để tạo nên Scene Understanding. Step 04 chịu trách nhiệm cụ thể:

- Viết **kiến trúc U-Net** cho semantic segmentation theo đúng spec;
- Đầu ra là **raw logits** (không softmax trong model);
- `num_classes` lấy từ **config** (mặc định 19 — chuẩn Cityscapes trainId), không hard-code;
- Kèm **unit tests** offline (random tensor, chạy CPU, không cần download/weights).

> **Bước này KHÔNG làm:** inference, training loop, optimizer/scheduler, checkpoint training, metrics, MiDaS, fusion, visualization, pipeline. Tất cả thuộc các bước sau.

### 1.2 Input / Output

| Model | Input | Output | Ý nghĩa |
|---|---|---|---|
| **U-Net** | Tensor `[B, 3, H, W]` float (ảnh RGB chuẩn hóa), ví dụ `[B, 3, 256, 512]` | **Raw logits** `[B, num_classes, H, W]` (vd `[2, 19, 128, 256]`) | Không phải probability; so sánh giữa 19 class để lấy prediction |
| **Class prediction (tại inference)** | Logits từ U-Net | Mask `[H, W]` long, giá trị `0..18` (trainId) | `argmax(dim=1)` theo chiều class |
| **MiDaS** | Tensor `[1, 3, H', W']` (ảnh RGB qua transform chuẩn MiDaS, input_size 384) | **Raw relative inverse depth** `[B, H', W']` | Giá trị lớn hơn = **gần camera hơn**; không phải mét |
| **Depth (sau resize)** | Raw depth | `[H, W]` float32 (relative, resolution nguồn) | Bilinear resize về đúng ảnh gốc |

**Quan trọng:** mọi shape trên đều đã được xác minh từ model và test thực tế (`y.shape == (2, 19, 128, 256)`; import test cho `torch.Size([1, 19, 128, 256])`).

### 1.3 Scope

- **Thuộc Step 04:** `models/unet/model.py` (U-Net + reusable blocks), `tests/test_unet.py`, tham số model trong `configs/unet.yaml`.
- **Bước sau:** `models/unet/inference.py` (Step 05), `models/midas/*` (Step 07), training/checkpoint (Step 12), evaluation (Steps 13–15), fusion (Step 09), visualization (Step 10), pipeline (Steps 11/15/16).
- Vì vậy, khi nói về MiDaS ở tài liệu này, thông tin lấy từ code thực trong repo nhưng **được ghi rõ là sản phẩm của bước sau**, không phải deliverable của Step 04.

---

## 2. PURPOSE & SURVEY

### 2.1 Purpose

- **Semantic Segmentation:** gán nhãn class cho từng pixel — giúp hiểu *nội dung cảnh* (có bao nhiêu vehicle, người ở đâu, đâu là vùng road…). Đây là “tầng ngữ nghĩa” của hệ thống.
- **Monocular Depth Estimation:** dự đoán độ sâu từ một ảnh duy nhất — giúp ước lượng *khoảng cách tương đối* của các vùng (near/middle/far). Đây là “tầng không gian” của hệ thống.

Hai nguồn này **bổ trợ nhau**: chỉ segmentation thì biết có vật gì nhưng không biết xa gần; chỉ depth thì biết gần xa nhưng không biết là gì. Kết hợp cả hai là dữ liệu cho Fusion.

### 2.2 U-Net Survey

U-Net là mạng **encoder–bottleneck–decoder** cổ điển cho segmentation, phù hợp quy mô coursework:

- **Encoder** giảm dần resolution, tăng dần channels → trích xuất đặc trưng ngữ nghĩa;
- **Bottleneck** — đặc trưng trừu tượng nhất, channels cao nhất / resolution thấp nhất;
- **Decoder** tăng dần resolution về kích thước gốc;
- **Skip connections** nối đặc trưng cùng cấp encoder → decoder, giữ **chi tiết không gian** — yếu tố quyết định chất lượng biên của segmentation mask.

Trong dự án, U-Net nhận ảnh Cityscapes đã chuẩn hóa (Step 03) và xuất logits 19 class (Step 04).

### 2.3 MiDaS Survey

**MiDaS** (Mixed Depth and Surface normals — Intel) là model monocular depth estimation **pretrained**. Dự án dùng variant **DPT-Large** (`dpt_large`) lấy qua **torch.hub** từ `intel-isl/MiDaS` (hoặc từ checkpoint cục bộ `checkpoints/dpt_large_384.pt` đã có trong repo).

Đầu ra của MiDaS là **relative inverse depth**: giá trị **lớn hơn = gần hơn**, **không có thang mét** (`is_metric: False`). Đây là quy ước cố định của model — xử lý cho ra depth metric (alignment) là chuyện của bước đánh giá, không nằm trong model.

### 2.4 Why U-Net + MiDaS?

| Model | Trả lời câu hỏi | Kiểu thông tin |
|---|---|---|
| **U-Net** | *“what is where?”* — vùng nào là gì | categorical (19 class) |
| **MiDaS** | *“which regions are closer / farther?”* — vùng nào gần/xa | continuous (relative depth) |

Không model nào thay thế được model kia: segmentation không cho khoảng cách, depth không cho nhãn class. Cả hai cùng chạy trên **một ảnh RGB** (same-image contract) → không gian thống nhất để Fusion căn chỉnh.

> Lưu ý: U-Net **không** làm depth, MiDaS **không** làm segmentation — mỗi model đúng một vai trò được triển khai.

---

## 3. DATA / MODEL RELATIONSHIP

### 3.1 U-Net and Cityscapes

- U-Net là model cho **semantic segmentation**; **Cityscapes** cung cấp semantic ground truth tương ứng (19 trainable class, ignore 255).
- Luồng được nối từ Step 03: `CityscapesDataset` trả `image` tensor `(3, H, W)` đã chuẩn hóa + `label` trainId → U-Net forward → logits `(19, H, W)`.
- U-Net sẽ được **huấn luyện** trên Cityscapes ở bước training (không phải Step 04).

### 3.2 MiDaS and KITTI

- MiDaS là model depth; **KITTI** cung cấp depth ground truth **metric, thưa (sparse), tính bằng mét** để **đánh giá** sự chính xác của MiDaS.
- Vì MiDaS output là *relative inverse depth*, trước khi so sánh với KITTI cần **alignment** (lệ thuộc bước đánh giá depth) — model MiDaS tự nó *không* trả mét.
- Bước này chỉ nối **dữ liệu → model**: U-Net nhận Cityscapes, MiDaS độc lập nhận ảnh (evaluation trên KITTI là bước sau).

### 3.3 Training vs Pretrained

| Câu hỏi | U-Net | MiDaS |
|---|---|---|
| Model huấn luyện trong dự án? | **Có** (được train từ đầu trên Cityscapes — bước training) | **Không** — dùng **pretrained** nguyên bản |
| Pretrained? | Không (khởi tạo ngẫu nhiên cho tới bước training) | Có — qua torch.hub `intel-isl/MiDaS` hoặc checkpoint `checkpoints/dpt_large_384.pt` |
| Trọng số có được cập nhật? | Được (khi training, Step 12) | **Không** — chỉ dùng `eval()`/`no_grad` tại inference |
| Checkpoint từ đâu? | `checkpoints/unet_cityscapes.pth` (repo hiện có — sản phẩm của bước training sau) | `checkpoints/dpt_large_384.pt` (pretrained, tải sẵn) |

### 3.4 Dataset Pairing Rule

**Cityscapes và KITTI là hai dataset riêng biệt, KHÔNG ghép cặp với nhau.** Image Cityscapes #1 không tương ứng với KITTI #1; hai dataset không có ground-truth dùng chung. Trong full pipeline sau này, **cùng một ảnh RGB** được đưa vào cả U-Net lẫn MiDaS (ảnh đó có thể là ảnh tuỳ ý, không thuộc hai dataset trên).

---

## 4. IMPLEMENTATION PLAN

### 4.1 U-Net Model

Kiến trúc từ `models/unet/model.py` (mọi giá trị xác minh từ code):

| Thành phần | Cấu tạo | Chi tiết |
|---|---|---|
| **Input** | `in_channels = 3` | RGB tensor `[B, 3, H, W]`, resolution tùy ý |
| **`DoubleConv`** *(block tái sử dụng)* | Conv → BN → ReLU → Conv → BN → ReLU | `Conv2d(kernel=3, padding=1)`, giữ nguyên kích thước không gian; BatchNorm + ReLU(inplace) sau mỗi Conv |
| **Encoder stages** | `Down = MaxPool2d(2)` → `DoubleConv` | down1: 64→128; down2: 128→256; down3: 256→512; mỗi lần giảm nửa cạnh, nhân đôi channels |
| **Bottleneck** | `MaxPool2d(2)` → `DoubleConv` | 512 → **1024 channels** — resolution thấp nhất, channels cao nhất |
| **Decoder stages** | `Up = ConvTranspose2d(2, stride=2)` → **concat skip** → `DoubleConv` | up1: 1024→512; up2: 512→256; up3: 256→128; up4: 128→64; concat dọc chiều channel trước DoubleConv |
| **Skip connections** | concat `[upsampled, skip]` (dim=1) | 4 skip từ encoder `e0..e3`; trước khi concat, nếu lệch size sẽ `F.interpolate(bilinear)` cái upsampled cho khớp `skip` (an toàn với input chia không chẵn, vd `100×150`) |
| **Output layer** | `Conv2d(64, num_classes, kernel_size=1)` | Đầu ra **raw logits** `[B, num_classes, H, W]` — **không softmax** |

Các block `DoubleConv` / `Down` / `Up` là thành phần riêng (reusable), `UNet` lắp ráp chúng — không viết một hàm forward khổng lồ.

### 4.2 U-Net Architecture

```
Input RGB [B, 3, H, W]
        │
   ┌──── Encoder ────┐            (đặc trưng: H/8, W/8)
   │  inc: 3→64       │
   │  down1: 64→128   │── e1 ───────────────┐  (skip)
   │  down2: 128→256  │── e2 ─────────────────────┐  (skip)
   │  down3: 256→512  │── e3 ───────────────────────────┐  (skip)
   └──────────────────┘                                 │
        │                                               │
    Bottleneck 512→1024                                 │
        │                                               │
   ┌──── Decoder ───────────────────────────────────────┤
   │  up1: 1024→512  ────────────── cat(e3) ────────────┘
   │  up2:  512→256  ────────────── cat(e2)
   │  up3:  256→128  ────────────── cat(e1)
   │  up4:  128→64   ────────────── cat(e0)
   └──────────────────────
        │
   Conv2d 1×1: 64 → num_classes (raw logits)
        │
   Segmentation logits [B, 19, H, W]
        │
   argmax (tại inference) → class prediction mask [H, W]
```

Dù sửa đổi kích thước (vì `MaxPool` ×4 + `ConvTranspose` ×4), U-Net đảm bảo **output cùng H, W với input** nhờ cơ chế align khi concat skip.

### 4.3 MiDaS Model

`models/midas/model.py` + `inference.py` (triển khai ở **Step 07**, trình bày để hoàn thiện bức tranh tầng model):

- **DPT-Large** pretrained, lấy qua `torch.hub.load("intel-isl/MiDaS", variant, pretrained=True)` hoặc load weights từ `weights_path` (đã có `checkpoints/dpt_large_384.pt`);
- **Input preprocessing:** `MidDepthPredictor.preprocess` — ảnh → PIL RGB → **official MiDaS transform** (nếu có) hoặc built-in `ImageTransform`, resize về **input size 384** (vuông) → tensor `[1, 3, H', W']` đưa lên device;
- **Inference:** `model.eval()`, `torch.no_grad()`, forward → raw depth map;
- **Output:** `[H, W]` float32 **relative inverse depth**, resize **bilinear** về **resolution nguồn** của ảnh.

> **MiDaS output là Relative Inverse Depth.** Giá trị lớn hơn = gần camera hơn. KHÔNG phải depth metric tính bằng mét (`is_metric=False`). Chuyển sang mét chỉ xảy ra ở bước đánh giá (alignment).

### 4.4 MiDaS Architecture

Ở mức repository hỗ trợ (code chỉ dùng model có sẵn, không tự xây layers):

- **DPT (Dense Prediction Transformer)** — kiến trúc dựa trên **Vision Transformer** làm backbone, được thiết kế cho dense prediction;
- Output head dự đoán **bản đồ depth** (đơn kênh);
- Một ảnh → backbone → **depth map** cùng kích thước không gian (sau khi resize).

Dự án **không reimplement** mạng MiDaS mà chỉ **tích hợp** (wrapper `MiDaSModel` quanh backend) + validate weights/checkpoint (`MiDaSError` khi thiếu/trùng/lệch shape).

### 4.5 Hyperparameters

**U-Net — model parameters (đúng Step 04, từ `configs/unet.yaml` nhóm `model`):**

| Tham số | Giá trị |
|---|---|
| `model.name` | `unet` |
| `num_classes` | `19` (Cityscapes trainId) |
| `in_channels` | `3` |
| `base_channels` | `64` |
| Input image tensor | `[B, 3, H, W]` (resolution tùy ý, hỗ trợ size lẻ) |
| `inference.image_size` | `[512, 1024]` *(config inference)* |
| `data.image_size` | `[256, 512]` *(config training, cho bước sau)* |
| `env.device` | `auto` (CPU/CUDA tùy môi trường) |

**U-Net — training hyperparameters (đã khai báo trong config nhưng hoạt động training thuộc Step 12 — không phải Step 04):** `train.epochs=60`, `learning_rate=0.001`, `optimizer=adam`, `loss=weighted_ce` (block `train`); block `training` (Step 12): `epochs=20, batch_size=1, learning_rate=0.0001`.

**MiDaS — model/inference settings (từ `configs/midas.yaml`):**

| Tham số | Giá trị |
|---|---|
| `model.variant` | `dpt_large` (tùy chọn `dpt_hybrid`/`midas_small`) |
| `model.source` | `hub` |
| `preprocessing.input_size` | `384` |
| `inference.device` / `dtype` | `auto` / `float32` |
| `inference.clip_percentile` | `[0.05, 0.995]` |
| `depth.representation` | `relative` (không phải mét) |
| `output.larger_is_farther` | `true` (cho visualization) |

### 4.6 Pretrained vs Training

| Aspect | U-Net | MiDaS |
|---|---|---|
| Main task | Semantic segmentation | Monocular depth estimation |
| Training in project | **Có** (từ đầu trên Cityscapes, bước sau) | **Không** |
| Pretrained | Không | Có (torch.hub / `dpt_large_384.pt`) |
| Weight update | Được (bước training) | Không (chỉ `eval`/`no_grad`) |
| Checkpoint | `checkpoints/unet_cityscapes.pth` (bước training) | `checkpoints/dpt_large_384.pt` |
| Output | Logits `[B, 19, H, W]` → mask trainId | Relative inverse depth `[H, W]` (larger = closer) |

### 4.7 Implementation

**U-Net flow:**

```text
Input image
 → preprocessing (Step 03: ImageTransform [H,W] → (3,H,W) → ImageNet normalize)
 → tensor [1, 3, H, W]
 → U-Net forward
 → logits [1, 19, H, W]
 → argmax(dim=1)+resize nearest → class prediction mask [H, W]
```

**MiDaS flow:**

```text
Input image
 → PIL RGB → official MiDaS transform (resize 384)
 → tensor [1, 3, H', W']
 → DPT-Large forward (eval, no_grad)
 → raw prediction [B, H', W']
 → bilinear resize về resolution nguồn
 → relative inverse depth [H, W] (larger = closer)
```

### 4.8 Post-processing

- **U-Net (Step 04):** model chỉ trả **logits** — không hậu xử lý nào bên trong model. `argmax(dim=1)` và resize-nearest về ảnh nguồn, cùng confidence map (`max softmax probability`) khi cần, nằm ở **inference (Step 05)** trong `models/unet/inference.py` — không thuộc Step 04.
- **MiDaS:** raw output is inverse-relative; `MidDepthPredictor.predict` tự resize **bilinear** về ảnh gốc; `normalize_visualization` chuẩn hóa min-max về `[0, 1]` **chỉ cho hiển thị** (đảo convention để larger = farther) — không phải calibration sang mét.
- **Không trộn với Fusion ở đây:** Fusion (rule-based) xử lý hai đầu ra này ở bước sau, không nằm trong model implementation.

---

## 5. SYSTEM BUILD FLOW

### 5.1 U-Net Build

```text
UNet(num_classes=19, in_channels=3, base_channels=64)   # model.py, đọc từ config
 → from_config(cfg)                                     # nhóm model: num_classes/in_channels/base_channels
 → forward(x): inc → down1..3 → pool → bottleneck → up1..4 (+skip concat) → out 1×1
 → raw logits [B, 19, H, W]
 → (bước sau) load checkpoint safe (validate keys/shape) → eval/no_grad → argmax → mask
```

### 5.2 MiDaS Build

```text
build_midas_model(variant="dpt_large", weights_path=...)   # torch.hub hoặc checkpoint local
  → backend pretrained, wrapper MiDaSModel, .to(device), .eval()
build_midas_transform(...)                                 # official transforms
MidDepthPredictor(model, device, input_size=384, transform=official)
  → preprocess → forward → raw relative depth → bilinear resize về ảnh nguồn
```

### 5.3 Full Pipeline Connection

- Step 04 chỉ định nghĩa **model-level output**: U-Net → mask, MiDaS → depth map; **Fusion/Scene Understanding thuộc các bước pipeline sau** (fuse hai output đã được căn chỉnh không gian).
- Ràng buộc same-image: ảnh RGB giống nhau được đưa vào **cả hai model**; hai output cùng resolution nguồn → sẵn sàng cho Fusion.

### 5.4 Unit Tests

`tests/test_unet.py` — **17 test** (offline, random tensors, CPU; CUDA chỉ chạy khi có GPU), kiểm tra hành vi thật (không phải import-only):

| Nhóm | Test | Xác minh |
|---|---|---|
| Khởi tạo | `model = UNet(num_classes=19)` | Trả `nn.Module` |
| Forward | `x = randn(2,3,128,256)` | `y.shape == (2,19,128,256)` |
| Số class | `num_classes ∈ {5, 32, 1}` (parametrize) | `y.shape[1] == num_classes` |
| Resolution | `(128,256)`, `(256,512)`, `(64,64)` (parametrize) | output khớp input H,W |
| Size lẻ | `randn(2,3,100,150)` | output `(2,19,100,150)` — align an toàn |
| Batch | batch 4 | `y.shape[0] == 4` |
| Gradient | `model(x).mean().backward()` | tồn tại gradient hữu hạn |
| CPU/CUDA | CPU luôn; CUDA `skipif` | chạy được CPU; CUDA tùy chọn |
| Raw logits | `softmax(y) != y` | model **không** trả probability |
| `in_channels` | `UNet(...,in_channels=1)` | khả dụng với 1 kênh |
| `base_channels` | `base_channels=8` | khả dụng |
| `from_config` | nhóm `model` trong `configs/unet.yaml` | đọc đúng cấu hình |

Kết quả chạy thực tế:

```text
.venv/bin/python -m pytest tests/test_unet.py -q
17 passed in 8.96s
```

Import/forward check (đúng lệnh spec):

```bash
.venv/bin/python -c "from models.unet.model import UNet; import torch; m=UNet(num_classes=19); x=torch.randn(1,3,128,256); y=m(x); print(y.shape)"
# → torch.Size([1, 19, 128, 256])
```

Các test Step 02/03 vẫn phải xanh (tổng bộ pytest chạy không break gì).

### 5.5 CI/CD

**Không nằm trong Step 04** — repository chưa có GitHub Actions cho tầng model; CI/CD được nhắc đến ở các bước sau (dùng pytest làm cổng kiểm tra cục bộ tại bước này).

---

## 6. EVALUATION

*Section này chỉ mô tả thiết kế đánh giá — không có số liệu.*

### 6.1 U-Net Evaluation

Sẽ được đánh giá trên Cityscapes val bằng **mIoU** (class-balanced) và các metric hỗ trợ (Per-class IoU, Pixel/Class Accuracy). Chi tiết/chạy thuộc bước đánh giá segmentation.

### 6.2 MiDaS Evaluation

Sẽ được đánh giá trên **KITTI depth GT** (metric, sparse) sau khi **alignment** (vd least-squares scale+shift) vì output là relative; metric chính **AbsRel**, kèm RMSE/δ accuracy, cap 80 m. Chi tiết thuộc bước đánh giá depth.

### 6.3 Model-level vs Pipeline-level

- **Model-level:** đánh giá từng model trên ground truth riêng của dataset (U-Net↔Cityscapes, MiDaS↔KITTI).
- **Pipeline-level:** đánh giá toàn pipeline (same RGB image → segmentation → depth → fusion → scene) — không cần GT, dùng các phiên bản ảnh tuỳ ý.

Step 04 không nằm trong nhánh nào của evaluation; chỉ cung cấp model để các bước sau đo.

### 6.4 Metrics

- **Segmentation:** `mIoU` (chính), IoU per-class, Pixel Accuracy.
- **Depth:** `AbsRel` (chính), RMSE, δ1/δ2/δ3 accuracy, sau alignment với GT KITTI hợp lệ (valid mask).

Chỉ nêu tên metric theo Step 01; không có con số nào ở bước này.

### 6.5 Results

> **Step 04 focuses on model implementation; experimental results are reported in later evaluation steps.**

## 7. AI-ASSISTED DEVELOPMENT

### 7.1 Prompt

Task brief dùng cho bước này: `prompts/04_unet.md` — yêu cầu chỉ U-Net model + tests, đọc trước 01/02/03, không làm inference/training/MiDaS/metrics.

### 7.2 Generated / Modified Scripts

| File | Trách nhiệm |
|---|---|
| `models/unet/model.py` | Định nghĩa U-Net: `DoubleConv`, `Down`, `Up`, `UNet` + `from_config` |
| `tests/test_unet.py` | 17 unit test offline (khởi tạo, forward, class, resolution, batch, gradient, CPU/CUDA, logits, config) |
| `configs/unet.yaml` (nhóm `model`) | `num_classes: 19`, `in_channels: 3`, `base_channels: 64`, `name: unet` — nguồn cấu hình model |

*Không có checkpoint/output model nào được sinh ra ở Step 04* (model chưa training).

### 7.3 Testing

17 test `test_unet.py` xanh trên CPU, random tensors — **không** cần Cityscapes download, weights, GPU hay internet. Test quan trọng: shape khớp, logits raw (không softmax), gradient chạy, resolution lẻ align.

### 7.4 Version History

Repository quản lý bằng Git. Lịch sử có commit `unet module` (thời điểm sớm) và commit `/docs report step 1-4` gộp báo cáo tiến độ các bước 1–4; **không có** commit riêng tách cho riêng Step 04 để xác minh. Thông tin này dừng ở mức verify được từ `git log`.

### 7.5 Output

- `models/unet/model.py`, `tests/test_unet.py` + tài liệu này;
- Không có checkpoint Step 04 (checkpoint `checkpoints/unet_cityscapes.pth` trong repo là kết quả của bước training sau).

---

## 8. DISCUSSION

### 8.1 Strengths

- **Đúng chuẩn baseline U-Net:** encoder–bottleneck–decoder + skip concat, reusable blocks, đơn giản dễ đọc (spec cấm thêm attention/ASPP/residual).
- **Không hard-code 19:** `num_classes`/`in_channels`/`base_channels` từ config — model tái sử dụng cho số class khác (test `5, 32, 1`).
- **Xử lý size an toàn:** nếu upsampled lệch kích thước skip sẽ bilinear-align trước khi concat → chạy được cả `100×150` (không giả định chia chẵn).
- **Raw logits, không softmax trong model** — linh hoạt cho loss CE (kết hợp lớp softmax/argmax bên ngoài).
- **Device-agnostic** (không `.cuda()` cứng), hoạt động CPU lẫn CUDA.

### 8.2 Weaknesses

- **Chưa training:** model chỉ khởi tạo ngẫu nhiên cho tới bước training — Step 04 chưa có "chất lượng" nào để nói.
- **API lệch nhẹ với spec:** spec gợi ý `Unet` (viết hoa) nhưng code dùng `UNet` — thống nhất theo code hiện có.
- **Bottleneck/`Up` dùng bilinear-align** khi size lệch hơi không thuần `ConvTranspose` — cố ý và đã test, nhưng khác cách U-Net gốc 100%.

### 8.3 Limitations

- **Model limitation (U-Net):** baseline đơn kênh resolution cao (`512×1024`) có thể tốn VRAM; nếu cần chất lượng hơn là chuyện của các bước sau.
- **Implementation limitation:** phần inference/confidence (`return_confidence`, resize nearest về nguồn) đã hiện diện trong `models/unet/inference.py` nhưng là **Step 05**, không phải deliverable của Step 04.
- **Intentionally deferred:** training, checkpoint, metrics, MiDaS, fusion, evaluation — rõ ràng thuộc các bước sau (Step 05 trở đi). Không biến giả định thành sự thật.

---

## 9. CONCLUSION

Tóm tắt A-to-Z dễ trình bày:

1. **Why U-Net:** hệ thống cần biết *vùng nào là gì* → semantic segmentation; U-Net là baseline chuẩn, phù hợp coursework.
2. **What U-Net receives:** tensor `[B, 3, H, W]` RGB đã chuẩn hóa (từ Cityscapes loader Step 03).
3. **What U-Net outputs:** raw logits `[B, 19, H, W]` (19 class Cityscapes); `argmax` ở inference thành mask.
4. **Why MiDaS:** hệ thống cần biết *vùng gần/xa* → monocular depth; dùng model pretrained để khỏi huấn luyện lại.
5. **What MiDaS receives:** ảnh RGB qua official transform, resize 384.
6. **What MiDaS outputs:** raw **relative inverse depth** map `[H, W]`.
7. **Why relative inverse depth:** đó là đầu ra cố hữu của MiDaS (larger = closer); chuyển sang mét cần alignment — chỉ làm ở bước đánh giá.
8. **How both combine later:** cùng một ảnh RGB → U-Net mask + MiDaS depth → Fusion (rule-based) → Scene Understanding (bước sau).
9. **Trained vs pretrained:** U-Net được train trong dự án; MiDaS pretrained, trọng số không cập nhật.
10. **Completed at Step 04:** U-Net model + 17 unit test offline xanh, tích hợp config, forward chạy đúng shape — nền tảng model sẵn sàng cho inference/training/evaluation ở các bước sau.

---

## 10. REFERENCES

- `prompts/04_unet.md` — task brief Step 04.
- `prompts/01_architecture_specification.md` — kiến trúc tổng thể, quy ước model/config.
- `models/unet/model.py`, `models/unet/inference.py` — U-Net định nghĩa và inference.
- `models/midas/model.py`, `models/midas/inference.py` — MiDaS wrapper và inference (Step 07).
- `configs/unet.yaml`, `configs/midas.yaml` — tham số model.
- `tests/test_unet.py` — unit tests.
- `docs/report/coursework_report.md` — tài liệu báo cáo dự án (mô tả U-Net/MiDaS/kiến trúc).
- `docs/coursework/00..03` — tài liệu các bước trước (thống nhất thuật ngữ, kiến trúc).