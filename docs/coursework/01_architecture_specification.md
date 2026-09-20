# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

## Coursework Report — Step 01: Architecture Specification

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Tài liệu này mô tả **thiết kế kiến trúc** (Architecture Specification, deliverable của Step 00 được dùng làm đầu vào cho Step 01) dựa trên `prompts/01_architecture_specification.md` và repository thực tế.
> Step 01 là bước **thiết kế (design)**, không phải bước thử nghiệm — tài liệu này không nêu bất kỳ kết quả đo đạc nào.

---

## 1. REQUIREMENT

### 1.1 Problem

Bài toán của dự án là xây dựng hệ thống **Scene Understanding** cho ảnh đường phố với mục tiêu:

> **Segmentation + Depth → Scene Understanding**

Hệ thống cần nhận một cảnh giao thông, hiểu *cảnh đó gồm những gì* (semantic) và *các thành phần ở khoảng cách nào* (depth), sau đó kết hợp hai nguồn thông tin này thành một phân tích bối cảnh (traffic / scene context) dễ đọc và dễ giải thích.

### 1.2 Input / Output

| Item | Giá trị (theo spec) |
|---|---|
| Input | Một ảnh RGB đường phố duy nhất (single RGB street-scene image) |
| Output 1 | Semantic segmentation mask (pixel-wise class labels) |
| Output 2 | Monocular depth map |
| Objective | Kết hợp cả hai đầu ra thành phân tích bối cảnh giao thông có thể diễn giải |
| Datasets | **Cityscapes** → semantic segmentation; **KITTI Depth** → depth estimation/evaluation |
| Models | **U-Net** (segmentation), **MiDaS** (monocular depth, pretrained) |
| Nature | University coursework; fusion **interpretable, rule-based — không có learned fusion network** |

**Ràng buộc dữ liệu tới hạn (critical data constraint):**

- Cityscapes và KITTI là các dataset **riêng biệt (separate)**. Một ảnh từ dataset này **không bao giờ** tương ứng với ảnh từ dataset kia.
- **Không** tạo ra paired segmentation–depth ground truth.
- Fusion chỉ chạy trên hai đầu ra **dự đoán** của cùng **một ảnh đầu vào** tại thời điểm inference.
- Mỗi tác vụ được đánh giá trên ground truth **của chính dataset của nó**.

### 1.3 Scope (tóm tắt)

- **MUST (trong phạm vi):** segmentation, depth estimation, fusion rule-based, scene/context analysis, evaluation, difficulty analysis, visualization, orchestration theo config, CI/CD mức coursework.
- **MUST NOT (ngoài phạm vi):** object detection/tracking, lane detection, instance segmentation, 3D object detection, LiDAR–camera fusion, điều khiển phương tiện, LLM/chatbot, cloud deployment, learned neural fusion network, “AI Agent”. Chi tiết ở mục 7.

### 1.4 Functional Requirements (theo spec)

Spec định nghĩa 20 functional requirement với trách nhiệm gán cho từng module:

| ID | Yêu cầu | Module chịu trách nhiệm |
|---|---|---|
| FR-1 | Load ảnh Cityscapes + annotation fine ghép cặp | `preprocessing/cityscapes.py` |
| FR-2 | Chuyển `labelIds` → trainId 19 lớp, pixel ignore | `preprocessing/cityscapes.py` |
| FR-3 | Load ảnh KITTI + depth GT thưa (Eigen split) | `preprocessing/kitti.py` |
| FR-4 | Transform, giữ image/label **alignment** không gian | `preprocessing/*`, `utils/config.py` |
| FR-5 | Chia dữ liệu train/validation/test | `preprocessing/*` |
| FR-6 | Định nghĩa và build U-Net | `models/unet/model.py` |
| FR-7 | Train / fine-tune U-Net, lưu & resume checkpoint | `models/unet/train.py` *(đề xuất; triển khai thực tế: `training/`)* |
| FR-8 | U-Net inference → segmentation mask | `models/unet/inference.py` |
| FR-9 | Load MiDaS pretrained, preprocess, inference → relative depth | `models/midas/*` |
| FR-10 | Xử lý MiDaS output là depth **relative** (không phải metric) | `models/midas/inference.py` |
| FR-11 | Fuse segmentation + depth → thông tin depth per-class/per-object | `scene_understanding/fusion.py` |
| FR-12 | Phân tích dữ liệu fused → báo cáo traffic/scene context | `scene_understanding/analyzer.py` |
| FR-13 | Đánh giá segmentation (mIoU chính) và depth (AbsRel/δ sau alignment) | `evaluation/segmentation_metrics.py`, `evaluation/depth_metrics.py` |
| FR-14 | Đo pipeline metrics (thời gian, robustness, scene correctness) | `evaluation/pipeline_metrics.py` |
| FR-15 | Phân loại difficulty Easy/Medium/Hard theo luật | `evaluation/difficulty_analysis.py` |
| FR-16 | Visualize image/mask/overlay/depth/fused/pred-vs-GT | `visualization/*` |
| FR-17 | Đọc mọi cài đặt từ `configs/*.yaml` | `utils/config.py` |
| FR-18 | Chạy có seed, log, tái lập được | `utils/seed.py`, `utils/logger.py` |
| FR-19 | Điều phối toàn pipeline từ `main.py` (CLI) | `main.py` |
| FR-20 | Unit/integration test cho mọi thành phần | `tests/*` |

### 1.5 Non-functional Requirements / Architecture Constraints

| Yêu cầu | Tiêu chí (theo spec) |
|---|---|
| Reproducibility | Fixed random seed; mọi tham số nằm trong YAML; logged config hash |
| Depth semantic correctness | MiDaS output **không bao giờ** mô tả là mét; luôn gắn nhãn *relative* |
| Interpretability | Fusion & scene analysis đều rule-based / analytical — không có black-box |
| Modularity | Dependency graph một chiều, không có cycle (mục 5/6) |
| Performance | Mục tiêu ≥ ~2 FPS trên GPU coursework, resolution 512×1024; variant MiDaS small tùy chọn |
| Dependency hygiene | Không thêm thư viện nặng ngoài stack đã chứng minh cần thiết |
| Testability | Mỗi module test được độc lập; unit test trên CPU, GPU tùy chọn |
| Docs | Không thay thế tài liệu tiếng Việt (01–03) của dự án; spec này là phần "how" |

---

## 2. PURPOSE & SURVEY

### 2.1 Purpose

Bước Step 01 phân tích **kiến trúc hiện có** của repository, thiết kế kiến trúc chi tiết (từng module, interface, cấu hình, pool test, CI/CD), và chỉ ra các vấn đề kèm fix đề xuất. Nguyên tắc của spec:

- **Giữ nguyên baseline structure** — không redesign lại toàn bộ.
- Không viết code trong bước này; spec đủ chi tiết để một developer/AI coding agent triển khai theo từng bước mà không phải đổi kiến trúc.
- Tất cả các fix đề xuất đều là **sửa trong file hiện có** (hoặc thêm file nằm trong package đã có), bảo toàn cây thư mục.

### 2.2 Technology Survey

Spec khảo sát và quyết định stack với lý do chọn từng công nghệ:

| Công nghệ | Vì sao cần | Module dùng | Bắt buộc? |
|---|---|---|---|
| **Python** | Ngôn ngữ triển khai | all | Required |
| **PyTorch 2.x** | Models, `Dataset`/`DataLoader`, metrics | `models`, `preprocessing`, `evaluation`, `main` | Required |
| **torchvision** | `transforms`, image I/O, backbones tiềm năng | `preprocessing`, `models/unet`, `models/midas` | Required |
| **OpenCV** (`opencv-python`) | Đọc ảnh, phép tính hình học, colormap, overlay | `preprocessing` (tùy chọn), `visualization` | Optional (Pillow/NumPy cover được; giữ nếu thực sự dùng) |
| **NumPy** | Tính toán mảng, masks, metrics, alignment (least squares) | `preprocessing`, `evaluation`, `fusion`, `visualization` | Required |
| **Pillow** | Đọc/ghi ảnh (đã dùng trong `cityscapes.py`, `inference.py`) | `preprocessing`, `models/unet`, `visualization` | Required |
| **Matplotlib** | Figure montages, panel pred-vs-GT (lưu file, không hiển thị) | `visualization`, `notebooks` | Optional |
| **scikit-learn** | Chỉ nếu có classifier nhỏ; ngược lại **bỏ** | `evaluation` (optional) | Optional — không ép dùng |
| **PyYAML** | Đọc `configs/*.yaml` — spec lưu ý cần thêm vào `requirements.txt` (hiện repo đã có) | `utils/config.py` | Required |
| **pytest** | Test runner — spec lưu ý cần thêm vào `requirements.txt` (hiện repo đã có) | `tests`, CI | Required |
| **tqdm** | Progress bar khi train/eval | `models/unet/train.py`, `main.py` | Optional |
| **Git / GitHub** | Version control, CI qua GitHub Actions | repo | Required |
| **torch.hub** (`intel-isl/MiDaS`) | Lấy pretrained MiDaS chính thống, tránh reimplement | `models/midas` | Required (network một lần) |

Các thư viện **không** được thêm: `ultralytics`, `mmsegmentation`, `open3d`, `datasets`, LLM/cloud SDK.

### 2.3 Capability Comparison

Step 01 không thực hiện một benchmark so sánh độc lập giữa các kiến trúc thay thế (ví dụ U-Net vs DeepLabV3). Lựa chọn model trong spec dựa trên:

- **U-Net** hiện có trong repo (`models/unet/model.py`) — đã được kiểm tra là đúng (input 3 → output 19, resolution giữ nguyên khi side chia hết cho 16), nên **giữ nguyên, không thiết kế lại**.
- **MiDaS** dùng pretrained chính thống qua torch.hub — **không được tự viết lại mạng** DPT.

> Lưu ý: thông tin so sánh năng lực (capability comparison) giữa các phương án khác không nằm trong Step 01, nên tài liệu này không đưa vào.

### 2.4 Why U-Net + MiDaS?

Hai nguồn thông tin bổ trợ cho nhau:

- **Semantic Segmentation (U-Net)** trả lời câu hỏi *“vùng này là gì?”* — biết class của từng pixel nhưng không biết khoảng cách.
- **Monocular Depth Estimation (MiDaS)** trả lời câu hỏi *“vùng này ở độ sâu nào?”* — biết độ tương đối gần/xa nhưng không biết class.

Kết hợp cả hai: biết *một vùng gần camera là `car` hay `person`*, biết *đối tượng động nào đang gần nhất*, biết *vùng drivable (`road`, `sidewalk`) bao phủ bao nhiêu* — tức là dữ liệu cho bước **Scene Understanding**. Cả hai mô hình nhận **cùng một ảnh RGB** (same-image contract), nên kết quả của chúng có thể căn chỉnh không gian với nhau tại Fusion.

---

## 3. DATA

### 3.1 Cityscapes

Dùng cho nhánh **Semantic Segmentation** (huấn luyện + đánh giá model-level):

- Loader `preprocessing/cityscapes.py`: ghép cặp ảnh ↔ label theo quy ước tên file `*_leftImg8bit.png` ↔ `*_gtFine_labelIds.png` (spec chỉ ra cần sửa bug double `replace` — xem 4.10).
- Hỗ trợ layout chuẩn (`leftImg8bit/{train,val,test}`, `gtFine/{train,val,test}`) hoặc layout phẳng, điều khiển bằng config.
- Nhãn `labelIds` được map sang **trainId 0–18**, `255` = ignore.
- Transform **paired** (cùng phép hình học cho image và label; label dùng nearest-neighbour).

### 3.2 KITTI

Dùng cho nhánh **Monocular Depth** (đánh giá model-level):

- Loader `preprocessing/kitti.py`: ảnh RGB + depth PNG thưa (sparse depth GT), ghép cặp theo tên file.
- Giữ **valid mask** (nơi `depth > 0`); có depth cap (`80` m) và crop margin (Eigen protocol center-crop) cấu hình được.
- Trả về `(image, depth_gt, valid_mask)` — phục vụ đánh giá MiDaS, không dùng để “ghép” với Cityscapes.

### 3.3 Data Split

- Cityscapes: split chuẩn `train` / `val` / `test`, qua `get_split(name)`.
- KITTI: split theo file list chính thức (Eigen) `train_val`, qua `get_split(name)`.
- Không có sự trộn ảnh giữa hai dataset.

### 3.4 Data Format

- **Cityscapes labels:** `labelIds` (0..33) → `trainId` (0..18), ignore = `255`. Spec yêu cầu **một bảng class map duy nhất** đặt trong `preprocessing/cityscapes.py`, dùng chung cho dataset, metrics và visualizer. Bảng tiêu biểu (theo Cityscapes `labels.py`):

| labelId | trainId | name | color (R G B) |
|---|---|---|---|
| 7 | 0 | road | 128 64 128 |
| 8 | 1 | sidewalk | 244 35 232 |
| 11 | 2 | building | 70 70 70 |
| 12 | 3 | wall | 102 102 156 |
| 13 | 4 | fence | 190 153 153 |
| 17 | 5 | pole | 153 153 153 |
| 19 | 6 | traffic light | 250 170 30 |
| 20 | 7 | traffic sign | 220 220 0 |
| 21 | 8 | vegetation | 107 142 35 |
| 22 | 9 | terrain | 152 251 152 |
| 23 | 10 | sky | 70 130 180 |
| 24 | 11 | person | 220 20 60 |
| 25 | 12 | rider | 255 0 0 |
| 26 | 13 | car | 0 0 142 |
| 27 | 14 | truck | 0 0 70 |
| 28 | 15 | bus | 0 60 100 |
| 31 | 16 | train | 0 80 100 |
| 32 | 17 | motorcycle | 0 0 230 |
| 33 | 18 | bicycle | 119 11 32 |
| khác | 255 | ignore/void | — |

- **KITTI depth:** PNG 16-bit, đơn vị millimetre → mét (`/ scale_mm`, mặc định 1000); `0` = invalid.

### 3.5 PNG vs JPG

Step 01 không so sánh riêng PNG vs JPG; toàn bộ format dữ liệu trong spec dùng **PNG**:

- Ảnh Cityscapes `*_leftImg8bit.png`, label `*_gtFine_labelIds.png` (giá trị index nguyên — cần lossless).
- Depth KITTI PNG **16-bit** để giữ giá trị millimetre chính xác.

So với JPG (lossy), PNG phù hợp cho label index và depth raw. Custom images cho full-pipeline demo có thể là nhiều định dạng ảnh thông thường khác.

### 3.6 Data Visualization

Step 01 không có phần riêng về “data visualization” cho dataset; khâu hình ảnh được thiết kế ở module `visualization/` (mask màu Cityscapes palette, overlay, depth relative colormap, montage) — trình bày trong mục 4.7.

---

## 4. IMPLEMENTATION PLAN

### 4.1 U-Net Model Usage

- **Input:** RGB, normalize theo config; resolution huấn luyện/dự đoán **512×1024** (chia hết cho 16 → an toàn với 4× pooling).
- **Output:** logits `[1, 19, H, W]` — **không có softmax trong model**; `argmax` tại inference (class space `trainId` 0–18, ignore `255`).
- **Loss:** Weighted Cross-Entropy (+ Dice tùy chọn) để xử lý class imbalance (road ~40% so với pole/bicycle ~1%).
- Modules: `models/unet/model.py` (giữ nguyên, đã kiểm tra đúng), `models/unet/inference.py` (refine), `models/unet/train.py` (đề xuất thêm; repo triển khai trong `training/`).

### 4.2 U-Net Architecture (per-block, khớp `models/unet/model.py`)

| Block | Spec | Channels |
|---|---|---|
| Encoder 1 — `DoubleConv` (Conv 3×3, BN, ReLU ×2) | input RGB | 3 → 64 |
| Downsample | `MaxPool2d(2)` | — |
| Encoder 2 | `DoubleConv` | 64 → 128 |
| Downsample | `MaxPool2d(2)` | — |
| Encoder 3 | `DoubleConv` | 128 → 256 |
| Downsample | `MaxPool2d(2)` | — |
| Encoder 4 | `DoubleConv` | 256 → 512 |
| Downsample | `MaxPool2d(2)` | — |
| **Bottleneck** | `DoubleConv` | 512 → 1024 |
| Decoder 4 | `ConvTranspose2d(1024,512)` → concat skip(e4) → `DoubleConv` | 1024 → 512 |
| Decoder 3 | `ConvTranspose2d(512,256)` → concat skip(e3) → `DoubleConv` | 512 → 256 |
| Decoder 2 | `ConvTranspose2d(256,128)` → concat skip(e2) → `DoubleConv` | 256 → 128 |
| Decoder 1 | `ConvTranspose2d(128,64)` → concat skip(e1) → `DoubleConv` | 128 → 64 |
| Output head | `Conv2d(64, num_classes, kernel_size=1)` | 64 → 19 |

- **Convolution:** tất cả 3×3, padding 1 (giữ không gian).
- **Downsampling:** MaxPool giảm nửa cạnh; BN+ReLU sau mỗi conv.
- **Bottleneck:** `DoubleConv` sâu nhất — đặc trưng ngữ nghĩa trừu tượng nhất.
- **Upsampling:** transposed-conv nhân 2 rồi concat skip.
- **Skip connections:** **concat** (không cộng) — bảo toàn số kênh, giữ chi tiết không gian cho segmentation.

### 4.3 MiDaS Model Usage

- **Model loading:** `torch.hub.load("intel-isl/MiDaS", "DPT_Large")` (mặc định; `MiDaS_small` tùy chọn cho tốc độ). **Không reimplement mạng.** Load cả transform MiDaS từ cùng source. `model.eval()`, device/dtype theo config; weights cache cục bộ sau lần download đầu.
- **Preprocessing:** resize giữ tỉ lệ về input size (DPT_Large: **384**); normalize MiDaS **đúng như bản gốc** (ImageNet mean/std + `shift`/`NormalizeImage`/`PrepareForNet`) — *copy transform, không copy số* để tránh lệch.
- **Inference:** một forward trong `torch.no_grad()`, float32.
- **Output handling:** inverse shift → depth map **relative**; clip theo percentile ổn định (vd 5th–99.5th); resize bilinear về resolution pipeline.

### 4.4 MiDaS Architecture

- Kiến trúc là **DPT (Vision Transformer)** pretrained — spec chỉ tích hợp, không tự xây.
- Output: **relative inverse depth** — giá trị **lớn hơn = gần hơn**, không có thang mét.
- **Depth normalization cho fusion/visualization:** min–max về [0,1] theo từng ảnh; spec khuyến nghị **chọn 1 convention duy nhất** (khuyến nghị: larger = farther) và ghi rõ trong tài liệu. Visualization luôn ghi rõ *"relative depth (unitless)"*.

**Giới hạn (phải được log/hiển thị):**

- Output affine-invariant (scale & shift tùy ảnh) — không so sánh được giữa các ảnh nếu không alignment.
- Không đáng tin ở vùng sky/far-field và vùng ít texture.
- Chỉ depth đơn ảnh, không có geometry thật.
- So sánh với KITTI **chỉ hợp lệ** thông qua thủ tục alignment (mục 4.9).

### 4.5 Hyperparameters (thiết kế cho U-Net)

| Nhóm | Giá trị thiết kế (Step 01) |
|---|---|
| Training | **From scratch** trên Cityscapes fine set (2,975 ảnh train), 512×1024 |
| Class weights | Inverse-frequency, làm mềm về `[0.02, 1.0]` cho Cross-Entropy |
| Augmentation | Horizontal flip, random scale/crop (paired trên label), color jitter; fixed seed |
| Optimizer | Adam, `lr 1e-3` — lịch giảm về `1e-4` |
| Epochs / val | 60 epochs, `val_interval=5` |
| Batch | Cấu hình: 4–8 (GPU 512×1024), 2 (CPU) |
| Checkpoint | Best-val-mIoU `Unet_best.pth` + periodic + latest; lưu `{state_dict, epoch, class_map_version, config}` |
| Fine-tuning | Tùy chọn: load checkpoint tiếp tục, hoặc encoder backbone pretrained (ghi là future work, không bắt buộc) |
| Resume | `--resume <ckpt>` khôi phục từ epoch/optimizer state |

> Các giá trị sinh ra sau cùng được lưu trong `configs/*.yaml`; Step 01 chỉ trình bày thiết kế.

### 4.6 Pretrained vs Training

- **MiDaS:** **pretrained**, dùng nguyên bản qua torch.hub — không huấn luyện lại, không tự viết mạng.
- **U-Net:** **train từ đầu** trên Cityscapes (hoặc fine-tune tùy chọn) — là mô hình duy nhất được huấn luyện trong dự án.
- KITTI **chỉ dùng để đánh giá** MiDaS; không dùng chung để huấn luyện chéo với Cityscapes.

### 4.7 Implementation

**Fusion (rule-based, module contract):**

```text
fusion.fuse(seg_mask: H×W trainId, rel_depth: H×W float, roi_classes, thresholds) -> FusionResult
```

- Chạy trên **cùng một ảnh**; hai map aligned theo construction (cùng resize). Nếu resolution lệch, fusion chịu trách nhiệm resample.
- `per_class`: median/mean/percentile/min/max/coverage của relative depth theo từng class (dùng percentile → chống outlier của MiDaS).
- `objects`: connected components (4-connectivity) trong các class dynamic (`person, rider, car, truck, bus, train, motorcycle, bicycle`) → thống kê depth per-object.
- `free_space`: median depth của `road`+`sidewalk`, nearest obstacle depth, tỉ lệ vùng drivable.
- `layout_bands`: phân bố sky/ground theo hàng, ước lượng horizon.
- `valid_mask`: nơi depth đáng tin.

**Scene Analyzer:** tiêu thụ `FusionResult` → báo cáo dict (`near/far objects`, `object-region depth`, `road context`, `scene complexity`, `traffic context`) với ngưỡng cấu hình được (vd `near_quantile=0.35`, `obstacle_frac=0.5`). Đầu ra là dict thuần để `pipeline_metrics` và `visualization/scene.py` cùng dùng mà không phụ thuộc lẫn nhau.

**Config/Utils:** mọi cài đặt nằm trong `configs/{unet,midas,pipeline}.yaml`, đọc qua `utils/config.py`; `utils/` chỉ chứa plumb config/seed/logger — **không** chứa model, dataset, metrics, transforms, visualization. `main.py` là composition root (orchestrator) — không chứa logic nghiệp vụ.

**Visualization:** `visualization/segmentation.py` (mask màu Cityscapes palette, overlay, pred-vs-GT), `visualization/depth.py` (relative colormap, luôn ghi nhãn relative — không ghi mét), `visualization/scene.py` (montage original/mask/overlay/depth/fused + annotation near/far). Tất cả là hàm **save-to-disk**, không GUI.

### 4.8 Post-processing

- **Segmentation:** `argmax` → mask `[H,W]`; resize về ảnh nguồn bằng **nearest-neighbour** (không bilinear cho class ID).
- **Depth:** inverse shift → clip percentile → bilinear resize về resolution pipeline/ảnh nguồn.
- **Fusion:** chuẩn hóa depth min-max về [0,1] theo từng ảnh + mask void (`255`) và pixel depth không hợp lệ trước khi tính thống kê.
- Convention depth được chọn trước và ghi rõ (relative, unitless) — tránh mô tả sai thành mét.

### 4.9 Evaluation Architecture (thiết kế — không có kết quả ở Step 01)

Spec thiết kế **cách** đánh giá trước khi triển khai:

- **Segmentation (`segmentation_metrics.py`):** **mIoU là metric chính** (class-balanced, chuẩn Cityscapes, kháng imbalance); kèm Per-class IoU, Pixel Accuracy (auxiliary), Class Accuracy, Dice. Tất cả mask theo ignore=255.
- **Depth (`depth_metrics.py`):** bắt buộc **alignment trước khi tính số** — least-squares scale+shift (Eigen) hoặc median-ratio trên valid GT pixel; sau đó tính **AbsRel (chính)**, sq_rel, RMSE, MAE, δ accuracy (δ1/δ2/δ3); báo `valid_frac` (độ phủ thưa). Depth cap 80 m.
- **Pipeline (`pipeline_metrics.py`):** scene-understanding correctness (rule-based trên vài mẫu nhãn tay), inference time per-stage + FPS, robustness theo nhóm difficulty, distribution Easy/Medium/Hard.
- **Difficulty (`difficulty_analysis.py`):** score rule-based tổ hợp 5 indicator (số object, object size, occlusion proxy, segmentation complexity, depth variation) với trọng số đề xuất `0.25/0.20/0.20/0.20/0.15`, clamp [0,1]; bins Easy < 0.4, Medium < 0.7, Hard ≥ 0.7.

> **Không có con số kết quả nào ở Step 01.** Mọi target/reference (nếu có) thuộc các bước đánh giá sau.

### 4.10 Design Decisions & Recommended Fixes (Step 01 phát hiện — chưa áp dụng khi viết spec)

Spec giữ nguyên baseline và đề xuất các fix **trong file hiện có**:

| # | Vấn đề phát hiện | Fix đề xuất |
|---|---|---|
| I-1 | `utils/config.py` `ROOT_DIR = Path(__file__).parent` trỏ nhầm về `utils/` | `Path(__file__).resolve().parent.parent` |
| I-2 | `configs/*.yaml` chưa được dùng; PyYAML/pytest chưa có trong `requirements.txt` | Thêm `load_config()`; thêm PyYAML, pytest (repo hiện đã có) |
| I-3 | `cityscapes.py` double `.replace` sinh label path sai; `label_dir` bị bỏ qua | Thay đúng `_leftImg8bit.png → _gtFine_labelIds.png`; dùng thật `label_dir` |
| I-4 | Label không được transform/resize cùng ảnh (lệch shape) | Paired transforms; label = nearest-neighbour |
| I-5 | Loader trả raw `labelIds` nhưng model/eval giả định `trainId` | Map `labelIds → trainId` qua 1 class map dùng chung |
| I-6 | Không xử lý train/val/test split (glob trộn mọi thứ) | `get_split(name)` |
| I-7 | `models/midas/*` trống | Triển khai theo §13 dùng pretrained (torch.hub) |
| I-8 | Chưa có training module cho U-Net | Thêm `models/unet/train.py` (repo hiện đặt trong `training/`) |
| I-9 | `main.py` là demo segmentation cứng nhắc | Chuyển thành CLI orchestrator đọc `pipeline.yaml` |
| I-10 | Visualization segmentation chỉ ghi grayscale class id | Thêm Cityscapes palette, overlay, pred-vs-GT |
| I-11 | Metrics mới có `pixel_accuracy`, `mean_iou`; depth/pipeline/difficulty trống | Triển khai theo §16 với aligned depth metrics |
| I-12 | `tests/*` trống | Bộ test 10 mục theo §19 + `tests/test_pipeline.py` |
| I-13 | Thiếu `.gitignore` | Thêm `.venv/`, `data/*`, `outputs/*`, `__pycache__`, checkpoints |
| I-14 | Depth không được đánh giá trên KITTI | Giữ hai dataset độc lập cho training/eval; đưa **bất kỳ** ảnh nào qua cả hai model ở inference |

**Kết luận thiết kế:** không redesign; sau các fix, luồng pipeline khớp chính xác với ý tưởng ban đầu (mục 5.3).

---

## 5. SYSTEM BUILD FLOW

### 5.1 U-Net Build

```text
Cityscapes RGB --> CityscapesDataset (ghép cặp theo tên file)
labelIds --> trainId (0..18, ignore 255)
paired transform (hflip / scale-crop / color jitter)
        --> U-Net forward --> logits [1,19,H,W] --> argmax --> segmentation mask [H,W]
        --> (metrics vs Cityscapes GT ở bước đánh giá sau)
```

### 5.2 MiDaS Build

```text
RGB image (KITTI / bất kỳ) --> MiDaS preprocessing (resize 384, official normalize)
        --> MiDaS forward (no_grad) --> relative depth
        --> inverse shift --> clip percentile --> resize bilinear --> depth map [H,W]
```

### 5.3 Full Pipeline

```text
                         RGB Image
                             |
                  +----------+----------+
                  |                     |
                  v                     v
                U-Net                MiDaS
                  |                     |
                  v                     v
          Semantic Mask       Relative Inverse Depth
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

Diễn giải quan trọng (theo spec):

- **Cùng một ảnh RGB** được gửi tới **cả hai** model (same-image contract). Không trộn ảnh từ hai dataset.
- Hai đầu ra phải **tương thích về không gian** trước khi hợp nhất — fusion chịu trách nhiệm resample nếu cần.
- Nhánh Segmentation và nhánh Depth **độc lập** (model khác nhau, dataset khác nhau, metrics khác nhau); chúng **chỉ hợp nhất ở Fusion** trên cùng một ảnh đầu vào.
- **Dataset-level evaluation** (mỗi model trên dataset riêng của nó) khác với **full-pipeline inference** (chạy trên ảnh bất kỳ, không cần ground truth) — hai khái niệm không được trộn lẫn.
- Evaluation và Visualization chạy song song sau khi cả hai nhánh ra đầu ra; Scene Analyzer là consumer hạ nguồn của Fusion.

Luồng phát triển theo phase (spec §21):

| Phase | Việc | Kiểm chứng |
|---|---|---|
| 1 | Config + utils (root-path fix, YAML, seed, logger) | test_config |
| 2 | Data (Cityscapes loader fix, KITTI loader, splits) | test_cityscapes, test_kitti |
| 3 | Segmentation (train.py, checkpoint/resume, mIoU) | test_unet, test_metrics |
| 4 | Depth (MiDaS loader/inference + metrics + alignment) | test_midas, test_metrics |
| 5 | Fusion + analyzer | test_fusion |
| 6 | Visualization (overlay, montage) | manual + smoke |
| 7 | Pipeline (`main.py` CLI, pipeline_metrics, difficulty) | test_pipeline |
| 8 | CI/CD + docs | CI green |

### 5.4 Unit Tests

Spec yêu cầu bộ test pytest với 10 mục (dataset loading, preprocessing/alignment, U-Net forward, U-Net inference, MiDaS inference, segmentation metrics, depth metrics, fusion, scene analyzer, end-to-end pipeline) — mỗi test xác định input / expected / validation, CPU-safe (image size nhỏ, `device=cpu`). Các file `tests/test_kitti.py`, `test_metrics.py`, `test_midas.py`, `test_unet.py`, `test_fusion.py` đã tồn tại; file mới `tests/test_pipeline.py` được đề xuất cho test end-to-end.

### 5.5 CI/CD

Coursework-level, GitHub Actions, **không có cloud infra**:

- **Tự động (push + PR):** `pytest -m "not gpu"` trên CPU runner; import/smoke (build U-Net `base_channels=4`, chạy 1 forward CPU; MiDaS smoke bỏ qua nếu không có weights — đánh dấu `slow`/`network`); lint (`ruff`/`flake8`, tùy chọn) + check circular import.
- **Thủ công (GPU):** train Cityscapes đầy đủ; eval KITTI/MiDaS; eval end-to-end; difficulty sweep — qua `workflow_dispatch`, nếu không có GPU runner thì chạy local và upload metrics report.
- Unit/CPU test chạy trên GitHub-hosted runners; model evaluation trên GPU (manual) hoặc local; artifact JSON + figure commit vào `outputs/analysis/`.
- Train/eval **không** nằm trong PR CI (quá đắt). Gate trước merge: mọi automated test xanh + metrics report mới nhất.

---

## 6. PROJECT STRUCTURE

Cấu trúc repo hiện tại (chỉ các thư mục/file thực tế quan trọng):

```text
cv-project/
├── configs/               # unet.yaml, midas.yaml, pipeline.yaml
├── data/                  # cityscapes/, kitti/, pipeline/images/
├── models/
│   ├── unet/              # model.py, inference.py
│   └── midas/             # model.py, inference.py
├── preprocessing/         # cityscapes.py, kitti.py, transforms.py
├── scene_understanding/   # fusion.py, analyzer.py, pipeline.py
├── visualization/         # segmentation.py, depth.py, scene.py, io.py
├── evaluation/            # evaluate_*.py, metrics, difficulty_analysis.py
├── training/              # U-Net training (trainer.py, train_unet.py)
├── utils/                 # config.py, device.py, seed.py, logger.py
├── tests/                 # pytest suite
├── notebooks/             # exploration / analysis
├── outputs/               # analysis/, segmentation/, depth/, custom/, visualization/
├── checkpoints/           # unet_cityscapes.pth, dpt_large_384.pt
├── main.py                # demo entry point
└── requirements.txt
```

Trách nhiệm từng thư mục (spec §9):

| Thư mục | Owns (sở hữu) | MUST NOT own (không được) |
|---|---|---|
| `data/` | raw datasets (git-ignored) | logic |
| `preprocessing/` | loaders, transforms, splits, class maps | model inference |
| `models/` | model definitions, inference | visualization, evaluation UI |
| `scene_understanding/` | fusion + scene analysis (rule-based) | training logic |
| `evaluation/` | metrics, difficulty, pipeline KPIs | visualization |
| `visualization/` | saving plots/overlays/montages | evaluation math |
| `configs/` | YAML settings | code |
| `utils/` | config/seed/logger plumbing | domain logic |
| `outputs/` | generated artifacts | tracked source |
| `tests/` | pytest suite mirroring module layout | production code |
| `notebooks/` | exploration/analysis (documentation) | production imports |

**Dependency rule (spec §10):**

```text
utils/  (config · seed · logger)
   ↓
preprocessing/
   ↓
models/
   ↓
scene_understanding/
   ↓
evaluation/ · visualization/
   ↓
main.py  (composition root)
```

- `utils` dependency-free → không có cycle.
- `scene_understanding` chỉ dùng **output** của `models` (thin interface), không dùng code nội bộ model.
- **Forbidden imports (lint-enforced):** `models/* → visualization|scene_understanding/*`; `visualization/* → evaluation/*`; `preprocessing/* → models/*`; `utils/* → anything ngoài stdlib/configs`.

---

## 7. SCOPE

### 7.1 In Scope

Các khả năng nằm trong kiến trúc:

- **Semantic Segmentation** — U-Net huấn luyện trên Cityscapes.
- **Monocular Depth Estimation** — MiDaS pretrained, relative depth.
- **Rule-based Fusion** — kết hợp segmentation + depth, interpretable, không black-box.
- **Scene / Context Analysis** — báo cáo traffic context.
- **Task Evaluation** — model-level (U-Net/Cityscapes, MiDaS/KITTI) + pipeline-level.
- **Difficulty Analysis** — Easy/Medium/Hard theo luật.
- **Visualization** — mask, overlay, depth, montage, pred-vs-GT.
- **Config-driven orchestration** — `configs/*.yaml` + `main.py` CLI.
- **CI/CD mức coursework** — GitHub Actions, CPU test tự động, GPU eval thủ công.

### 7.2 Out of Scope (enforced)

Các khả năng **cố tình không** nằm trong kiến trúc, để giữ ranh giới dự án rõ ràng:

| Mục | Trạng thái |
|---|---|
| Object Detection | Không dự phóng bounding box |
| Object Tracking | Không theo dõi theo thời gian |
| Lane Detection | Không phát hiện làn đường |
| Instance Segmentation | Chỉ cấp semantic segmentation |
| 3D Object Detection | Không có độ sâu 3D thật |
| LiDAR–camera fusion | Depth chỉ từ monocular |
| Drivable-control / autonomous-driving actuation | Chỉ phân tích, không điều khiển |
| LLM / Chatbot | Không tích hợp ngôn ngữ lớn |
| Cloud deployment | Không triển khai cloud |
| Learned neural fusion network | Fusion là rule-based |
| “AI Agent” integration | Không dùng khái niệm agent; tích hợp gọi là **Fusion Module** / **Scene Understanding Module** |

---

## 8. ARCHITECTURE SUMMARY

Tóm tắt kiến trúc từ A đến Z, có thể nói mạch lạc khi trình bày:

> “Dự án nhận một RGB image. Ảnh này được đưa **song song** vào U-Net và MiDaS — cùng một ảnh, không trộn dataset.

- U-Net (encoder–bottleneck–decoder với skip connections) trả về **semantic segmentation mask** theo 19 lớp Cityscapes (trainId 0–18, ignore 255).
- MiDaS (DPT-Large pretrained qua torch.hub) trả về bản đồ **relative inverse depth** — giá trị lớn hơn nghĩa là gần hơn, **không phải mét**.

Hai bản đồ được căn chỉnh không gian rồi đưa vào **Fusion Module** — kết hợp **rule-based/analytical**, không phải mạng học — để tính thống kê depth per-class, per-object (dynamic connected components), vùng drivable và layout cảnh.

**Scene Understanding Module** sinh báo cáo bối cảnh giao thông (near/far objects, road context, traffic density), và **Difficulty Analysis** gán mức Easy/Medium/Hard bằng heuristic.

Toàn bộ cài đặt nằm trong `configs/*.yaml`, mọi module theo dependency một chiều, `main.py` là điểm điều phối duy nhất.”

Điểm mấu chốt của kiến trúc: **hai nhánh (segmentation, depth) độc lập về model/dataset/metrics, chỉ hợp nhất tại Fusion trên cùng một ảnh đầu vào; Cityscapes và KITTI là hai dataset riêng biệt không ghép cặp; fusion cố ý interpretable và rule-based để phù hợp quy mô coursework.**

---

*Kết thúc Step 01: Architecture Specification. Các bước triển khai tiếp theo (data, model, fusion, evaluation…) được thực hiện theo thứ tự phase trong mục 5.3. Không có kết quả thử nghiệm nào ở bước này.*