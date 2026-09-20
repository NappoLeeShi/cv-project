# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

## Coursework Report — Step 03: Dataset Loaders and Preprocessing

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Tài liệu này mô tả tầng **Dataset Loaders và Preprocessing** cho hai dataset **Cityscapes** và **KITTI**, dựa trên `prompts/03_dataset.md`, `prompts/01_architecture_specification.md` và code/repository thực tế (`preprocessing/`, `tests/`, `configs/`, `data/`, `README.md`).
> Bước này **chỉ** dựng tầng dữ liệu: không có U-Net, MiDaS, training, fusion, evaluation metrics hay visualization nào ở đây.

---

## 1. REQUIREMENT

### 1.1 Problem

Dự án là hệ thống **Segmentation + Depth → Scene Understanding** với hai nhánh:

- **Cityscapes + U-Net → Semantic Segmentation**
- **KITTI → Depth evaluation** (MiDaS sẽ được đánh giá trên depth GT của KITTI)

Sau Step 02 (config + utils), Step 03 phải dựng **tầng dataset và preprocessing** — nơi quyết định chất lượng dữ liệu vào của toàn bộ hệ thống. Yêu cầu cốt lõi:

- Loader thiết kế cho **dataset thật** ngay từ đầu (không tạo "dataset mini" riêng);
- Cùng một loader phục vụ **subset nhỏ** (development/test) lẫn **full dataset** (experiment), điều khiển bằng `max_samples`;
- Ghép cặp image ↔ label / image ↔ depth **theo stem tên file** (không sort hai list độc lập), có validate;
- Cityscapes chuyển `labelIds` → `trainId` (0..18, 255 = ignore) **theo bảng chính thức**;
- KITTI giữ đúng bản chất depth: xử lý invalid pixel, không biến depth thành "label";
- **Không ghép cặp Cityscapes với KITTI** — hai dataset độc lập hoàn toàn.

### 1.2 Input / Output

| Hướng | Mô tả |
|---|---|
| **Input** | Cấu hình từ Step 02 (`configs/pipeline.yaml`, block `data.cityscapes` / `data.kitti`) + ảnh PNG gốc trong `data/` |
| **Output** | Hai PyTorch `Dataset` chuyển raw files thành sample chuẩn hóa: |
| — Cityscapes | `{image, label, image_path, label_path}` với `image` tensor `(3, H, W)` float, `label` tensor `(H, W)` long (trainId/255) |
| — KITTI | `{image, depth, valid_mask, image_path, depth_path}` với `depth` tensor `(H, W)` float (mét), `valid_mask` boolean |
| **Kiểm chứng** | `DatasetError` báo lỗi rõ ràng cho cấu trúc/sample lỗi; unit tests chạy trên fixture tạm thời |

### 1.3 Scope

**Trong phạm vi:**

1. Cityscapes dataset loader (`preprocessing/cityscapes.py`);
2. KITTI dataset loader (`preprocessing/kitti.py`);
3. Dataset preprocessing (`preprocessing/transforms.py`) và errors (`preprocessing/errors.py`);
4. Tích hợp dataset configuration (dùng lại block `data.*` của Step 02 qua `from_config`);
5. Dataset validation;
6. Dataset-related tests (`tests/test_cityscapes.py`, `tests/test_kitti.py`).

**Ngoài phạm vi (cấm ở Step 03):** U-Net, MiDaS, training, fusion, scene analyzer, evaluation metrics, visualization, full pipeline, model inference. **Không tự động download dataset.** Không thêm framework dữ liệu ngoại lai.

---

## 2. PURPOSE & SURVEY

### 2.1 Purpose

Vì sao cần một tầng dataset đúng đắn:

- **Sample chuẩn hóa thống nhất** — mọi component phía sau (models, evaluation, fusion) nhận cùng một định dạng dữ liệu, không phải tự lo phần đọc file/labels/depth.
- **Ghép cặp chính xác** — pair theo stem tên file + validate: nếu lệch sẽ báo rõ, không sinh cặp sai âm thầm (sai một cặp là hỏng metrics).
- **Tách "nỗi lo dữ liệu" khỏi model** — dataset lo phần raw bitmap → tensor; model không biết file nằm ở đâu (đúng dependency rule: `preprocessing` chỉ phụ thuộc `utils`).
- **Một loader cho hai chế độ** — `max_samples: 10` lúc dev, `null` lúc full; đúng kiến trúc đã định ở Step 01.

### 2.2 Technology Survey

| Công nghệ | Vai trò | Vì sao phù hợp |
|---|---|---|
| **PyTorch `Dataset`/`DataLoader`** | Interface chuẩn cho cả hai loader | Kết nối trực tiếp với U-Net/MiDaS sau này; hỗ trợ `__len__`/`__getitem__` |
| **Pillow (`PIL.Image`)** | Đọc PNG (RGB, 8-bit label, 16-bit depth) | Không introduce framework mới; đã có trong stack Step 01 |
| **NumPy** | Chuyển ảnh → array, lookup label, mask | Thao tác mask/scale nhanh, trả về tensor qua `torch.as_tensor` |
| **PyTorch tensors** | Đầu ra chuẩn hóa | Đúng khớp input của PyTorch model |
| **PyYAML (Step 02)** | Đọc block `data.*` từ config | Không duplicate logic config |

Không thêm bất kỳ "dataset framework" nào (`torchvision.datasets`, `datasets`, …) — loader tự viết để kiểm soát layout và lỗi theo yêu cầu dự án.

### 2.3 Capability Comparison

**Cách ghép cặp image–label / image–depth:**

| Cách | Ưu | Nhược | Quyết định |
|---|---|---|---|
| Sort hai list độc lập rồi zip | Đơn giản | Nếu một list thiếu 1 file ⇒ toàn bộ cặp từ đó lệch sang | ❌ (bị cấm bởi spec) |
| Match theo **stem tên file** + validate | Đúng tuyệt đối, không lệch | Cần quy ước tên file rõ ràng | ✅ **chọn** — Cityscapes khớp `*_leftImg8bit.png` ↔ `*_gtFine_labelIds.png`; KITTI khớp cùng stem |

**Xử lý sample thiếu:**

| Chính sách | Hành vi | Dùng khi nào |
|---|---|---|
| `missing_policy: error` | Raise `DatasetError` liệt kê file thiếu | Mặc định — không cho phép lỗi âm thầm |
| `missing_policy: skip` | Bỏ sample thiếu, cảnh báo qua log | Khi dữ liệu chưa đủ |
| `require_labels: false` | Label không bắt buộc (trả `None`) | Khi chỉ cần ảnh (vd thu thập ảnh bất kỳ) |

**Tham số phát hiện sai → lỗi ngay** (không để `IndexError` rơi xuống): `max_samples` sai kiểu/âm → `DatasetError`; `scale_mm ≤ 0` → `DatasetError`; label path trùng stem (ambiguity) → `DatasetError`.

### 2.4 Why These Datasets?

Hai nhánh của dự án cần **hai nguồn dữ liệu đúng chuyên môn**:

| Nhánh | Dataset | Vì sao phù hợp |
|---|---|---|
| **Semantic Segmentation** | **Cityscapes** | Dataset chuẩn hàng đầu cho driving-scene segmentation: ảnh đường phố 1024×2048, 30+ class, chuẩn hoá **19 trainable class** (trainId) + ignore; có train/val/test chính thức. U-Net học dự đoán class từng pixel. |
| **Monocular Depth Evaluation** | **KITTI** | Dataset chuẩn cho depth từ camera: ảnh RGB + **depth ground truth PNG 16-bit** (đơn vị mm, cấp-pixel, thưa do occlusion). MiDaS output là *relative inverse depth* nên **chỉ đánh giá** được bằng cách căn chỉnh (align) với depth GT metric của KITTI — không huấn luyện MiDaS trên đó. |

**Quan trọng:** Cityscapes và KITTI là hai dataset **khác nguồn, không tương ứng, không ghép cặp**. Chúng chỉ *gặp nhau* ở **Fusion** khi pipeline chạy trên một ảnh RGB bất kỳ (cả hai model nhận cùng ảnh đó) — nhưng việc đó thuộc các bước sau, không phải Step 03.

---

## 3. DATA

### 3.1 Cityscapes

- **Mục đích:** huấn luyện/đánh giá U-Net semantic segmentation (19 class).
- **RGB images:** ảnh đường phố, suffix `_leftImg8bit.png`, đọc qua Pillow `convert("RGB")`.
- **Semantic labels:** PNG `_gtFine_labelIds.png` chứa **raw `labelIds` (0..33)** từ chính thức bộ devkit; loader dùng mapping `CITYSCAPES_LABEL_ID_TO_TRAIN_ID` (trích từ Cityscapes `labels.py`) để chuyển sang trainId.
- **trainId mapping (19 class, khớp bảng chính thức):**

| labelId | trainId | Class | labelId | trainId | Class |
|---|---|---|---|---|---|
| 7 | 0 | road | 24 | 11 | person |
| 8 | 1 | sidewalk | 25 | 12 | rider |
| 11 | 2 | building | 26 | 13 | car |
| 12 | 3 | wall | 27 | 14 | truck |
| 13 | 4 | fence | 28 | 15 | bus |
| 17 | 5 | pole | 31 | 16 | train |
| 19 | 6 | traffic light | 32 | 17 | motorcycle |
| 20 | 7 | traffic sign | 33 | 18 | bicycle |
| 21 | 8 | vegetation | — | **255** | ignore/void |
| 22 | 9 | terrain | | | |
| 23 | 10 | sky | | | |

Mapping được triển khai qua lookup table **256 phần tử** (`_TRAIN_ID_LOOKUP`) — mọi `labelId` ngoài 19 class trên và các class void đều thành **255 (ignore)**, không bị "rà ngẫu nhiên".

- **Repository location:** `data/cityscapes/` với `images/leftImg8bit/` và `labels/gtFine/` (đúng layout chính thức; hiện **chưa có file ảnh** — dataset chưa được download). Loader hỗ trợ cả layout chính thức (`<root>/<images>/<split>/...`) lẫn layout phẳng (`images/`, `labels/`).
- **Cách U-Net dùng:** loader trả `{image, label, ...}` — `label` đã là trainId tensor (0..18 / 255), sẵn sàng làm target của loss segmentation.

### 3.2 KITTI

- **Mục đích:** đánh giá monocular depth (MiDaS) — **không** dùng để huấn luyện.
- **RGB images:** `data/kitti/images/`, suffix `.png`, đọc qua Pillow.
- **Depth ground truth:** `data/kitti/depth/`, PNG đơn kênh **16-bit**, giá trị **millimetre**; loader chuyển sang **mét** bằng `value / scale_mm` (mặc định 1000).
- **Valid/invalid depth:** `0` = invalid (occlusion/thưa); loader giữ riêng **`valid_mask = depth > 0`**; pixel invalid giữ giá trị `0.0` và bị nhấc ra khỏi mask — **không biến depth thành label**, không tính chiều bừa:
  - `depth_cap_m` (vd 80 m): pixel vượt xa hơn ngưỡng cũng bị đánh dấu invalid;
  - khi resize depth (bilinear), **`valid_mask` được tính lại từ chính depth đã resize** → interpolation không "lé" giá trị vào pixel invalid.
- **Repository location:** `data/kitti/images/` và `data/kitti/depth/` — repo hiện có **1.000 ảnh + 1.000 depth** thật (tên dạng Eigen-style, ảnh ~1216×352, depth PNG 16-bit; trong một mẫu thử: ~76.8% pixel invalid — đúng đặc tính depth thưa).
- **Cách MiDaS dùng sau này:** loader cung cấp depth GT metric + `valid_mask`; phần alignment (vì MiDaS output là *relative depth*) thuộc bước đánh giá depth.

### 3.3 Data Split

- **Cityscapes:** cấu hình `split` (`train` / `val` / `test`) chọn thư mục con trong layout chính thức; split linh hoạt bằng tham số. Không giả định **test labels** luôn có (test split có thể không có ground truth cho supervised eval).
- **KITTI:** hai cơ chế:
  1. **Split file:** `data/kitti/splits/<split>.txt` (mỗi dòng một stem, hỗ trợ stem trần/đường dẫn tương đối/`scene/frame`, bỏ qua `#` và dòng trống, giữ thứ tự trong file);
  2. **Tự động:** nếu không có split file, dùng **giao của hai stem-map** (`images` ∩ `depth`), sắp xếp deterministic.
  - Repo hiện **chưa có** file `splits/*.txt` → hạnh vi mặc định (giao các stem hiện có) áp dụng.
- **`max_samples`:** số sample tối đa, áp dụng **trên danh sách đã sắp xếp deterministic** → cùng subset giữa các lần chạy. `null` = full, `0` = cho phép loader rỗng.

> Split đúng chuẩn (Eigen train/val cho KITTI, hay train/val/test Cityscapes) có thể được khai báo qua config/file split; Step 03 cung cấp cơ chế, không tự bịa danh sách split nào.

### 3.4 Data Format

| Loại | Format file | Tensor đầu ra |
|---|---|---|
| RGB image (Cityscapes/KITTI) | PNG 8-bit RGB | `(3, H, W)` float32, `/255`, chuẩn hóa ImageNet (mean 0.485/0.456/0.406, std 0.229/0.224/0.225) |
| Cityscapes label | PNG 8-bit grayscale (raw `labelIds`) | `(H, W)` int64 → long tensor, **trainId 0..18 / 255** |
| KITTI depth | PNG 16-bit grayscale (`I;16`), **mm** | `(H, W)` float32 **mét** (+) `valid_mask` bool `(H, W)` |

Quy ước `image_size` là `[H, W]` trong config; code chuyển sang `(W, H)` cho PIL `resize` (`hw_to_pil_size`).

### 3.5 PNG vs JPG

Trong tầng dữ liệu Step 03, **toàn bộ đầu vào là PNG** (loader chỉ quét `*.png`):

- **Cityscapes labels** (`_gtFine_labelIds.png`) **bắt buộc lossless**: giá trị là *index* (labelId); nén JPG làm nhiễu/mất giá trị → alias class sai.
- **KITTI depth** PNG **16-bit lossless**: giữ nguyên millimetre chính xác;
- **RGB images** của cả hai dataset dùng PNG (chuẩn gốc của dataset). Nén lossy (JPG) cho ảnh colors cũng được — chỉ là dataset không cung cấp sẵn ở dạng đó.

Vì vậy với **label và depth**, JPEG là không chấp nhận được; PNG bảo toàn **từng value nguyên** đúng nghĩa đen. (Ảnh RGB *tùy ý* của người dùng có thể là `.jpg/.jpeg/.bmp/.webp/.tiff` trong demo full-pipeline — nhưng đó thuộc các bước sau, không phải tầng dataset này.)

### 3.6 Data Visualization

**Trạng thái thật ở Step 03: chưa có output visualization nào của bước này.**

Lý do nêu rõ: spec Step 03 liệt kê *Visualization* vào danh sách **"Do NOT implement"** — Step 03 chỉ dựng loader/preprocessing. Trong repo:

- **`visualization/`** (segmentation.py, depth.py, scene.py, fusion.py, io.py) là module code **có thật** nhưng thuộc các bước sau (Step 10+), không phải sản phẩm của Step 03;
- Các ảnh lưu dưới `outputs/` (`pipeline_demo_*.png` trong `outputs/{segmentation,depth,analysis}`, `outputs/custom/` như `gta5_*`, `streetest_*`) là output **demo full-pipeline của các bước sau** — **không** được trưng vào đây như kết quả Step 03;
- **Năng lực "nhìn" dữ liệu có thể có:** loader trả sample chuẩn hóa (`image`, `label`/`depth`, `valid_mask`) — về lý thuyết có thể denormalize để xem ảnh, nhưng **không có script nào của Step 03 sinh figure**.

> Kết luận: *visualization functionality/output not available at this step.* Nếu cần ảnh minh họa dataset, chúng sẽ đến từ các bước visualization/pipeline sau; Step 03 chỉ bảo đảm dữ liệu chuẩn hóa đúng để các bước đó sử dụng.

---

## 4. IMPLEMENTATION PLAN

### 4.1 Dataset Loading

**CityscapesDataset** (`preprocessing/cityscapes.py`):

- Tham số: `root`, `split` (`train`/`val`/`test`/`None`=flat), `image_dir`/`label_dir` (hoặc auto-detect `leftImg8bit`/`images`, `gtFine`/`labels`), `image_size`, `max_samples`, `require_labels`, `missing_policy`, `label_suffix`.
- Quy trình khởi tạo: resolve dirs → scan label map (rglob `*.png`, lọc suffix hợp lệ, phát hiện **ambiguity**) → build samples (rglob ảnh `*_leftImg8bit.png`, match label theo stem) → validate/apply `max_samples`.
- Trả `{image, label, image_path, label_path}`; `label=None`/`label_path=None` khi không yêu cầu label.

**KittiDepthDataset** (`preprocessing/kitti.py`):

- Tham số: `root`, `split`, `image_dir`/`depth_dir`, `image_size`, `scale_mm` (1000), `depth_cap_m`, `max_samples`, `missing_policy`, `stem_key`.
- Quy trình: build stem-map cho images và depth (rglob `*.png`, stem = relative path bỏ suffix) → chọn theo split file hoặc giao hai map → validate → `max_samples`.
- Trả `{image, depth, valid_mask, image_path, depth_path}`.

**Hook `stem_key`:** cho phép transform stem để khớp quy ước tên khác nhau giữa images và depth (dùng khi tên file thật không trùng trực tiếp — xem 8.3).

### 4.2 Preprocessing

`preprocessing/transforms.py` — chia nhỏ, tách rời việc nặng (remap label, scale depth) khỏi tensor conversion:

| Transform | Việc | Interpolation |
|---|---|---|
| `ImageTransform` | resize → `/255` → tensor `(3,H,W)` → ImageNet normalize (tùy chọn) | **BILINEAR** |
| `resize_nearest` (label) | resize label giữ nguyên giá trị class | **NEAREST** |
| `DepthTransform` | resize depth → tính lại `valid = depth > 0` → tensor mét + mask | **BILINEAR** (depth là giá trị liên tục) |
| `label_to_tensor` / `image_to_tensor` | chuyển array → `torch.long` / float `(3,H,W)` | — |

### 4.3 Label Processing

1. Đọc label PNG → grayscale array (int64);
2. `labels_to_train_ids()`: tra lookup table 256 phần tử → **labelIds → trainId (0..18 / 255)**;
3. Resize label bằng **nearest-neighbour** (không bao giờ bilinear — bilinear sẽ tạo ra giá trị class *phân số* hợp lệ → vô nghĩa/giả class);
4. `label_to_tensor` → long tensor `(H, W)`.

Bảng mapping duy trì trong `CITYSCAPES_LABEL_ID_TO_TRAIN_ID` (19 mục nhập, trích từ bảng chính thức) — đã dùng chung cho dataset và được test đầy đủ (gồm `labelIds → trainId` của road/car/bicycle và label `void` → 255).

### 4.4 Depth Processing

1. Đọc depth PNG (`I;16`) giữ **int64** (không mất bit) → float32;
2. Chia `scale_mm` (1000) → **mét**;
3. Nếu `depth_cap_m`: pixel `> cap` → đặt `0.0` (invalid);
4. `DepthTransform`: resize **bilinear** (depth là giá trị liên tục — nearest sẽ vỡ bậc); **tính lại `valid_mask = depth > 0`** sau resize → pixel invalid giữ `0.0` và bị bỏ khỏi mask, không bị "nhòe value".

*Vì sao depth resize dùng bilinear nhưng valid mask tính lại:* nhằm bảo toàn depth liên tục; mask được suy ra *từ chính depth đã resize* nên interpolation không thể đưa giá trị vào pixel vốn invalid.

### 4.5 Image/Label Alignment

- Cặp image–label (Cityscapes) và image–depth (KITTI) khớp do **cùng stem tên file** — không phụ thuộc thứ tự trên đĩa (có test riêng cho trường hợp order khác nhau).
- Cùng **một `image_size`** áp dụng cho image và target (xuyên cùng size config `data.image_size`); label/depth dùng interpolation đúng loại dữ liệu → sau transform, image và target **vẫn thẳng pixel chính xác**.
- Với KITTI, `valid_mask` cùng kích thước với depth → mask align đúng toạ độ.

### 4.6 Data Validation

Mọi lỗi cấu trúc/sample dẫn đến **`DatasetError`** mô tả rõ (tránh `IndexError`:

| Kiểm tra | Hành vi |
|---|---|
| Thiếu thư mục image/label/depth | `DatasetError` chỉ rõ đường dẫn + cấu trúc mong đợi |
| Không tìm thấy cặp nào | `No Cityscapes/KITTI depth pairs found ...` |
| Ảnh thiếu label/depth | `error` (liệt kê 5 file đầu) hoặc `skip` (cảnh báo log) |
| Stem trùng lặp (ambiguous) | `DatasetError` |
| `max_samples` / `missing_policy` / `scale_mm` sai | `DatasetError` ngay lúc khởi tạo |
| Split file trỏ entry thiếu file | `error` hoặc `skip` như trên |

### 4.7 Visualization Pipeline

**Không nằm trong Step 03** (xem 3.6). Module `visualization/` và đầu ra `outputs/*.png` là của các bước sau; Step 03 dừng ở việc cung cấp dữ liệu chuẩn hóa.

---

## 5. SYSTEM BUILD FLOW

### 5.1 Cityscapes Data Flow

```text
data/cityscapes/  (images/leftImg8bit ↔ labels/gtFine)
  → resolve dirs (official split layout | flat) + split
  → scan labels (rglob *_gtFine_labelIds.png, theo label_suffix)
  → scan images (rglob *_leftImg8bit.png, sorted → deterministic)
  → match label theo STEM (không sort-zip)
  → validate: thiếu label → error | skip
  → max_samples (deterministic)
  → label: resize NEAREST → labelIds → trainId (0..18 / 255)
  → image: resize BILINEAR → (3,H,W) float → ImageNet normalize
  → {image, label, image_path, label_path}  → training / evaluation input
```

### 5.2 KITTI Data Flow

```text
data/kitti/  (images/ ↔ depth/)
  → build stem-map images & depth (rglob *.png, sorted)
  → chọn: splits/<split>.txt (nếu có) | giao hai stem-map
  → validate: entry thiếu → error | skip
  → max_samples (deterministic)
  → depth: PNG int64 mm → /scale_mm → cap → resize BILINEAR → valid=depth>0
  → image: resize BILINEAR → (3,H,W) → normalize
  → {image, depth, valid_mask, image_path, depth_path}  → depth evaluation input
```

### 5.3 Preprocessing Flow

```text
RGB PNG        → PIL RGB     → resize BILINEAR → (3,H,W)/255 → normalize → float tensor
Label PNG 8bit → grayscale   → resize NEAREST → lookup[256] → trainId/255 → long tensor
Depth PNG 16bit→ int64(mm)   → /scale_mm → cap → resize BILINEAR → valid=depth>0 → float+mask
```

### 5.4 Visualization Flow

Không có flow riêng ở Step 03 — visualization chính thức đến từ các bước sau.

### 5.5 Unit Tests

Chạy bằng pytest với **fixture tổng hợp (synthetic)** trong thư mục tạm — **không cần** dataset thật:

| File test | Số test | Kiểm tra |
|---|---|---|
| `tests/test_cityscapes.py` | 17 | Mapping đầy đủ & đúng (road/car/bicycle, void→255); remap labelIds→trainId; load sample + trainId; `max_samples` (truncate / 0 / giá trị sai); thiếu label → error/skip; label tùy chọn; **pair theo stem không theo order**; layout split chính thức (train/val); bỏ qua file khác suffix (`instanceIds`); không ảnh → error; normalize ImageNet; ambiguity → error; `from_config` + override |
| `tests/test_kitti.py` | 18 | mm→mét; `0` = invalid (+mask); `depth_cap_m`; **pair theo stem không theo order**; thiếu depth → error/skip; thư mục con lồng; split file (thứ tự, dòng comment, entry dạng relative path); entry thiếu → error/skip; `max_samples`; no pairs/empty images → error; **resize depth tái tính valid mask**; `scale_mm` cấu hình được; `from_config` |

Chạy thực tế (gồm cả 36 test Step 02 — để chắc không phá vỡ gì):

```text
.venv/bin/python -m pytest tests/test_cityscapes.py tests/test_kitti.py \
    tests/test_config.py tests/test_seed.py tests/test_logger.py tests/test_device.py -q
71 passed in 3.44s
```

Import check theo yêu cầu spec:

```bash
.venv/bin/python -c "from preprocessing.cityscapes import *; from preprocessing.kitti import *; print('Dataset modules OK')"
# → Dataset modules OK
```

### 5.6 CI/CD

Chưa nằm trong Step 03 (chưa thiết lập GitHub Actions) — "CI/CD" hiện tại là **chạy pytest cục bộ** như trên để xác nhận tầng dữ liệu đúng.

---

## 6. EVALUATION

### 6.1 Data Validation

Thiết kế đã liệt kê ở mục 4.6: loader tự kiểm tra root/dirs/files/matching ngay khi khởi tạo, báo `DatasetError` có thông tin. Đây là "evaluation" cấp dữ liệu duy nhất của Step 03.

### 6.2 Dataset-level Checks

- Unit tests cho toàn bộ hành vi loader (17 + 18) đều xanh;
- Không có dataset thật nào bị thay đổi; không download gì tự động;
- Với KITTI repo (1.000 cặp), loader nhận diện đúng cấu trúc; với cấu trúc tên Eigen-style hiện tại cần hook `stem_key` để khớp (xem 8.3).

### 6.3 Visualization-based Inspection

**Không khả dụng ở Step 03** — chưa có ảnh dataset nào được sinh ra ở bước này; khâu inspection bằng hình ảnh sẽ nằm trong các bước visualization sau.

### 6.4 Metrics

Step 03 **không định nghĩa hay chạy bất kỳ metric mô hình nào** (không mIoU, không AbsRel…). Các số liệu ở đây chỉ là số test (71 passed). **Model-level numerical evaluation is covered in later steps** (đánh giá U-Net/Cityscapes và MiDaS/KITTI ở các bước sau).

---

## 7. AI-ASSISTED DEVELOPMENT

### 7.1 Prompt

Nhóm thực hiện Step 03 theo brief `prompts/03_dataset.md` (trong repo), kế thừa kết quả Step 02 và tuân thủ kiến trúc `prompts/01_architecture_specification.md`. Prompt giới hạn nghiêm ngặt: chỉ loaders + preprocessing + validation + tests; cấm models/training/fusion/evaluation/visualization/pipeline và cấm download dataset.

### 7.2 Generated / Modified Scripts

| Loại | File | Trạng thái |
|---|---|---|
| Loader | `preprocessing/cityscapes.py`, `preprocessing/kitti.py` | Tạo |
| Preprocessing | `preprocessing/transforms.py` | Tạo |
| Errors | `preprocessing/errors.py` (`DatasetError`) | Tạo |
| Tests | `tests/test_cityscapes.py`, `tests/test_kitti.py` | Tạo |
| Config | Block `data.cityscapes` / `data.kitti` trong `configs/pipeline.yaml` (đã có từ Step 02) | Dùng lại qua `from_config` — không duplicate |
| Docs | Layout dataset + dev-mode ghi trong `README.md` | Tài liệu |

### 7.3 Testing

- 35 test mới (17 Cityscapes + 18 KITTI) xanh trên fixture synthetic;
- 36 test Step 02 vẫn xanh (tổng **71 passed in 3.44s**);
- Import check cả hai module dataset OK; không cần internet/dataset đầy đủ.

### 7.4 Version History

Quản lý bằng Git trên repo (`git log`); lịch sử commit tách riêng cho từng bước coursework không được ghi chi tiết ở đây — ghi nhận là **chưa sẵn có** ở mức bước.

### 7.5 Output

Toàn bộ tầng dataset/preprocessing (mục 3–5) kèm tài liệu này (`docs/coursework/03_data_and_preprocessing.md`).

---

## 8. DISCUSSION

### 8.1 Strengths

- **Ghép cặp an toàn:** pair theo stem + validate mọi sample → không thể sinh cặp sai âm thầm; có test riêng cho trường hợp thứ tự đĩa khác nhau.
- **Mapping Cityscapes chính thức:** `labelIds → trainId` theo bảng devkit, lookup 256 phần tử; mọi class ngoài 19/void → 255; test xác minh từng giá trị.
- **Interpolation đúng loại dữ liệu:** label luôn nearest (giữ nguyên ID class); depth dùng bilinear nhưng **valid mask tái tính** để pixel invalid không bị "nhòe".
- **Một loader cho subset & full:** `max_samples` chỉ cắt danh sách deterministic, không tạo class riêng nào.
- **Dependency chuẩn:** `preprocessing` chỉ import `utils` (config/get) — không import model/eval/fusion (đúng dependency rule Step 01).
- **Lỗi rõ ràng:** mọi vấn đề cấu trúc dữ liệu báo `DatasetError` kèm đường dẫn, không để `IndexError` mơ hồ rơi xuống.

### 8.2 Weaknesses

- **Hai quy ước `image_size` có thể phân kỳ:** `data.image_size` (dùng cho các bước sau) và kích thước file thật; cần giữ thống nhất qua config.
- **Bilinear trên depth là đánh đổi:** mượt hơn nearest nhưng làm trơn biên đối tượng; mask tái tính giúp an toàn nhưng độ phân giải biên giảm.
- **Cityscapes chưa có file thật trong repo** → layout chính thức chỉ được kiểm chứng bằng cấu trúc thư mục + fixture, chưa chạy trên dữ liệu 1024×2048 thật.

### 8.3 Limitations

- **Class imbalance Cityscapes:** `road`/`sidewalk` chiếm phần lớn pixel; một số class (pole, bicycle) rất hiếm — loader cung cấp dữ liệu nhưng cân bằng class thuộc phần huấn luyện (Step 12), không thuộc Step 03.
- **KITTI depth thưa (sparse):** ~ 76% pixel invalid trong một mẫu kiểm tra — chỉ đánh giá được trên vùng valid; mask áp dụng đúng nhưng "độ phủ" thấp.
- **Quy ước tên KITTI hiện tại:** tên image (`..._image_NNNN_image_02`) và depth (`..._groundtruth_depth_NNNN_image_02`) **không trùng stem trực tiếp** (overlap mặc định = 0 cho 1.000 cặp). Loader lo được nhờ hook **`stem_key`** (transform stem trước khi match, ví dụ thay `_groundtruth_depth` → `_image` cho overlap đầy đủ); đây là *giả định cấu trúc* cần được cấu hình khi dùng dataset thật (eval depth sau này sử dụng).
- **Khác biệt miền (domain gap):** Cityscapes (ảnh châu Âu, ánh sáng tốt) và KITTI (Đức, sensor khác) khác nhau về nội dung/độ phân giải; hai nhánh không trộn dữ liệu.
- **Resolution khác nhau giữa hai dataset** (Cityscapes 1024×2048 gốc vs KITTI ~1241×376) — mỗi nhánh xử lý theo config riêng; chỉ thống nhất tại fusion.
- **Không có visualization** ở bước này — việc kiểm tra bằng mắt dữ liệu bị trì hoãn đến các bước sau.

---

## 9. CONCLUSION

Step 03 đã xây dựng xong tầng dữ liệu: hai loader PyTorch cho Cityscapes và KITTI, ghép cặp theo stem có validate, chuyển label chuẩn Cityscapes (`labelIds → trainId 0..18 / 255`), xử lý depth KITTI đúng bản chất (mm → mét, `valid_mask`, depth cap), tiền xử lý đúng loại interpolation (label = nearest, depth = bilinear + tái tính mask), cấu hình theo block `data.*` của Step 02, và **35 unit test** xanh trên fixture synthetic không cần download.

Kết quả là từ nay các bước sau (U-Net, MiDaS, evaluation, fusion) vừa nhận **sample chuẩn hóa thống nhất**, vừa được bảo vệ khỏi lỗi ghép cặp — nền tảng dữ liệu đã sẵn sàng cho hai nhánh; Cityscapes và KITTI vẫn là hai dataset **độc lập, không ghép cặp**. Khâu visualization dữ liệu sẽ xuất hiện ở các bước tiếp theo.

---

## 10. REFERENCES

- `prompts/03_dataset.md` — task brief của Step 03 (trong repository).
- `prompts/01_architecture_specification.md` — kiến trúc, dependency rule và cấu trúc config.
- `README.md` — layout dữ liệu, quy ước ghép cặp theo stem, dev mode `max_samples`.
- `docs/coursework/00_project_architecture.md`, `docs/coursework/01_architecture_specification.md`, `docs/coursework/02_config_utils.md` — tài liệu các bước trước.
- Source trực tiếp: `preprocessing/{cityscapes,kitti,transforms,errors}.py`, `tests/test_{cityscapes,kitti}.py`, `configs/pipeline.yaml`.
- Bảng label chính thức Cityscapes được trích theo Cityscapes scripts (`helpers/labels.py`) — đường dẫn tham chiếu trong docstring `preprocessing/cityscapes.py` (github.com/mcordts/cityscapesScripts).