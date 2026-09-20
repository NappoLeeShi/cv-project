# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

## Coursework Report — Step 02: Config + Utils

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Tài liệu này mô tả **tầng cấu hình và tiện ích (Config + Utils)** — khâu nền tảng đầu tiên được triển khai, làm cơ sở cho các bước Dataset, U-Net, MiDaS, Evaluation, Fusion, Pipeline.
> Nguồn sự thật: `prompts/02_config_utils.md`, `prompts/01_architecture_specification.md` và code/repository thực tế (`configs/`, `utils/`, `tests/`, `requirements.txt`).
> Step 02 **chỉ** triển khai hạ tầng; không có dataset, model, training hay kết quả đánh giá nào ở bước này.

---

## 1. REQUIREMENT

### 1.1 Problem

Dự án là hệ thống **Segmentation + Depth → Scene Understanding**. Kiến trúc đã được định nghĩa chi tiết ở Step 01 với yêu cầu xuyên suốt:

- Kết hợp U-Net (semantic segmentation, Cityscapes) và MiDaS (monocular depth, pretrained);
- Fusion rule-based/analytical — không có learned fusion network;
- Cityscapes và KITTI là hai dataset riêng biệt, **không ghép cặp**;
- Toàn pipeline dùng **cùng một ảnh RGB** cho cả hai model; depth là **relative inverse depth**, không phải mét.

Tuy nhiên, trước khi bất kỳ component nào (dataset, model, evaluation, fusion, pipeline) được viết, repository cần một **tầng nền tảng chung**:

- Vị trí lưu mọi cài đặt (thay vì hard-code trong code);
- Cách đọc cấu hình từ file;
- Tiện ích seed (tái lập kết quả), logger (theo dõi run), device (chọn CPU/CUDA).

Step 02 chính là **tầng nền tảng (foundational infrastructure)**: chuẩn bị "mặt bằng" để các bước sau không phải tự lo phần này và thống nhất cách truy cập cấu hình trên toàn dự án.

### 1.2 Input / Output

| Hướng | Mô tả |
|---|---|
| **Input** | Các file YAML trong `configs/` (`unet.yaml`, `midas.yaml`, `pipeline.yaml`) + tham số chọn device/seed khi chạy |
| **Output 1** | Python `dict` hợp lệ từ `utils/config.load_config()` — nguồn cài đặt duy nhất cho mọi module phía sau |
| **Output 2** | Trạng thái RNG đã gieo seed (Python / NumPy / PyTorch / CUDA) |
| **Output 3** | Logger cấu hình sẵn (console + file) cho mọi module |
| **Output 4** | `torch.device` được chọn theo `auto` / `cpu` / `cuda` |

### 1.3 Scope

**Trong phạm vi (`configs/` + `utils/` + test):**

1. Các file YAML configuration (`unet`, `midas`, `pipeline`);
2. Configuration loader (`utils/config.py`);
3. Random seed utility (`utils/seed.py`);
4. Logger utility (`utils/logger.py`);
5. Device utility (`utils/device.py`);
6. Unit tests cho các utility trên (`tests/test_config.py`, `test_seed.py`, `test_logger.py`, `test_device.py`);
7. Bổ sung package cần thiết vào `requirements.txt` (tối thiểu `PyYAML`, `pytest`).

**Ngoài phạm vi (bị cấm ở Step 02):**

- Cityscapes dataset, KITTI dataset, U-Net, MiDaS, training, fusion, scene analyzer, evaluation metrics, visualization, full pipeline;
- **Không download** dataset hay model weights;
- Không redesign lại kiến trúc, không sửa cấu trúc thư mục không cần thiết.

Yêu cầu kiểm chứng đầu cuối: dự án phải chạy/test được mà **không cần dataset hay model**.

---

## 2. PURPOSE & SURVEY

### 2.1 Purpose

Vì sao cần một tầng config + utils riêng:

- **Config-driven (theo Step 01):** mọi tham số (resolution, epochs, learning rate, device, seed, ngưỡng fusion…) nằm trong `configs/*.yaml`; code không hard-code. Điều này giúp chạy thử nghiệm lại với cấu hình khác mà không cần sửa code.
- **Single source of truth:** có một nơi duy nhất đọc cấu hình (`utils/config.py`), giảm sai lệch giữa các module.
- **Reproducibility:** seed thống nhất để kết quả lặp lại được; logger + config fingerprint để biết một run chạy với cấu hình nào.
- **Separation of concerns:** phần "dẫn đường" (config/seed/logger/device) tách khỏi logic nghiệp vụ (model, dataset, fusion…) — đúng dependency rule của Step 01: `utils/` là tầng không phụ thuộc gì, chỉ có tầng khác phụ thuộc vào nó.

### 2.2 Technology Survey

| Công nghệ | Vì sao chọn | Dùng ở đâu | Ghi chú |
|---|---|---|---|
| **YAML + PyYAML** | Đọc/ghi cấu hình có comment, lồng nhau, dễ đọc với con người | `configs/*.yaml`, `utils/config.py` | `yaml.safe_load()` để an toàn; PyYAML đã có trong `requirements.txt` |
| **Pytest** | Chạy unit test gọn, không cần dataset/weights/internet | `tests/` | Theo yêu cầu Step 02 |
| **Python `logging` (stdlib)** | Logger chuẩn, đủ INFO/WARNING/ERROR, không cần framework ngoài | `utils/logger.py` | Cấu hình một root logger duy nhất |
| **`RotatingFileHandler` (logging.handlers)** | File log tự xoay vòng (1 MB × 3), tránh log phình to | `utils/logger.py` | Tùy chọn file logging |
| **PyTorch** | Chọn `torch.device`, seed cho CUDA khi có GPU | `utils/device.py`, `utils/seed.py` | Ở Step 02 chỉ dùng phần device/RNG |
| **NumPy** | Gieo seed numpy — dùng ở các bước sau | `utils/seed.py` | |

Quyết định chung: **không thêm framework ngoài** (không logging framework, không config framework) — giữ tầng này nhẹ, dễ hiểu, phù hợp coursework.

### 2.3 Capability Comparison

| Tiêu chí | YAML | JSON | Python literal / argparse |
|---|---|---|---|
| Comment | ✅ Có | ❌ | ❌ |
| Lồng nhóm dữ liệu | ✅ tốt | ✅ nhưng khó đọc khi dài | ⚠️ tùy cách viết |
| Dễ đọc với người không code | ✅ | ⚠️ | ❌ |
| An toàn khi parse | ✅ `safe_load` chuẩn hoá | ✅ `json.load` | ⚠️ cần `ast.literal_eval` |
| Áp dụng trong dự án | ✅ **đã chọn** | — | — |

Lý do chọn YAML: đọc được comment giải thích từng nhóm cài đặt (quan trọng khi demo/nộp bài), biểu diễn lồng nhau khớp nhóm `model/data/train/...` của kiến trúc, và `safe_load` tránh lỗ hổng khi nạp nội dung lạ.

Về **loader**, thiết kế so sánh giữa 3 cách:

| Cách | Ưu | Nhược | Kết luận |
|---|---|---|---|
| Đọc lại từ file mỗi lần | Luôn mới nhất | I/O lặp lại | Có `use_cache` mặc định |
| Cache + `deepcopy` trả ra | Nhanh; caller không làm hỏng config gốc | — | ✅ **chọn** |
| Cache + trả tham chiếu | Nhanh | Caller vô tình sửa chung | ❌ |

Loader còn hỗ trợ `overrides` deep-merge (dùng cho CLI, ví dụ `--device`) mà **không làm ô nhiễm cache** — đã có test xác minh (`test_overrides_do_not_poison_cache`).

### 2.4 Why U-Net + MiDaS?

**Không áp dụng ở Step 02.** Bước này chưa cài đặt hay so sánh model nào; U-Net và MiDaS sẽ được triển khai và luận giải ở các bước sau (Step 04–07). Tại Step 02 chỉ chuẩn bị cấu hình và tiện ích để các model đó sử dụng.

---

## 3. DATA

**Chưa có ở Step 02.** Đúng như scope, bước này **không** triển khai Cityscapes loader hay KITTI depth loader, **không** download dataset. Chi tiết dữ liệu (layout, class map trainId 0–18 / ignore 255, split, KITTI depth PNG mm, `valid_mask`, `depth_cap_m`) sẽ được trình bày ở Step 03 (Dataset).

Trong `configs/pipeline.yaml`, cấu hình dữ liệu cho hai nhánh **đã được khai báo trước** (root, split, `scale_mm: 1000`, `depth_cap_m: 80`, `max_samples: null`) để các bước sau chỉ việc dùng — nhưng chưa có logic đọc dữ liệu.

---

## 4. IMPLEMENTATION PLAN

Vì đây là phần "what was implemented", mục 4 trình bày trực tiếp các thành phần đã triển khai theo đúng 7 hạng mục của Step 02.

### 4.1 Configuration Files

Ba file YAML tạo trong `configs/`, nhóm cấu hình theo kiến trúc (Section 18 của spec):

```text
configs/
├── unet.yaml       # U-Net semantic segmentation (Cityscapes, 19 classes)
├── midas.yaml      # MiDaS monocular depth (pretrained, relative depth)
└── pipeline.yaml   # full pipeline + fusion + analyzer + difficulty + output
```

**Các trường quan trọng trong `configs/unet.yaml`:**

| Nhóm | Giá trị | Ý nghĩa |
|---|---|---|
| `model.num_classes` | `19` | Số class Cityscapes trainId (0..18) |
| `model.base_channels` | `64` | Channels đầu của U-Net encoder |
| `data.image_size` | `[256, 512]` | Resolution huấn luyện [H, W] |
| `inference.image_size` | `[512, 1024]` | Resolution dự đoán (giữ chi tiết) |
| `train.optimizer / loss` | `adam` / `weighted_ce` | Adam; cross-entropy có trọng số class |
| `train.epochs` | `60` (block `train`) | Số epoch thiết kế |
| `training` (khối riêng) | `epochs 20, batch 1, lr 1e-4` | Cấu hình huấn luyện Step 12 (giảm batch do GPU 4 GB) |
| `env.device / seed` | `auto` / `42` | Mặc định toàn hệ thống |

> Ghi chú từ code: resolution `[512, 1024]` + batch 4 gây OOM trên GPU ~4 GB (batch 1 @ 512×1024 đạt ~3 GiB); cấu hình hiện tại `[256, 512]` + batch 1 đỉnh < 1 GiB. Có thể khôi phục bằng CLI `--image-size 512 1024 --batch-size 4` khi có VRAM lớn hoặc AMP.

**`configs/midas.yaml`:**

| Nhóm | Giá trị | Ý nghĩa |
|---|---|---|
| `model.variant` | `dpt_large` | DPT-Large (MiDaS) |
| `model.source` | `hub` | Lấy từ torch.hub (`intel-isl/MiDaS`) |
| `preprocessing.input_size` | `384` | Resize giữ tỉ lệ về 384 trước khi forward |
| `inference.clip_percentile` | `[0.05, 0.995]` | Clip đầu ra relative depth chống outlier |
| `depth.representation` | `relative` | **Relative inverse depth — KHÔNG phải mét** |
| `output.larger_is_farther` | `true` | Convention chuẩn hóa depth cho fusion/visualization |
| `eval.depth_cap_m` | `80` (× `align: lsq_scale_shift`) | Đánh giá KITTI sau này (Step 08) |

**`configs/pipeline.yaml`:**

| Nhóm | Nội dung chính |
|---|---|
| `data` | `image_size: [512, 1024]` chung; block `cityscapes` và `kitti` |
| `system` | `device: auto`, `seed: 42` |
| `steps` | Cờ bật/tắt `run_seg / run_depth / run_fuse / run_scene / run_eval / run_viz` |
| `output` | `base_dir`, `segmentation_dir`, `depth_dir`, `analysis_dir`, `run_subfolder` |
| `fusion` | `roi_classes [11..18]` (person..bicycle), `drivable_classes [0,1]` (road, sidewalk), quantiles, `min_cc_area` |
| `analyzer` | `near_quantile`, `obstacle_frac`, `person_near_road_margin` |
| `difficulty` | `factor_weights` (5 hệ số), bins `easy 0.4 / medium 0.7` |
| `visualization` | `enabled`, `output_dir`, alpha overlay |
| `logging` | `level INFO`, file `outputs/logs/pipeline.log`, rotate 1 MB × 3 |

Thiết kế giữ cấu hình **gọn, nhóm rõ theo trách nhiệm** để các bước sau (fusion, difficulty, visualization) chỉ việc đọc đúng nhóm của mình.

### 4.2 Configuration Loader — `utils/config.py`

Module duy nhất đọc YAML. Bảo toàn fix root-path của spec (I-1): `ROOT_DIR = Path(__file__).resolve().parent.parent` → đúng thư mục gốc dự án (thay vì rơi vào `utils/`).

| Hàm | Nhiệm vụ |
|---|---|
| `load_config(name, overrides=None, use_cache=True)` | Đọc config theo **tên** (ví dụ `"unet"` hoặc `"unet.yaml"`), trả `dict` đã deep-copy, hỗ trợ deep-merge `overrides`, cache lại kết quả |
| `config_path(name)` | Tự thêm `.yaml` nếu thiếu, trả đường dẫn trong `configs/` |
| `get(cfg, "a.b.c", default)` | Lấy giá trị theo dot notation, trả default khi thiếu |
| `resolve_path(path)` | Giải quyết path so với project root; giữ nguyên absolute và mở rộng `~` |
| `clear_config_cache()` | Xóa cache (dùng trong test) |

**Xử lý lỗi rõ ràng** — tất cả qua `ConfigError`:
- File không tồn tại → `Configuration file not found: <path>`;
- YAML sai cú pháp → `Failed to parse configuration ...`;
- Root không phải mapping → báo type nhận được.

**3 thiết kế quan trọng:**
1. `yaml.safe_load` — an toàn, không tự do tạo object.
2. Trả về bản **deep-copy** → caller sửa thoải mái, config gốc/cache không đổi.
3. `overrides` **deep-merge** (dùng cho CLI như `--device`) nhưng không ghi vào cache (test: `test_overrides_do_not_poison_cache`).

Ví dụ dùng (khớp API thực tế):

```python
from utils.config import load_config, get

cfg = load_config("unet")                 # hoặc "unet.yaml"
n = get(cfg, "model.num_classes")         # 19
```

> Lưu ý điểm khác giữa prompt và code: prompt đưa ví dụ `load_config("configs/unet.yaml")`; API triển khai nhận **tên config** (`"unet"` / `"unet.yaml"`) rồi nối vào `CONFIG_DIR` — nên `load_config("unet")` là cách dùng đúng.

### 4.3 Random Seed — `utils/seed.py`

`set_seed(seed)` gieo seed đồng bộ cho:

| Nguồn RNG | Cách gieo |
|---|---|
| Python `random` | `random.seed(seed)` |
| Hash-managed ordering | `os.environ["PYTHONHASHSEED"] = str(seed)` |
| NumPy | `np.random.seed(seed)` |
| PyTorch (CPU + CUDA) | `torch.manual_seed(seed)`; nếu có CUDA: `torch.cuda.manual_seed(_all)(seed)` |
| cuDNN | `torch.backends.cudnn.deterministic = True`, `benchmark = False` |

Mục đích là **reproducibility**. Lưu ý (đúng yêu cầu Step 02): không khẳng định determinism hoàn toàn trên mọi GPU/môi trường phần mềm — cuDNN deterministic giúp việc lặp lại kết quả khả thi nhưng có thể **chậm hơn** ở một số tác vụ convolution.

### 4.4 Logger — `utils/logger.py`

```
setup_logging(level, log_file, ...)  → root logger "cvproject" (idempotent)
get_logger(name)                     → logger con thừa kế handler của root
```

- Console (stdout) + file tùy chọn; `RotatingFileHandler` 1 MB × 3 backup.
- **Idempotent:** mỗi lần `setup_logging` thay handler mới (không ghi trùng → không bị duplicate messages khi gọi nhiều lần). `get_logger` tự cấu hình mặc định nếu chưa có.
- Thêm hai tiện ích phục vụ ghi nhật ký run:
  - `config_hash(config)` — fingerprint SHA-256 (16-hex) từ JSON sắp xếp key: để biết log tương ứng cấu hình nào;
  - `log_run_header(logger, config)` — ghi dòng `RUN | config_hash=...`.

### 4.5 Device Utility — `utils/device.py`

`resolve_device(device="auto")` chọn thiết bị theo luồng:

```text
device = auto
     ↓
CUDA available?
    /       \
   yes      no
   ↓        ↓
 cuda      cpu
```

- Hỗ trợ `"auto"` (mặc định), `None` (coi như auto), `"cpu"`, `"cuda"`; giá trị khác truyền qua `torch.device` (device không hợp lệ → raise).
- `cuda_available()` — kiểm tra CUDA;
- `device_summary()` — `dict` mô tả host (`device`, `cuda_available`, `device_count`, `torch_version`) để ghi vào log/report.
- Không hard-code GPU cụ thể nào.

### 4.6 Requirements & Documentation

`requirements.txt` hiện đã có `PyYAML` và `pytest` (đáp ứng yêu cầu tối thiểu của Step 02) cùng stack đã quyết định ở Step 01 (`torch`, `torchvision`, `numpy`, `Pillow`, `opencv-python`, `matplotlib`, `tqdm`, `scikit-learn`).

Tài liệu sử dụng các tiện ích được ghi tóm tắt trong chính docstring mỗi module (`config`, `seed`, `logger`, `device`) và tại mục 5.4 (cách chạy test).

---

## 5. SYSTEM BUILD FLOW

- **5.1 U-Net Build / 5.2 MiDaS Build / 5.3 Full Pipeline:** chưa áp dụng ở Step 02 — chưa có model hay pipeline (sẽ trình bày ở các bước từ Step 04 trở đi).
- **5.5 CI/CD:** chưa áp dụng ở Step 02 (chưa có GitHub Actions; chỉ chạy test local).

### 5.4 Unit Tests

Bốn file test cho tầng config/utils, chạy bằng pytest, **không** cần dataset/weights/internet:

| File | Số test | Kiểm tra chính |
|---|---|---|
| `tests/test_config.py` | 18 | `ROOT_DIR`/`CONFIG_DIR` đúng, 3 file config tồn tại, load đúng giá trị từng config, kiểu list/int, `load_config("unet") == load_config("unet.yaml")`, file thiếu → `ConfigError`, trả deep copy (không làm hỏng cache), `overrides` merge + không poison cache, `get()` dot access, `resolve_path` (relative/absolute/không tạo file) |
| `tests/test_seed.py` | 4 | `set_seed` lặp lại cùng seed → cùng dãy Python/NumPy/PyTorch; seed khác → dãy khác; giá trị biết trước; không raise trên CPU |
| `tests/test_logger.py` | 8 | setup trả logger, ghi được console + file, **idempotent** (không nhân đôi handler), logger con thừa kế, `config_hash` deterministic + khác khi đổi cấu hình, `log_run_header` ghi `config_hash=` |
| `tests/test_device.py` | 6 | explicit `cpu`, `auto` (cuda nếu available, ngược lại cpu), `None == auto`, device lạ raise, `cuda_available` là bool, `device_summary` đủ field |

Cách chạy (theo README):

```bash
.venv/bin/python -m pytest -q
```

Kết quả kiểm chứng (chạy tại môi trường hiện tại):

```text
tests/test_config.py tests/test_seed.py tests/test_logger.py tests/test_device.py
36 passed in 5.16s
```

Kiểm chứng đọc cấu hình (API thực tế):

```bash
.venv/bin/python -c "from utils.config import load_config; c=load_config('unet'); print(c['model']['num_classes'], c['model']['base_channels'])"
# → 19 64
```

---

## 6. EVALUATION

**Không áp dụng ở Step 02.** Bước này chưa có model hay pipeline nên không có metric nào (mIoU, AbsRel…) và không có kết quả đánh giá. "Đánh giá" ở đây chỉ là **kiểm tra unit test** (mục 5.4) đảm bảo tầng hạ tầng hoạt động đúng. Mọi con số về độ chính xác sẽ thuộc các bước từ Step 13 trở đi.

---

## 7. AI-ASSISTED DEVELOPMENT

### 7.1 Prompt

Bước được thực hiện theo task brief `prompts/02_config_utils.md` (nằm trong repository), dựa trên kiến trúc `prompts/01_architecture_specification.md`. Prompt xác định rõ scope (chỉ config + utils) và cấm triển khai dataset/model/fusion/evaluation.

### 7.2 Generated / Modified Scripts

| Loại | File | Trạng thái |
|---|---|---|
| Config | `configs/unet.yaml`, `configs/midas.yaml`, `configs/pipeline.yaml` | Tạo |
| Utils | `utils/config.py`, `utils/seed.py`, `utils/logger.py`, `utils/device.py` | Tạo |
| Tests | `tests/test_config.py`, `tests/test_seed.py`, `tests/test_logger.py`, `tests/test_device.py` | Tạo |
| Requirements | `requirements.txt` | Bổ sung `PyYAML`, `pytest` |

### 7.3 Testing

- Chạy bộ pytest của Step 02: **36 passed in 5.16s**;
- Smoke test đọc cấu hình `load_config("unet")` trả giá trị đúng;
- Không yêu cầu dataset, model weights hay internet.

### 7.4 Version History

Dự án quản lý phiên bản bằng Git; các file của Step 02 đang hiện diện trong cây làm việc hiện tại. Lịch sử commit riêng cho từng bước coursework không được lưu chi tiết ở đây (không có thông tin tách biệt mức commit cho Step 02) — ghi nhận là **chưa sẵn có**.

### 7.5 Output

Đầu ra của Step 02 là toàn bộ tầng `configs/ + utils/ + tests` đã mô tả ở mục 4–5, kèm tài liệu này (`docs/coursework/02_config_utils.md`).

---

## 8. DISCUSSION

### 8.1 Strengths

- Tầng hạ tầng **nhẹ, không phụ thuộc framework ngoài** (chỉ thêm PyYAML/pytest), dễ hiểu khi trình bày;
- Lỗi config **báo rõ ràng** bằng `ConfigError` (file thiếu / YAML sai / root sai kiểu);
- Loader **an toàn** (deep copy + không poison cache + `safe_load`);
- Logger **idempotent** — tránh lỗi log trùng lặp phổ biến;
- Device `auto` giúp code chạy được cả trên CPU lẫn GPU mà không sửa gì;
- Cấu hình chia nhóm đúng kiến trúc (Section 18 của spec) → các bước sau chỉ đọc đúng nhóm của mình.

### 8.2 Weaknesses

- API `load_config` nhận **tên** config trong khi prompt mô tả bằng **đường dẫn** ví dụ — người mới dễ gọi `load_config("configs/unet.yaml")` (sẽ lỗi `not found`);
- `image_size` xuất hiện ở **hai nơi khác nhau** (`data.image_size` cho training, `inference.image_size` cho prediction) — dễ nhầm nếu không đọc kỹ comment;
- Trong `unet.yaml` tồn tại **hai khối** gần trùng tên (`train` và `training`) — cố ý cho hai bước khác nhau, nhưng gây hơi rối ràng.

### 8.3 Limitations

- `set_seed` **không đảm bảo determinism tuyệt đối** trên mọi GPU/software env (cuDNN deterministic chỉ khả thi, có thể chậm);
- Các giá trị config chưa được component nào tiêu thụ ở bước này → tính đúng đắn của toàn bộ chuỗi config sẽ chỉ được kiểm chứng khi các bước Dataset/Model/Pipeline triển khai;
- Chưa có CI/CD; toàn bộ kiểm tra chỉ ở mức local.

---

## 9. CONCLUSION

Step 02 đã xây dựng xong tầng nền tảng **Config + Utils**: ba file cấu hình chuẩn theo từng nhóm (U-Net, MiDaS, pipeline), loader YAML an toàn với cache + overrides và lỗi rõ ràng, seed tái lập được, logger idempotent kèm config fingerprint, và device utility với chế độ `auto`. Toàn bộ đi kèm **36 unit test** chạy qua, không cần dataset lẫn model weights. Đây chính là "mặt bằng" thống nhất để các bước tiếp theo (Dataset → U-Net → MiDaS → Fusion → Pipeline) chỉ việc đọc cấu hình qua một điểm duy nhất và không phải giải quyết lại các vấn đề seed/log/device.

---

## 10. REFERENCES

- `prompts/02_config_utils.md` — task brief của Step 02 (trong repository).
- `prompts/01_architecture_specification.md` — kiến trúc tổng thể, nguồn quyết định cấu trúc config và dependency rule.
- `README.md` — tổng quan dự án và cách chạy test.
- Các module trực tiếp: `utils/{config,seed,logger,device}.py`, `configs/*.yaml`, `tests/test_{config,seed,logger,device}.py`.