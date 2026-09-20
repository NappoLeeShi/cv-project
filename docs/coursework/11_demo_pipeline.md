# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

## Coursework Report — Step 11: Demo / Full Pipeline

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Tài liệu này mô tả bước **Demo / Full Pipeline** — nối các module đã có thành
> một pipeline hoàn chỉnh chạy trên một ảnh đường phố bất kỳ, theo đúng
> `prompts/11_demo_pipeline.md`.
> Mọi chi tiết được đối chiếu với repository hiện tại.

---

## 1. REQUIREMENT

### 1.1 Problem

Sau các bước trước, project đã có từng module riêng lẻ (U-Net, MiDaS, Fusion,
Scene Understanding, Visualization) nhưng chưa có một **orchestration layer**
nối chúng lại: đưa một ảnh street vào, chạy đủ chuỗi xử lý, và trả về kết quả
có cấu trúc. Step 11 yêu cầu tạo pipeline này.

### 1.2 Objective

Nối toàn bộ module thành pipeline hoàn chỉnh:

```text
Street Image
    ↓
U-Net ──────────────> Semantic Segmentation
    +
MiDaS ──────────────> Relative Inverse Depth
    ↓
Fusion
    ↓
Scene Understanding
    ↓
Visualization
    ↓
Saved Outputs
```

Step này **chỉ tích hợp các module hiện có** — không viết lại model, không đổi
architecture, metrics, fusion/analyzer logic, không tải dataset, không tự động
download pretrained weights.

### 1.3 Input / Output

- **Input**: một ảnh street RGB bất kỳ (cùng một image cho cả U-Net và MiDaS);
- **Output**: `PipelineResult` chứa tối thiểu `segmentation`, `depth`, `fusion`,
  `scene_report`; kèm `visualization_paths` (map kiểu output → đường dẫn file)
  khi bật visualization.

### 1.4 Scope

- Tạo/sử dụng `scene_understanding/pipeline.py` — **orchestration layer**, không
  chứa neural-network logic;
- Implement `SceneUnderstandingPipeline` nhận predictor qua **dependency injection**
  (không hard-code U-Net/MiDaS bên trong);
- Tuân thủ **same-image contract**; align shape theo policy rõ ràng về resolution
  của ảnh gốc; `visualize=True/False`; output dir từ config; CLI; tests offline.

### 1.5 Explicit Out of Scope / STOP

Theo `prompts/11_demo_pipeline.md` §12, §19, §21, §22:

- **Không** tự động download Cityscapes/KITTI/U-Net weights/MiDaS weights;
  checkpoint/weights thiếu → raise lỗi rõ ràng ("U-Net checkpoint is required
  for real inference"), không silently fallback sang random weights;
- **Không** tạo duplicate implementation của model/fusion/analyzer/visualization;
- **Không** tạo thêm `training/`, `agent/`, neural fusion network, `detector/`,
  `tracker/`, `lane_detection/`, `3D reconstruction/`, cloud deployment;
- **Không** dùng `except:` bare để nuốt lỗi;
- Pipeline demo **độc lập dataset**: không import `CityscapesDataset`/`KITTIDataset`;
  dataset loaders chỉ phục vụ evaluation;
- Test không được trigger model download;
- STOP sau Step 11 (không tự sang Step 12).

---

## 2. PURPOSE & SURVEY

### 2.1 Purpose

Biến các module rời rạc thành một **hệ thống thống nhất** mà người dùng chỉ cần
đưa một ảnh vào và nhận kết quả hiểu cảnh (scene understanding) kèm ảnh trực quan.
Pipeline cũng là nền cho CLI demo và cho các bước đánh giá pipeline sau này.

### 2.2 Relevant Technical Background

- **Orchestration pattern**: tách lớp điều phối (gọi predictor → align → fusion →
  analyzer → visualization) khỏi lớp thực thi model;
- **Dependency injection**: pipeline không biết predictor cụ thể là ai — chỉ cần
  object có `predict(image)`; nhờ vậy test dùng dummy predictors mà không cần model
  thật;
- **Same-image contract** (quan trọng của Step 11): cùng một image object được
  truyền vào cả U-Net và MiDaS; tuyệt đối không lấy ảnh Cityscapes cho U-Net và
  ảnh KITTI cho MiDaS, không pair hai dataset thành một scene;
- **Shape alignment policy**: segmentation resize bằng **nearest-neighbour** (giữ
  nguyên class ID), depth resize bằng **bilinear** — cùng về resolution ảnh gốc;
  không silently resize bên trong fusion.

### 2.3 Technology / Method Survey

Pipeline dùng các module hiện có:

- `scene_understanding/fusion.fuse()` → `FusionResult` (per-class stats,
  region map far/middle/near theo quantiles `[0.3333, 0.6667]`);
- `scene_understanding/analyzer.analyze_fusion()` → scene report
  (`scene`, `semantic_distribution`, `depth_distribution`, `regions`,
  `traffic_context`, `interpretation`);
- `visualization/*` → overlay, depth, fusion, overview, scene report JSON;
- `utils/config.load_config()` + `get()/resolve_path()` → cấu hình `pipeline.yaml`.

### 2.4 Why This Approach?

- Orchestration tách riêng là **đơn giản, dễ test, dễ tái dùng**;
- Dependency injection giúp test **offline hoàn toàn** (dummy predictors), không
  cần weights/internet/CUDA;
- Policy align rõ ràng tránh "silent resize" gây sai lệch màu vẫn không biết;
- Giữ đúng giới hạn Step 11: chỉ tích hợp, không thêm model mới.

---

## 3. DATA

Step 11 **không xử lý dataset** — không dùng Cityscapes/KITTI trong demo.

### 3.1 Dataset / Input

- Demo nhận **một arbitrary street image** (độc lập dataset);
- Dataset loaders (`CityscapesDataset`, `KITTIDataset`) được giữ riêng cho
  evaluation, **không** được import vào pipeline demo (§19).

### 3.2 Data Format

- Image: PIL `Image` hoặc numpy array HWC RGB `[H, W, 3]` (uint8, hoặc float
  `[0,1]` được chuyển đổi);
- Segmentation prediction: tensor/numpy `[H, W]` integer train IDs (ép `int64`);
- Depth prediction: tensor/numpy `[H, W]` float, **relative inverse depth**
  (`larger = closer`), không phải meters.

### 3.3 Data Flow

```text
image (PIL / np uint8 RGB)
   ├─> segmentation_predictor.predict(image) ─> [H, W] trainId map
   ├─> depth_predictor.predict(image)        ─> [H, W] relative inverse depth
   └─> align (nearest / bilinear) về resolution ảnh gốc
         └─> fuse() ─> FusionResult
               └─> analyze_fusion() ─> scene_report dict
                     └─> (optional) visualization files + scene_report.json
```

### 3.4 Data Visualization

Visualization sinh ra (khi `visualize=True`) — đúng chuẩn đã xây ở Step 10:

| File | Nội dung |
|---|---|
| `scene_segmentation.png` | Segmentation colorized (palette 19 class) |
| `scene_depth.png` | Depth colorized (relative inverse depth) |
| `scene_fusion.png` | Segmentation + depth overlay |
| `scene_overview.png` | Figure 4+ panel tổng quan |
| `scene_report.json` | Scene report UTF-8 JSON (`indent=2`) |

---

## 4. IMPLEMENTATION PLAN

### 4.1 Architecture / Components

```
site_package: scene_understanding/
├── pipeline.py   (Step 11)  # SceneUnderstandingPipeline, PipelineResult, PipelineError
├── fusion.py     (trước đó)  # fuse(), FusionResult
└── analyzer.py   (trước đó)  # analyze_fusion()
```

`SceneUnderstandingPipeline` là **orchestration layer** — không chứa logic mạng
thần kinh, không import model implementation riêng.

### 4.2 Main Implementation

| Thành phần | Mô tả |
|---|---|
| `SceneUnderstandingPipeline` | Class pipeline; constructor nhận `segmentation_predictor`, `depth_predictor`, optional `cfg`, `output_dir`, `roi_classes`, `drivable_classes`, `quantiles`; kiểm tra predictor có `predict` callable |
| `PipelineResult` | `@dataclass`: `image`, `segmentation`, `depth`, `fusion`, `scene_report`, `visualization_paths`; method `to_dict()` trả metadata JSON-serializable (không nhét numpy/tensor thô) |
| `PipelineError` | Lỗi orchestration (kế thừa `RuntimeError`) |
| `run(image, *, visualize=True, output_dir=None)` | Luồng chính: coerce image → predict cả hai → align → fuse → analyze → visualize (tùy chọn) |

Các helper private:

- `_coerce_image()` — chuyển PIL/np array về `uint8` HWC RGB; raise `PipelineError`
  khi shape/dtype không hợp lệ;
- `_prediction_ndarray()` — nhận tensor/np array, yêu cầu 2D, không zero-size;
- `_segmentation_ndarray()` — yêu cầu integer dtype, ép `int64`;
- `_depth_ndarray()` — yêu cầu không boolean, ép `float64`;
- `_resize_nearest_ids()` / `_resize_bilinear()` — align policy (không đổi gì nếu
  đã đúng resolution);
- `_write_visualizations()` — ghi 4 ảnh + scene report vào output dir.

### 4.3 Configuration

`configs/pipeline.yaml` (dùng mặc định qua `load_config("pipeline")`):

- `visualization.output_dir: outputs/visualization` — mặc định khi không truyền
  `output_dir`;
- `visualization.alpha_segmentation: 0.5`, `alpha_depth: 0.35` — hệ số overlay;
- `fusion.region_quantiles: [0.3333, 0.6667]`, `roi_classes` (11..18),
  `drivable_classes` (0,1) — chuyển tiếp xuống fusion/analyzer;
- `system.seed: 42` — seed cho những bước demo/đánh giá (qua `utils.seed`).

### 4.4 Processing Flow

1. `_coerce_image(image)` — validate, chuyển về `uint8` HWC RGB;
2. `segmentation = _segmentation_ndarray(seg.predict(image))`;
3. `depth = _depth_ndarray(depth.predict(image))` — **cùng image object**;
4. align: `_resize_nearest_ids(seg, image_size)`, `_resize_bilinear(depth, image_size)`;
5. `fusion = fuse(segmentation, depth, quantiles=self.quantiles)`;
6. `scene_report = analyze_fusion(fusion, cfg=self.cfg, roi_classes=..., drivable_classes=...)`;
7. nếu `visualize=True`: `_write_visualizations(...)`; ngược lại `visualization_paths={}`;
8. trả `PipelineResult`.

### 4.5 Error Handling

Theo Step 11 §21 — lỗi rõ ràng, không bare `except`:

| Tình huống | Lỗi |
|---|---|
| Image không phải HWC RGB / sai dtype | `PipelineError` (mô tả rõ) |
| Predictor không có `predict` | `PipelineError` |
| Prediction không 2D / zero-size | `PipelineError` |
| Segmentation không integer | `PipelineError` |
| Depth boolean | `PipelineError` |
| Lỗi từ predictor | **propagate** nguyên vẹn (không nuốt) |
| Checkpoint/weights thiếu (CLI real demo) | `RuntimeError` rõ ràng |

### 4.6 Integration with Previous Steps

- Reuse `fusion` (Step 04/05), `analyzer` (Step 05), `visualization` (Step 10);
- `configs/pipeline.yaml` là nguồn cấu hình chung;
- CLI `main.py` gọi pipeline cho `--image` và `evaluation/custom_demo.py`
  (folder demo) cho `--input-dir`.

---

## 5. SYSTEM BUILD FLOW

### 5.1 Step-by-Step Execution Flow

```text
python main.py --image path/to/image.jpg \
    --unet-checkpoint path/unet.pt \
    --midas-weights path/midas.pt
```

1. `build_parser()` chuẩn bị CLI (mutually-exclusive `--image` / `--input-dir`);
2. `require_image_file()` — kiểm tra ảnh tồn tại;
3. `require_predictor_inputs()` — kiểm tra checkpoint/weights; thiếu → lỗi rõ ràng,
   không download;
4. `build_real_predictors()` — lazy import `UNetInference` + `MidDepthPredictor`
   (parsing/tests không trigger model load);
5. `SceneUnderstandingPipeline(unet, midas, output_dir=...)` — output dir mặc định
   từ `configs/pipeline.yaml > visualization.output_dir`;
6. `pipeline.run(image, visualize=not args.no_visualization)`; in kết quả
   (`segmentation shape`, `depth shape`, nearest dynamic class, các output path).

Có cờ `--device auto|cpu|cuda`, `--output-dir`, `--no-visualization`
(`--no-visualization` vẫn ghi scene-report JSON).

### 5.2 Integration with Existing Pipeline

- **Single-image**: `main._run_single_image()` → pipeline → visualization files ở
  `outputs/visualization/` (hoặc `--output-dir`);
- **Folder custom demo** (`--input-dir`): `evaluation/custom_demo.run_custom_demo()`
  — reuse `evaluate_single_image` (Step 15 core) + `save_custom_visualizations`
  (naming `<stem>_<kind>.*`) viết vào `outputs/custom/`; dùng checkpoint mặc định
  `checkpoints/unet_cityscapes.pth` / `checkpoints/dpt_large_384.pt`;
- **Evaluation** (Steps 13–15) vẫn độc lập khỏi demo: Cityscapes→U-Net→metrics,
  KITTI→MiDaS→metrics.

### 5.3 Testing

`tests/test_pipeline.py` — chạy offline, không model, không dataset, không CUDA,
không internet. Dùng:

- `RecordingSegmentationPredictor` / `RecordingDepthPredictor` (deterministic,
  ghi lại `calls` và `last_input`) — synthetic segmentation:
  sky=10, road=0, car=13, person=11;
- Dummy depth: `(yy/h)·60`, cộng thêm offset vùng car/person (foreground gần →
  inverse depth lớn hơn);
- `street_image` fixture: H=128, W=256.

Phạm vi 27 tests (đầy đủ theo Step 11 §15):

1. pipeline construction (+ reject predictor thiếu `predict`);
2. run với dummy predictors;
3. **same image passed to both predictors** (identity `seg.last_input is depth.last_input`);
4. segmentation shape = image shape;
5. depth shape = image shape;
6. fusion execution (`FusionResult`, thresholds low<high, region_map);
7. scene analyzer execution (đủ các key, vehicles/pedestrians);
8. `PipelineResult` structure + `to_dict()` JSON-serializable;
9. `visualize=False` → không tạo file;
10. `visualize=True` → đủ 5 output (`segmentation`, `depth`, `fusion`, `overview`,
    `scene_report`) và file tồn tại;
11. output dir auto-created (parent chain);
12. invalid image handling (sai ndim/channel/dtype/kiểu);
12b. PIL image được chấp nhận;
13. shape mismatch → align về resolution ảnh; non-2D bị reject; segmentation
    non-integer bị reject;
14. predictor failure **propagate** (không nuốt);
15. deterministic result (`scene_report`/mảng/`to_dict` giống nhau qua 2 lần chạy);

Synthetic demo: report có `scene.depth_convention == inverse_relative_larger_closer`,
`semantic_distribution.sky/road > 0`, `traffic_context.vehicles.car`,
`pedestrians.person`, `nearest_dynamic_class` ∈ {car, person}, `interpretation`
không rỗng; depth road (dưới) > depth sky (trên).

CLI tests (không load model): `--help` exit 0, thiếu `--image` → SystemExit,
ảnh không tồn tại → `ValueError`, parse đủ flags, missing predictor inputs →
`RuntimeError` với message rõ ràng, parse không import models.

Kết quả kiểm tra: `pytest tests/test_pipeline.py` → **27 passed** (4.00s).

### 5.4 CI/CD

**Không được thiết lập trong repository** — không có `.github/workflows` hay
workflow file. Kiểm thử chạy thủ công qua `pytest`. Full suite hiện tại:
**456 passed, 1 skipped** (22.65s) — xác nhận không regression so với trước
(`tests/test_pipeline.py` đóng góp 27 test mới).

---

## 6. EVALUATION

### 6.1 What Step 11 Actually Evaluates

Step 11 **không đánh giá hiệu năng model**. Các mục đánh giá ở đây là:

- **Functional validation**: pipeline chạy đúng luồng (predict → align → fusion →
  analyze → visualize) với dummy predictors;
- **Contract validation**: same-image contract, shape contract, output directory,
  deterministic, JSON serialization;
- **Regression**: toàn bộ test suite từ Step 02 → Step 10 vẫn pass.

### 6.2 Measured Results / Validation

| Hạng mục | Kết quả (đo thực tế) |
|---|---|
| `tests/test_pipeline.py` | **27 passed** (4.00s) |
| Full suite `pytest -q` | **456 passed, 1 skipped** (22.65s) |

(1 skipped = test CUDA path cho U-Net inference; không có GPU nên như vậy là
bình thường, đúng tinh thần Step 11 §23.)

### 6.3 Design Targets vs Measured

- Prompt §23 ghi "Expected: 305 passed, 1 skipped" — con số đó phản ánh kỳ vọng
  lúc viết prompt ở giai đoạn repo khác; **giá trị thực tế hiện tại** của repo là
  **456 passed, 1 skipped** (số đã tăng do các bước 12–17 code đã merge vào
  commit `47678b0`). Báo cáo ưu tiên số đo thực tế.
- Smoke test (các lần chạy demo) là **hàm kiểm tra nhanh/chạy demo**, không phải
  benchmark thống kê; không trình bày là thí nghiệm có ý nghĩa.

### 6.4 Interpretation

- Pipeline orchestrates đúng, deterministic, offline-testable;
- Same-image contract được chứng minh bằng test identity (`is`);
- Không có metric hiệu năng model mới nào ở Step 11; mọi số model-level lấy từ
  `outputs/analysis/*.json` (Step 06) — không thay đổi.

---

## 7. AI-ASSISTED DEVELOPMENT

### 7.1 Prompt

- Prompt file: `prompts/11_demo_pipeline.md` — mô tả pipeline class, same-image
  contract, shape policy, output dir, CLI, offline tests (15 mục test), synthetic
  demo, error handling, deliverables, và yêu cầu "STOP sau Step 11".

### 7.2 Files / Scripts Generated or Modified

- **Tạo mới:**
  - `scene_understanding/pipeline.py` — `SceneUnderstandingPipeline`,
    `PipelineResult`, `PipelineError`, `DEPTH_CONVENTION`;
  - `tests/test_pipeline.py` — 27 test offline.
- **Modified:**
  - `main.py` — CLI hoàn chỉnh (`--image` / `--input-dir`, `--output-dir`,
    `--unet-checkpoint`, `--midas-weights`, `--no-visualization`, `--device`,
    `--limit`), `require_predictor_inputs()`, `build_real_predictors()`;
  - `configs/pipeline.yaml` — giữ nguyên cấu hình chung (visualization section đã
    có từ Step 10; pipeline dùng `visualization.output_dir`, `fusion`,
    `analyzer`, `system.seed`).
- **Không** sửa model/metric/fusion/analyzer logic.

Git history: các file pipeline/main/tests xuất hiện qua các commit code-side
(`d4a21f6` "unet module", `a9a55ba` "70% processing", `47678b0`
"Steps 12-17: unet/midas evaluation, pipeline fusion, custom demo, report").
Step 11 là bước nối module theo nhóm prompt 11 (_demo_pipeline) trong trình tự
prompt 00–17; báo cáo coursework 00–11 trình bày lần lượt theo trình tự đó.

### 7.3 Testing / Validation Performed

- Viết test theo đúng §15 (15 mục) + CLI §16 (không load model) + synthetic demo
  §17;
- Chạy full suite để đảm bảo **regression zero**: 456 passed, 1 skipped.

### 7.4 Verified Facts vs Unavailable

- **Verified**: API, config keys, test counts, CLI behavior (đều đọc trực tiếp từ
  source).
- **Not available in the repository**: nếu có chi tiết thủ công khác của quá
  trình dùng AI mà repo không lưu (vd chat transcript ngoài prompt) — không nêu.

---

## 8. DISCUSSION

### 8.1 Strengths

- **Orchestration thuần** — pipeline không chứa logic neural network, dễ đọc/dễ test;
- **Dependency injection** — cùng pipeline chạy được với model thật hoặc dummy;
- **Same-image contract** được đảm bảo và được kiểm chứng bằng test (identity);
- **Align policy rõ ràng** — nearest cho class ID, bilinear cho depth;
- **Error handling minh bạch** — lỗi không bị nuốt, message rõ ràng;
- **Offline hoàn toàn** — test không cần weights/dataset/CUDA/internet;
- Integration với CLI và visualization có sẵn (kế thừa Step 10) — ít code trùng.

### 8.2 Weaknesses

- Align dùng interpolation **bilinear** cho depth có thể làm mờ biên vùng ở
  resolution thấp (chấp nhận được nhưng cần biết);
- `visualize=True` luôn ghi 5 file vào cùng output dir — chưa có cơ chế subfolder
  timestamp (config `output.run_subfolder` tồn tại nhưng pipeline dùng trực tiếp
  `output_dir`);
- Pipeline demo chỉ kiểm chứng functional bằng dummy — chưa có smoke với real
  weights trong test suite (demo real chạy thủ công qua CLI).

### 8.3 Limitations

- Demo là **inference-only**: không đọc ground-truth, không tính metric có nghĩa
  thống kê trên một ảnh;
- Không có CI/CD để tự chạy pipeline tests;
- Depth luôn là **relative inverse depth** — không bao giờ sinh metric depth mét.

### 8.4 What Step 11 Does and Does Not Prove

- **Proves**: pipeline nối đủ U-Net + MiDaS + Fusion + Scene Understanding +
  Visualization trên cùng một ảnh; result có cấu trúc; offline testable;
  deterministic; CLI chạy được.
- **Does not prove**: không chứng minh chất lượng phân đoạn/độ sâu (đó là việc
  của Steps 13–15/06); không chứng minh khả năng metric depth; không phải đánh
  giá thống kê; không nói về điều khiển phương tiện hay autonomous driving.

---

## 9. CONCLUSION

Step 11 hoàn thiện **full demo pipeline** cho project: `SceneUnderstandingPipeline`
tiếp nhận mọi object có `predict(image)`, đảm bảo cùng một ảnh vào cả U-Net và
MiDaS, align output về resolution ảnh gốc (nearest/bilinear), rồi chạy fusion,
scene analysis và visualization, trả về `PipelineResult` có cấu trúc và JSON
serializable. CLI `main.py` cho phép chạy single-image demo (`--image`) và
folder demo (`--input-dir`, qua `evaluation/custom_demo.py`).

Step này chỉ tích hợp: không thay đổi model/fusion/analyzer, không thêm
architecture, không tải dataset/weights tự động, vẫn giữ convention
`larger inverse depth = closer` và tách biệt hoàn toàn demo khỏi evaluation trên
dataset. Kiểm thử offline 27 test mới pass, full suite đạt **456 passed,
1 skipped** — không regression.

Step 11 là cầu nối giữa các module đơn lẻ (Steps 01–10) và phần đánh giá
pipeline (các bước sau), đồng thời là nền cho demo thực tế trên ảnh street tùy ý.

---

## 10. REFERENCES

- `prompts/11_demo_pipeline.md` — yêu cầu Step 11 (pipeline, same-image, CLI,
  tests, out-of-scope).
- `scene_understanding/pipeline.py` — `SceneUnderstandingPipeline`, `PipelineResult`,
  `PipelineError`, `DEPTH_CONVENTION`.
- `tests/test_pipeline.py` — 27 test offline (dummy predictors, CLI parsing).
- `main.py` — CLI entry (`--image`, `--input-dir`, `--limit`, `--output-dir`,
  `--unet-checkpoint`, `--midas-weights`, `--no-visualization`, `--device`).
- `evaluation/custom_demo.py` — folder-based custom-image demo (Step 17), reuse
  `evaluate_single_image`; `DEFAULT_UNET_CHECKPOINT`, `DEFAULT_MIDAS_WEIGHTS`,
  `DEFAULT_OUTPUT_DIR`.
- `configs/pipeline.yaml` — `visualization`, `fusion`, `analyzer`, `system.seed`.
- `scene_understanding/fusion.py` — `fuse()`, `FusionResult`.
- `scene_understanding/analyzer.py` — `analyze_fusion()`.
- `visualization/{segmentation,depth,fusion,scene,io}.py` — lớp hiển thị (Step 10).
- `utils/config.py` — `load_config`, `get`, `resolve_path`.
- `README.md` — hướng dẫn demo (`--input-dir`, `outputs/custom/`, same RGB image).
- `docs/coursework/00…10` — architecture, build flow, evaluation, visualization
  (Step 10) trước đó.
- `outputs/custom/pipeline_demo_00{1,2}_*` — sản phẩm demo (inference-only).
- Git history — `d4a21f6` (unet module), `a9a55ba` (70% processing),
  `47678b0` (Steps 12–17 code).
