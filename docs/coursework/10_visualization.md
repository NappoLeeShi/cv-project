# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

## Coursework Report — Step 10: Visualization

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Tài liệu này mô tả **bước Visualization** — tạo lớp hiển thị trực quan cho
> các output của U-Net, MiDaS, Fusion và Scene Understanding, theo đúng
> `prompts/10_visualization.md`.
> Mọi chi tiết API, cấu hình và kết quả kiểm thử được đối chiếu với repository hiện tại.

---

## 1. REQUIREMENT

### 1.1 Problem

Sau các bước trước, project đã có:

- U-Net trả về **semantic segmentation map** (Cityscapes train ID, 0–18, `255` = ignore);
- MiDaS trả về **relative inverse depth map** (không phải metric depth, giá trị lớn = gần hơn);
- Fusion/analyzer tạo ra **scene report** chứa thống kê per-class/per-region, traffic context, interpretation.

Vấn đề: các output này đều ở dạng **mảng số / dictionary**, khó trình bày cho demo, báo cáo
hoặc giảng viên. Cần một **lớp visualization** — chỉ đọc output từ các module khác, không đổi
logic model/fusion/analyzer, và lưu ra file ảnh để xem.

### 1.2 Objective

Step 10 yêu cầu:

- visualize ảnh đầu vào;
- visualize semantic segmentation (màu cố định, deterministic);
- visualize relative inverse depth (giữ đúng convention `larger = closer`, không ghi "meters");
- visualize fusion giữa segmentation và depth;
- visualize scene-understanding result;
- lưu visualization ra file (PNG/JPG, JSON cho scene report);
- tạo các helper cần thiết cho demo sau này.

### 1.3 Scope

Theo `prompts/10_visualization.md`, Step 10 **CHỈ** chịu trách nhiệm hiển thị và lưu file.
**KHÔNG** thay đổi logic của U-Net, MiDaS, segmentation/depth metrics, fusion, scene analyzer.
**KHÔNG** tải dataset, **KHÔNG** tải pretrained weights, **KHÔNG** gọi internet,
**KHÔNG** yêu cầu Cityscapes/KITTI để chạy test.

Các contract bắt buộc được giữ nguyên:

- Segmentation output `[B, H, W]` hoặc single-image `[H, W]` với Cityscapes train IDs;
- MiDaS depth output `[H, W]` là **relative inverse depth** (`larger value = closer to camera`);
- Fusion/analyzer dùng region quantiles `[0.3333, 0.6666...]` (near/middle/far).

Deliverables của Step 10:

```
visualization/
├── __init__.py
├── segmentation.py
├── depth.py
├── fusion.py
└── scene.py
tests/test_visualization.py
(có thể sửa configs/pipeline.yaml nếu cần)
```

> Ghi chú repository: ngoài các file trên, project còn có `visualization/io.py`
> dành riêng cho save utilities (`save_visualization`, `save_figure`).

---

## 2. PURPOSE & SURVEY

### 2.1 Purpose

Mục đích chính:

- **Trực quan hóa** để kiểm tra bằng mắt model làm gì với một ảnh;
- **Minh bạch** — nhìn thấy rõ semantic regions và near/far structure được ghép thế nào;
- **Tái lập** — tạo ra các file ảnh/JSON cố định để dùng cho demo và báo cáo;
- **Phục vụ pipeline** — `scene_understanding/pipeline.py` gọi các hàm visualization để sinh
  `scene_segmentation.png`, `scene_depth.png`, `scene_fusion.png`, `scene_overview.png`,
  `scene_report.json`.

### 2.2 Technical Background

- **Semantic segmentation visualization** truyền thống dùng một **palette màu cố định**
  (Cityscapes có bảng màu chuẩn cho 19 class) — mỗi train ID một màu, `255` (void/ignore)
  thường để đen. Đây là cách phổ biến để đọc bản đồ class.
- **Depth visualization**: raw MiDaS là relative inverse depth; để hiển thị ta **normalize trên MỘT bản sao**
  (`normalized = (depth − min) / (max − min)`) về `[0, 1]` rồi ánh xạ qua matplotlib colormap.
  **KHÔNG được ghi normalized value ngược về prediction gốc.**
- **Overlay**: blend ảnh gốc với lớp màu theo hệ số alpha (`dst = base·(1−α) + overlay·α`),
  ngành xử lý ảnh dùng phổ biến.

### 2.3 Technology / Method Survey

Các lựa chọn trong repository:

- **numpy + PIL** — tạo và lưu ảnh RGB `uint8` (`visualization/io.py`);
- **matplotlib (headless)** — tạo figure, colormap, legend, colorbar (`visualization/depth.py`,
  `segmentation.py`, `scene.py`). Test dùng backend `Agg` để chạy offline không cần GUI;
- **colormap `turbo`** — màu cho depth, liên tục từ đỏ (gần) đến tím/xanh (xa);
- **JSON save** — `save_scene_report` ghi scene report với `indent=2`, UTF-8, không mutate report.

### 2.4 Why This Approach?

- **Deterministic palette** (không random mỗi lần chạy) → kết quả tái lập, dễ so sánh;
- **Read-only** (không mutate input) → an toàn với model outputs, đúng nguyên tắc Step 10;
- **Headless matplotlib (Agg)** → test chạy được trong CI/môi trường không có display;
- **Pure rule-based** → không thêm neural network, không thêm training, không dependency mới
  ngoài matplotlib/PIL vốn đã có.

---

## 3. DATA

Theo `prompts/10_visualization.md`, Step 10 **không sử dụng dataset**.

### 3.1 Dataset / Input

- **Không** dùng Cityscapes, **không** dùng KITTI để chạy visualization hay test;
- Input của các hàm visualization là **output sẵn có của các module**:
  - `segmentation: np.ndarray` `[H, W]` integer train IDs (`255` = ignore);
  - `depth: np.ndarray` `[H, W]` relative inverse depth;
  - `image: np.ndarray` RGB `[H, W, 3]` (uint8 hoặc float `[0,1]`);
  - `region_map: np.ndarray` `[H, W]` với các code `far=0`, `middle=1`, `near=2`, `invalid=-1`
    (theo `scene_understanding/fusion.py`);
  - `scene report: dict` từ analyzer.

### 3.2 Data Format

- Overlay/colorized output: **RGB `uint8`** `[H, W, 3]`;
- Normalized depth: **float32** `[H, W]` trong `[0, 1]` (chỉ trên bản sao);
- Depth figure colorbar label bắt buộc: **"Relative inverse depth (larger = closer)"**;
- Scene report JSON: UTF-8, `indent=2`, `ensure_ascii=False`;
- File ảnh: `.png` / `.jpg` / `.jpeg` (giới hạn bởi `visualization/io.py`).

### 3.3 Data Flow

```text
segmentation map ─┐
depth map (rel.) ─┼─> visualization/ ─> RGB image / Figure ─> save_visualization/save_figure
RGB image        ─┘         │
scene report dict ──────────┴─> save_scene_report ─> JSON file
```

### 3.4 Data Visualization

Ví dụ output thực tế trong repository (`outputs/visualization/smoke/`,
`outputs/custom/pipeline_demo_00{1,2}_*.png`):

```
scene_segmentation.png   scene_depth.png
scene_fusion.png         scene_overview.png
scene_report.json
```

Kiểm tra thực tế: `outputs/visualization/smoke/scene_report.json` là JSON hợp lệ, chứa các key
`scene`, `semantic_distribution`, `depth_distribution`, `regions`, `traffic_context`,
`interpretation`; `scene.depth_convention = inverse_relative_larger_closer`; ví dụ
`traffic_context.nearest_dynamic_class` = `{"class": "person", "median_depth": 30.0,
"proximity": "middle"}` (dữ liệu demo/smoke, không phải kết quả đánh giá).

---

## 4. IMPLEMENTATION PLAN

### 4.1 Architecture / Components

```
visualization/
├── __init__.py      # re-export public API
├── segmentation.py  # colorize_segmentation, overlay, legend
├── depth.py         # normalize, colorize_depth, depth figure
├── fusion.py        # fusion overlay, region visualization
├── scene.py         # scene summary, save_scene_report, full visualization
└── io.py            # save_visualization, save_figure
```

### 4.2 Main Implementation

Các function public (đúng API trong prompt; type hints đầy đủ):

| Module | Function | Chức năng |
|---|---|---|
| `segmentation.py` | `colorize_segmentation(seg, palette=None, ignore_index=255)` | Map train ID → RGB `uint8`; palette cố định 19 màu (`CITYSCAPES_TRAINID_COLORS`), `255` → đen; validate 2D/integer/âm/ngoài palette |
| `segmentation.py` | `create_segmentation_overlay(image, seg, alpha)` | Blend màu segmentation lên ảnh; validate shape khớp và alpha ∈ [0,1]; float image phải ∈ [0,1]; không mutate input |
| `segmentation.py` | `create_segmentation_legend()` | Figure legend: class name + ID + màu (matplotlib) |
| `depth.py` | `normalize_depth_for_visualization(depth)` | float32 `[0,1]`; mask NaN/inf; map constant → zeros; empty → zeros; không mutate input |
| `depth.py` | `colorize_depth(depth, colormap="turbo")` | RGB `uint8`; giữ `larger = closer` (giá trị lớn = màu ứng với vị trí cao hơn trong colormap) |
| `depth.py` | `create_depth_figure(depth, title=None)` | Figure + colorbar có label `COLORBAR_LABEL = "Relative inverse depth (larger = closer)"` |
| `fusion.py` | `create_fusion_overlay(image, seg, depth, alpha_seg=0.5, alpha_depth=0.35)` | Blend segmentation overlay + depth colormap lên ảnh; validate shape/alpha; không biểu diễn depth như mét |
| `fusion.py` | `create_region_visualization(seg, depth, region_map)` | Region map near/middle/far thành màu cố định, đổ bóng theo normalized depth (near sáng hơn) |
| `scene.py` | `create_scene_summary(report)` | Figure: semantic distribution (barh), depth distribution (bar), region summary, traffic context, interpretation; key thiếu → bỏ qua, không crash |
| `scene.py` | `save_scene_report(report, path)` | JSON UTF-8 `indent=2`, bắt buộc `.json`, không mutate report |
| `scene.py` | `create_full_visualization(image, seg, depth, fusion_result=None, region_map=None)` | Figure ≥ 4 panel: Input Image, Semantic Segmentation, Relative Inverse Depth, Segmentation + Relative Depth (+ Depth Regions, Scene Report); title depth nói rõ relative inverse depth |
| `io.py` | `save_visualization(image, path)` | Save RGB ảnh PNG/JPG; tự tạo parent dir; không mutate source; không save nếu caller không yêu cầu |
| `io.py` | `save_figure(fig, path, dpi=150)` | Save matplotlib Figure PNG/JPG |

### 4.3 Configuration

Theo `configs/pipeline.yaml` (đã có `visualization` section):

```yaml
visualization:
  enabled: true
  output_dir: outputs/visualization
  alpha_segmentation: 0.5
  alpha_depth: 0.35
```

Giá trị được dùng bởi `scene_understanding/pipeline.py._write_visualizations()` qua
`_alpha_values()` (defaults `0.5` / `0.35` khi thiếu config). Test `test_visualization_config_group`
xác nhận đúng các giá trị trên, đồng thời các nhóm config khác (vd `fusion.roi_classes`) còn nguyên.

### 4.4 Processing / Post-processing

- Normalization depth chỉ trên **bản sao** (`normalize_depth_for_visualization` trả mảng mới);
- Overlay dùng công thức `base·(1−α) + overlay·α`, clip về `[0,255]`, ép `uint8`;
- `create_region_visualization`: màu vùng cố định (near đỏ `(255,60,60)`, middle vàng `(255,200,0)`,
  far xanh `(60,140,255)`, invalid đen) nhân với độ sáng `0.55 + 0.45·depth_norm` để near đọc sáng hơn;
- Full visualization: 4 panel bắt buộc + panel tùy chọn (Depth Regions, Scene Report), sắp lưới 4 cột;
  depth panel dùng `cmap="turbo"`, norm `[0,1]`, colorbar label relative inverse depth.

### 4.5 Error Handling

Theo `prompts/10_visualization.md` §14 — input invalid phải `raise ValueError` rõ ràng, không silent failure.
Xác nhận từ code:

- segmentation không phải 2D → `ValueError("...2D [H, W]...")`;
- segmentation không integer → `ValueError`;
- class ID âm hoặc vượt palette → `ValueError`;
- image sai shape (không `[H,W,3]`) → `ValueError`;
- shape mismatch image/segmentation/depth/region → `ValueError`;
- alpha / alpha_seg / alpha_depth ngoài `[0,1]` → `ValueError`;
- colormap không tồn tại → `ValueError` (`unknown matplotlib colormap`);
- region map chứa code lạ → `ValueError` (`unknown code(s) ...`);
- extension ảnh không phải PNG/JPG → `ValueError`;
- scene report không phải dict / không `.json` → `TypeError` / `ValueError`;
- `save_figure` nhận non-Figure → `TypeError`.

---

## 5. SYSTEM BUILD FLOW

### 5.1 Step-by-Step Workflow

1. Đọc các contract hiện có (`models/unet/inference.py`, `models/midas/inference.py`,
   `evaluation/segmentation_metrics.py`, `evaluation/depth_metrics.py`,
   `scene_understanding/fusion.py`, `scene_understanding/analyzer.py`, các `configs/*.yaml`);
2. Tạo package `visualization/` (segmentation → depth → fusion → scene → io);
3. Thêm section `visualization` vào `configs/pipeline.yaml` (nếu chưa có);
4. Tích hợp vào `scene_understanding/pipeline.py._write_visualizations()` — sinh 4 ảnh + scene report;
5. Tạo `tests/test_visualization.py` (offline, headless, backend Agg);
6. Chạy `pytest` toàn bộ suite.

### 5.2 Integration

- `scene_understanding/pipeline.py:215 _write_visualizations()` ghi:
  `scene_segmentation.png`, `scene_depth.png`, `scene_fusion.png`, `scene_overview.png`,
  `scene_report.json` vào `output_dir` (mặc định `visualization.output_dir` từ config);
- `main.py`:
  - `--image` path: dùng default `load_config("pipeline")["visualization"]["output_dir"]` khi không có `--output-dir`;
  - cờ `--no-visualization`: skip ghi ảnh nhưng **vẫn ghi scene-report JSON**;
  - mode batch (`--input-dir`) gọi `run_custom_demo(..., save_visualizations=not args.no_visualization)`
    với output mặc định `outputs/custom/` (theo README).

### 5.3 Testing

`tests/test_visualization.py` — 52 tests, chạy hoàn toàn offline:

- Segmentation: shape/dtype, deterministic palette, đủ 19 class, ignore index, overlay shape, alpha validation, không mutate input;
- Depth: range `[0,1]`, giữ thứ tự lớn/nhỏ, NaN/inf/constant/empty handling, không mutate, colorize shape/dtype, figure label "Relative inverse depth";
- Fusion: output shape, depth thông tin xuất hiện khi alpha_depth > 0, alpha validation, region visualization + invalid black + unknown code reject, không mutate;
- Scene report: minimal report, missing keys không crash, JSON roundtrip, UTF-8 không escape, không mutate, validate extension/type;
- Save utilities: PNG/JPG + parent dirs, grayscale replicate, bad extension, non-Figure;
- Full visualization: synthetic image `H=128, W=256`, segmentation nhiều class, depth background nhỏ / foreground lớn; kiểm tra titles và lưu headless (Agg);
- Config: `visualization.enabled/output_dir/alpha_*` đúng, các nhóm khác nguyên vẹn.

Kiểm tra thực tế: `pytest tests/test_visualization.py` → **52 passed trong 3.01s**.

### 5.4 CI/CD

**CI/CD không được thiết lập trong repository** — không có thư mục `.github/workflows`,
không có config gitlab-ci hay workflow file nào. Kiểm thử được chạy thủ công qua `pytest`.
Full suite hiện tại: **456 passed, 1 skipped** (test CUDA path skip khi không có GPU), 22.80s.

---

## 6. EVALUATION

Step 10 **không đánh giá model** — visualization không tự sinh metric hiệu năng.
Phần này tương ứng với việc **kiểm chứng (validation) lớp visualization** và giữ đúng phân biệt
model-level / pipeline-level của project.

### 6.1 Evaluation Procedure

- Test-driven: 52 unit tests trên synthetic fixtures (không dùng model thật, không dataset);
- Chạy toàn bộ pytest suite đảm bảo **không phá vỡ** các test cũ trước Step 10;
- Kiểm tra figure title / colorbar label để đảm bảo semantic depth "relative inverse", không "meters".

### 6.2 Metrics

Không có metric định lượng mới (không phải bước đánh giá). Tiêu chí đạt:

- RGB `uint8` output đúng shape;
- palette deterministic;
- depth figure label = "Relative inverse depth (larger = closer)";
- scene report JSON hợp lệ, UTF-8, indent=2;
- không mutate bất kỳ input nào.

### 6.3 Results

| Hạng mục | Kết quả verification |
|---|---|
| Visualization tests | **52 passed** (3.01s) |
| Full test suite | **456 passed, 1 skipped** (22.80s) |
| Scene report smoke (`outputs/visualization/smoke/scene_report.json`) | JSON hợp lệ, key chuẩn, depth convention đúng |
| Full visualization smoke (`outputs/visualization/smoke/scene_*.png`) | Có segmentation/depth/fusion/overview ảnh |

### 6.4 Interpretation

- Visualization **đọc được và hiển thị đúng** các output của model mà **không can thiệp** vào
  chúng (các test "no mutation" pass);
- Depth luôn được trình bày là **relative inverse depth** (larger = closer), không bao giờ là meters;
- **KHÔNG** có kết quả đánh giá hiệu năng U-Net/MiDaS mới trong bước này — mọi số model-level
  phải lấy từ `outputs/analysis/*.json` như đã ghi ở Step 06, không thiết kế lại ở Step 10.

---

## 7. AI-ASSISTED DEVELOPMENT

### 7.1 Prompt

- Prompt file: `prompts/10_visualization.md` — mô tả chi tiết API, contract, tests, deliverable;
- Prompt xác định rõ **phạm vi** (chỉ visualization), **cấm** đổi logic model/metric/fusion/analyzer,
  cấm dataset/weights/internet trong quá trình phát triển.

### 7.2 Generated / Modified Scripts

- **Tạo mới**:
  - `visualization/__init__.py`, `visualization/segmentation.py`, `visualization/depth.py`,
    `visualization/fusion.py`, `visualization/scene.py`, `visualization/io.py`;
  - `tests/test_visualization.py`.
- **Modified**:
  - `configs/pipeline.yaml` — thêm section `visualization` (`enabled`, `output_dir`,
    `alpha_segmentation`, `alpha_depth`);
  - `scene_understanding/pipeline.py` — `_write_visualizations()` gọi các hàm visualization;
  - `main.py` — cờ `--no-visualization`, wire `visualization.output_dir` làm output mặc định.
- Git history: các file visualization tích hợp vào commit `47678b0` ("Steps 12-17: unet/midas
  evaluation, pipeline fusion, custom demo, report").

### 7.3 Testing

- Viết trước 52 test theo đúng chỉ dẫn từng chức năng trong prompt §11;
- Kiểm thử hoàn toàn offline, headless (MPLBACKEND=Agg);
- Chạy full suite để bảo đảm regression zero: **456 passed, 1 skipped**.

### 7.4 Validation

- Các test về deterministic palette, depth normalization edge cases (NaN/inf/constant), region
  unknown code, JSON UTF-8 roundtrip, no-mutation — tất cả pass;
- Đối chiếu config `visualization` trong `configs/pipeline.yaml` với prompt §10: khớp
  (`enabled: true`, `output_dir: outputs/visualization`, `alpha_segmentation: 0.5`, `alpha_depth: 0.35`);
- Không có dependency mới.

---

## 8. DISCUSSION

### 8.1 Strengths

- **Minh bạch và tái lập** — palette deterministic, không random;
- **An toàn với model output** — toàn bộ thao tác read-only, không mutate input (có test);
- **Đúng semantic depth** — không bao giờ gọi depth là meters; colorbar label rõ ràng;
- **Offline/headless** — test chạy không cần GPU, dataset hay mạng;
- **Tích hợp sẵn** — pipeline sinh đủ 4 ảnh + JSON, có `--no-visualization` flag;
- **Error handling rõ ràng** — ValueError/TypeError thay vì silent failure.

### 8.2 Weaknesses

- Fusion overlay là **hỗn hợp màu đơn giản** (blend) — nhìn được tổng thể nhưng khó đọc riêng
  từng chi tiết depth khi alpha_depth nhỏ;
- Depth bar (scene summary) chỉ theo region tổng, không histogram chi tiết;
- Visualizations chưa được quản lý bởi một pipeline "rendering" chuẩn — phụ thuộc vào
  `_write_visualizations()` trong `scene_understanding/pipeline.py`.

### 8.3 Limitations

- Visualization **không thêm thông tin gì vào output** — chỉ đổi dạng biểu diễn;
- Chất lượng ảnh phụ thuộc chất lượng segmentation/depth (giới hạn đã nêu Step 08);
- Không có CI/CD để tự kiểm thử (mục 5.4);
- Smoke/visualization demo (vd `outputs/visualization/smoke/`) dùng dữ liệu mẫu — **không**
  phải kết quả đánh giá có ý nghĩa thống kê.

### 8.4 What Can / Cannot Be Concluded

**Can conclude:**
- Lớp visualization hoạt động đúng, xử lý đúng các trường hợp edge (NaN/inf/constant/missing keys);
- Output ảnh/JSON sinh đúng format và đúng semantic (relative inverse depth, deterministic palette);
- Pipeline end-to-end có thể sinh các file trực quan cho demo.

**Cannot conclude:**
- Không kết luận gì về chất lượng model từ các ảnh visualization riêng lẻ;
- Không khẳng định depth là metric depth/meters;
- Không khẳng định fusion/visualization cải thiện hiệu năng — bước này chỉ hiển thị, không đo.

---

## 9. CONCLUSION

Step 10 hoàn thành lớp **Visualization** cho project: từ segmentation map, relative depth map,
region map và scene report, các hàm trong `visualization/` (segmentation, depth, fusion, scene, io)
tạo ra RGB image và figure sẵn sàng lưu file, giữ nguyên toàn bộ convention của project
(19 Cityscapes train IDs, `255` = ignore, `larger inverse depth = closer`, region quantiles
`[0.3333, 0.6666]`). Lớp này được tích hợp vào `SceneUnderstandingPipeline` và CLI, được kiểm chứng
bởi 52 test offline; toàn bộ suite vẫn đạt **456 passed, 1 skipped**.

Điểm mấu chốt: visualization **chỉ đọc**, không sửa model/fusion/difficulty, và **không**
đổi MiDaS relative inverse depth thành metric depth. Kết quả trực quan sinh ra phục vụ demo,
báo cáo và kiểm tra bằng mắt — không phải là số liệu đánh giá mới.

---

## 10. REFERENCES

- `prompts/10_visualization.md` — yêu cầu Step 10 (scope, API, tests, deliverables).
- `visualization/{__init__,segmentation,depth,fusion,scene,io}.py` — implementation.
- `tests/test_visualization.py` — 52 test offline (synthetic fixtures).
- `configs/pipeline.yaml` — cấu hình `visualization` (enabled, output_dir, alphas); `fusion`,
  `analyzer`, `difficulty` hiện hữu.
- `scene_understanding/pipeline.py` — `_write_visualizations()` tích hợp visualization.
- `scene_understanding/fusion.py` — region codes (`REGION_NEAR=2`, `REGION_MIDDLE=1`,
  `REGION_FAR=0`), quantiles.
- `main.py` — CLI (`--image`, `--input-dir`, `--output-dir`, `--no-visualization`).
- `README.md` — cách chạy demo, output `outputs/custom/`, relative inverse depth.
- `outputs/visualization/smoke/scene_*.png` + `scene_report.json` — smoke output.
- `outputs/custom/pipeline_demo_00{1,2}_*` — visualization từ demo pipeline.
- `docs/coursework/00..09` — ngữ cảnh, đánh giá, discussion trước đó (đặc biệt Step 06
  cho số liệu model-level, Step 08 cho giới hạn).
- Git history — commit `47678b0` (Steps 12–17) chứa các file visualization.
