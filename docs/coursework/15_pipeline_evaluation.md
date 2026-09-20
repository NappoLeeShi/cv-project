# Step 15 — Full Pipeline Evaluation: Fusion + Scene Understanding

## 1. REQUIREMENT

### 1.1 Problem

Sau các bước:

- **Step 13** đánh giá model-level **U-Net** trên Cityscapes GT (mIoU, Pixel Accuracy, Dice).
- **Step 14** đánh giá model-level **MiDaS** trên KITTI GT (RMSE, MAE, AbsRel, δ1..δ3).

Hai dataset **Cityscapes và KITTI không được ghép cặp** với nhau. Do đó chưa tồn tại một
phép đánh giá **full-pipeline** chạy cùng lúc cả segmentation lẫn depth trên **cùng một ảnh
RGB**, rồi thực hiện fusion → scene analysis → difficulty analysis.

Vấn đề của Step 15: xây dựng một **full-pipeline evaluation/demo reproducible** đo lường cách
kết hợp các output segmentation + depth sẵn có để hiểu cảnh (scene understanding).

### 1.2 Objective

- Tạo `evaluation/evaluate_pipeline.py`: chạy **cùng một ảnh RGB** qua U-Net (segmentation) và
  MiDaS (relative inverse depth), align không gian, fuse hai output, chạy scene analyzer, tính
  difficulty indicators.
- Tạo `evaluation/difficulty_analysis.py`: **pipeline difficulty score** minh bạch, dạng
  rule-based (weighted sum của các indicator chuẩn hoá), phân loại `easy / medium / hard`.
- Tái sử dụng các module sẵn có: `models/unet/`, `models/midas/`,
  `scene_understanding/fusion.py`, `scene_understanding/analyzer.py`,
  `scene_understanding/pipeline.py`, `visualization/`, `evaluation/`.
- Dùng checkpoint sẵn có: `checkpoints/unet_cityscapes.pth`,
  `checkpoints/dpt_large_384.pt`.
- Ghi JSON kết quả `outputs/analysis/pipeline_evaluation.json` + visualization files.

### 1.3 Input / Output

| Input | Output |
|---|---|
| `data/pipeline/images/` — ảnh RGB street (standalone) | `outputs/analysis/pipeline_evaluation.json` |
| `checkpoints/unet_cityscapes.pth` | `outputs/segmentation/*_original.png`, `*_segmentation.png` |
| `checkpoints/dpt_large_384.pt` | `outputs/depth/*_depth.png` |
| `configs/pipeline.yaml` | `outputs/analysis/*_fusion.png`, `*_overview.png`, `*_scene_report.json` |

### 1.4 Scope

- Đánh giá **full-pipeline cùng một ảnh**: image → segmentation → inverse depth → fusion →
  scene analysis → difficulty.
- Hai loại đánh giá tách biệt (theo prompt):
  - **A. Same-image pipeline demo/evaluation** trên tập ảnh đứng riêng (`data/pipeline/images`).
  - **B. Model-level evaluation vẫn tách riêng** — giữ nguyên kết quả Step 13 (U-Net/Cityscapes)
    và Step 14 (MiDaS/KITTI), **không bao giờ** trộn hai dataset thành một paired benchmark.
- Chạy cả hai model ở `eval()` / `no_grad`.
- SMOKE TEST với `--limit 2` trước khi mọi đánh giá lớn hơn.

### 1.5 Out of Scope

Theo prompt Step 15, **KHÔNG** được:

- Retrain U-Net, train MiDaS, thiết kế lại model sẵn có.
- Pair một ảnh Cityscapes với một depth map KITTI.
- Mô tả raw MiDaS output là meters (chỉ là relative inverse depth).
- Tự tính metric depth tuyệt đối khi chưa có conversion được hỗ trợ rõ ràng.
- Bịa ra "ground-truth difficulty" — difficulty chỉ là rule-based score minh bạch.
- Tự động chạy full benchmark pipeline lớn mà không được yêu cầu.
- Thay đổi các module `scene_understanding/`, `models/`, `visualization/` — chỉ reuse.

---

## 2. PURPOSE & SURVEY

### 2.1 Purpose

- Đo xem các output segmentation và depth **sẵn có** kết hợp được với nhau như thế nào để thể
  hiện scene understanding trên ảnh street thực (standalone).
- Nối kết Step 13 + Step 14 thành một pipeline thống nhất: U-Net (semantic) ─ MiDaS (depth
  relative) ─ fusion (analytical) ─ scene analysis ─ difficulty.
- Cung cấp khung đánh giá **reproducible**: JSON + visualization + CLI + tests.

### 2.2 Technical Background

- **U-Net** (`models/unet/inference.py`): `UNetInference` wrapper nhận ảnh RGB → mask trainId
  `[H,W]`; hỗ trợ `predict(image, return_confidence=True)` trả về cả class map lẫn softmax
  confidence (dùng cho difficulty indicator "segmentation uncertainty").
- **MiDaS DPT-Large** (`models/midas/inference.py`): `MidDepthPredictor` → **relative inverse
  depth**; `larger = closer`, smaller = farther; **không phải meters**. Median scaling là phép
  alignment evaluation-time của Step 14, không áp dụng vào raw demo depth ở Step 15.
- **Fusion** (`scene_understanding/fusion.py`): rule-based `fuse()` → `FusionResult`
  (per-class stats, depth region terciles, analyzed mask loại pixel non-finite / void).
- **Scene analysis** (`scene_understanding/analyzer.py`): `analyze_fusion()` → structured report
  (semantic distribution, depth distribution, per-class regions, traffic context,
  interpretation).
- **Depth regions** mặc định là terciles tự suy từ chính prediction: near / middle / far.
- **Difficulty score**: weighted sum các indicator đã chuẩn hoá `[0,1]`, một **pipeline
  difficulty score / scene complexity score**, KHÔNG phải ground-truth difficulty.

### 2.3 Technology / Method Survey

| Thành phần | Vai trò |
|---|---|
| `evaluation/evaluate_pipeline.py` | Orchestrator + CLI (577 dòng) |
| `evaluation/difficulty_analysis.py` | Rule-based difficulty score (222 dòng) |
| `evaluation/pipeline_metrics.py` | File placeholder (0 byte, chưa có nội dung) |
| `scene_understanding/fusion.py` | `fuse()` → `FusionResult` (reuse) |
| `scene_understanding/analyzer.py` | `analyze_fusion()` → scene report (reuse) |
| `scene_understanding/pipeline.py` | Helpers alignment `_resize_nearest_ids`, `_resize_bilinear` (reuse) |
| `models/unet/inference.py` | `UNetInference` (reuse) |
| `models/midas/inference.py` | `MidDepthPredictor` (reuse) |
| `visualization/*.py` | `colorize_segmentation`, `colorize_depth`, `create_fusion_overlay`, `create_full_visualization`, `save_scene_report`, `save_visualization`, `save_figure` |
| `configs/pipeline.yaml` | Config fusion + analyzer + visualization + difficulty + output dirs |
| `tests/test_evaluate_pipeline.py` | 42 test offline/mock (582 dòng) |

### 2.4 Why This Approach?

- **Reuse** tối đa thay vì duplicating: model loading, fusion, analyzer, visualization, alignment
  helpers đều dùng module sẵn có.
- **Same-image contract**: đúng contract của prompt — mỗi ảnh chỉ đọc 1 lần, đối tượng ảnh giống
  hệt được đưa cho cả U-Net và MiDaS, fusion chỉ làm việc trên output của cùng 1 ảnh.
- **Difficulty minh bạch**: không bịa ground-truth; mọi indicator đều có định nghĩa rõ, weight
  và ngưỡng configurable (`difficulty.factor_weights`, `difficulty.bins`).
- **An toàn GPU**: U-Net bị cap inference size xuống nhỏ nhất trong `configs/unet.yaml`
  (`_safe_unet_inference_size`) để giữ headroom cho MiDaS trên GPU 4 GB; MiDaS bị khoá input
  `preprocessing.input_size: 384` rồi resize về resolution gốc. Toàn bộ thuộc existing behavior.

---

## 3. DATA

Trong repo đã tồn tại thư mục demo-input `data/pipeline/images/` gồm **4 ảnh RGB street**:

| File | Kích thước (pixel) |
|---|---|
| `gta5.png` | 1600 × 900 |
| `pipeline_demo_001.png` | 1216 × 352 |
| `pipeline_demo_002.png` | 1216 × 352 |
| `streetest.png` | 3000 × 2000 |

- Các ảnh này là **standalone demo images** — KHÔNG ghép cặp với GT của Cityscapes/KITTI.
- Không nhân bản toàn bộ Cityscapes/KITTI; chỉ dùng tập nhỏ này cho same-image demo.
- Final JSON chỉ đánh giá **2 ảnh** (`pipeline_demo_001.png`, `pipeline_demo_002.png`) —
  đúng tinh thần smoke test với `--limit 2`; chưa chạy benchmark pipeline quy mô lớn.
- No dataset GT được dùng trong Step 15 pipeline (model-level GT vẫn nằm riêng ở Step 13/14).

---

## 4. IMPLEMENTATION PLAN

### 4.1 Architecture

```
data/pipeline/images/{image}.png
        │  (SAME image object)
        ├──────────────► U-Net  ──────────► [H,W] trainId mask (+ confidence)
        │
        └──────────────► MiDaS  ──────────► [H,W] relative inverse depth
                                 │
                                 ▼
       alignment về resolution gốc (nearest cho class IDs, bilinear cho depth)
                                 │
                                 ▼
       fusion (scene_understanding.fusion.fuse) → FusionResult
                                 │
                                 ▼
       scene analysis (scene_understanding.analyzer.analyze_fusion)
                                 │
                                 ▼
       difficulty (evaluation.difficulty_analysis.compute_difficulty)
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   pipeline_evaluation.json              visualization files
```

### 4.2 Components

| File / hàm | Chức năng |
|---|---|
| `evaluation/evaluate_pipeline.py` | Module chính: load config+models, loop ảnh, aggregate, CLI |
| `_safe_unet_inference_size` | Chọn inference resolution U-Net nhỏ nhất (budget GPU) |
| `_load_unet` | `UNet(num_classes=19)` + `UNetInference` với checkpoint |
| `_load_midas` | `torch.hub.load(MODEL_SOURCE, "DPT_Large", pretrained=False)` + `_load_weights` + `MiDaSModel` + `MidDepthPredictor(input_size=384)` |
| `collect_images` | List ảnh hợp lệ (.png/.jpg/.jpeg/.bmp/.tiff), support `limit` |
| `evaluate_single_image` | Pipeline per-image: seg + depth + align + fuse + analyze + difficulty |
| `_save_visualizations` | Lưu original/segmentation/depth/fusion/overview/scene_report |
| `_aggregate_results` | Thống kê toàn tập: avg score, easy/medium/hard counts, avg confidence, avg depth variation, class occurrence |
| `run_evaluation` | Entry point chính (CLI + test); ghi JSON |
| `build_parser` / `main` | CLI `python -m evaluation.evaluate_pipeline ...` |
| `evaluation/difficulty_analysis.py` | `compute_difficulty`, `DEFAULT_WEIGHTS`, `DEFAULT_BINS`, 5 indicator helpers, `DifficultyResult` |

### 4.3 Configuration

`configs/pipeline.yaml` (sẵn có, Step 15 dùng):

- `difficulty.factor_weights: [0.25, 0.20, 0.20, 0.20, 0.15]` — weight 5 indicator
  (tổng = 1.0: **segmentation_uncertainty 0.25, depth_variation 0.20, scene_complexity 0.20,
  foreground_fraction 0.20, object_density 0.15**).
- `difficulty.bins: {easy: 0.4, medium: 0.7}` → score ≤ 0.4 = easy; ≤ 0.7 = medium; else hard.
- `fusion.roi_classes: [11..18]` (person..bicycle), `drivable_classes: [0,1]` (road, sidewalk);
  `region_quantiles: [0.3333, 0.6667]` → far/middle/near terciles.
- `visualization.alpha_segmentation: 0.5`, `alpha_depth: 0.35`.
- `output.*_dir`: `outputs/segmentation`, `outputs/depth`, `outputs/analysis`.
- `system.seed: 42`, `device: auto`.

CLI defaults: `--config pipeline`, `--input-dir` bắt buộc, `--device auto`,
`--unet-checkpoint checkpoints/unet_cityscapes.pth`, `--midas-weights checkpoints/dpt_large_384.pt`,
`--limit`/`--save-visualizations`/`--output` tùy chọn.

### 4.4 Processing Flow

1. `load_config("pipeline")`; `resolve_device(device)`; `setup_logging`; `set_seed(42)`.
2. `resolve_path` checkpoint; log journal.
3. Load `UNetInference` (dùng `load_config("unet")` cho budget size) và `MidDepthPredictor`
   (dùng `load_config("midas")`, input_size 384).
4. `collect_images(input_dir, limit)`.
5. Với mỗi ảnh: mở RGB → `seg_out = seg_predictor.predict(np_image, return_confidence=True)`
   (fallback `predict(np_image)` khi không hỗ trợ); `depth_pred =
   depth_predictor.predict(np_image)` trong `no_grad`; tách segmentation/depth ndarray.
6. Align: `_resize_nearest_ids(segmentation, image_size)`; `_resize_bilinear(depth, image_size)`.
7. `fr = fuse(segmentation, depth)`; `scene_report = analyze_fusion(fr, cfg=cfg)`.
8. `diff = compute_difficulty(...)` với weight/bins từ config (hoặc default); `mean_confidence`
   từ confidence tensor (nếu có, nếu không → None → indicator trung tính 0.5).
9. Depth stats (mean/std/min/max của finite inverse depth) + seg stats
   (`num_classes_present`, `mean_confidence`).
10. Nếu `--save-visualizations`: lưu 6 loại file cho mỗi ảnh.
11. Strip `_internals`; `_aggregate_results`; ghi `pipeline_evaluation.json`
    (`json.dumps(indent=2, ensure_ascii=False)`).

### 4.5 Error Handling

| Tình huống | Xử lý |
|---|---|
| Không khởi tạo được DPT_Large từ torch.hub | `MiDaSError` |
| Bhông tìm thấy ảnh trong input dir | Warning `No images found in ...`, trả `{per_image: [], aggregate:{num_images: 0}}` |
| `predict(..., return_confidence=True)` không hỗ trợ | `TypeError` → fallback `predict(image)`, `mean_confidence=None` |
| Không có finite depth | depth stats rỗng (chỉ ghi nếu có giá trị) |
| Confidence tensor rỗng/non-finite | Bỏ qua; `mean_confidence` giữ None nếu không có giá trị |
| GPU OOM / lỗi pipeline | `main()` bọc try/except, in `error: ...` ra stderr, trả exit 1 |
| Input không hợp lệ về sau | `main()` in `error` và exit 1 |

---

## 5. SYSTEM BUILD FLOW

### 5.1 Workflow

1. Inspect `scene_understanding/{fusion,analyzer,pipeline}.py`, `models/unet/inference.py`,
   `models/midas/inference.py`, `visualization/`, `evaluation/`, `configs/pipeline.yaml`
   trước khi đổi gì — hiểu API, không suy đoán thay đổi kiến trúc.
2. Chạy test suite hiện có (gồm các test fusion/analyzer/pipeline) trước khi thêm mới.
3. Thêm `evaluation/evaluate_pipeline.py` + `evaluation/difficulty_analysis.py` + tests.
4. Smoke test real-input `--limit 2` (không chạy dataset lớn tự động).
5. Full regression `pytest -q`.

### 5.2 Integration

- **Reuse**: không viết lại model loading (dùng `UNetInference`, `MidDepthPredictor`,
  `_load_weights`), không viết lại fusion (dùng `fuse`), không viết lại scene analysis
  (dùng `analyze_fusion`).
- **Hub-name workaround**: MiDaS hub export `DPT_Large` (PascalCase) trong khi dự án chuẩn hoá
  `dpt_large`; `_load_midas` dựng backend trực tiếp bằng
  `torch.hub.load(MODEL_SOURCE, "DPT_Large", pretrained=False, trust_repo=True)` rồi bọc
  `MiDaSModel` — khớp key checkpoint, không nhân bản loading logic (cùng kỹ thuật Step 14).
- **Budget GPU**: U-Net chạy ở resolution nhỏ nhất config (`unet`), MiDaS ở `input_size=384`;
  sau đó cả hai prediction được resize về resolution gốc → alignment hợp lệ trên GPU 4 GB.
- **JSON artifact** `outputs/analysis/pipeline_evaluation.json` phục vụ Step 16+ và báo cáo.

### 5.3 Testing

`tests/test_evaluate_pipeline.py` — **42 test** offline (mock/synthetic, CPU-only, không cần
checkpoint thật / CUDA / 4 GB VRAM). 10 nhóm theo prompt:
cover: config loading; same-image contract; spatial alignment (dùng
`_resize_nearest_ids`/`_resize_bilinear` khi 2 kích thước khác nhau); fusion từ cùng ảnh;
difficulty deterministic; Easy/Medium/Hard thresholds; JSON schema (per-image + aggregate);
`--limit` (ý nghĩa giới hạn, `None` = tất cả, lớn hơn count, lọc extension); CPU-safe với
mocks + `no_grad`; **không ghép Cityscapes/KITTI** (pipeline chỉ một input dir, không import
Cityscapes trong `evaluate_single_image`); helper indicators và `_aggregate_results`.

Lưu ý: các test về fusion/scene analyzer/difficulty-results đã tồn tại trong
`tests/test_fusion.py`, `tests/test_scene_analyzer.py`, `tests/test_pipeline.py` (giữ nguyên,
không sửa).

### 5.4 CI/CD

**KHÔNG CÓ CI/CD** trong dự án — không tồn tại `.github/workflows`; mọi kiểm tra được chạy thủ
công qua pytest. Bước này không thêm CI/CD.

### 5.5 Execution / Commands

```bash
# 1) Full regression (offline, không cần model/download)
.venv/bin/python -m pytest -q

# 2) Smoke test real-input 2 ảnh
python -m evaluation.evaluate_pipeline \
    --config configs/pipeline.yaml \
    --input-dir data/pipeline/images \
    --limit 2 \
    --save-visualizations \
    --device auto

# 3) Options: --device cpu|cuda|auto; --limit N; --save-visualizations;
#    --output DIR; --unet-checkpoint PATH; --midas-weights PATH
```

---

## 6. EVALUATION

### 6.1 Evaluation Objective

- Chạy full-pipeline (seg + depth + fusion + scene + difficulty) trên ảnh demo standalone và
  đối chiếu output JSON; verify các con số trong `outputs/analysis/pipeline_evaluation.json`.
- Đây là **SMOKE TEST / full-pipeline demo** trên 2 ảnh thực (Json hiện có), KHÔNG phải
  benchmark thống kê, KHÔNG phải model-level benchmark.

### 6.2 Evaluation Method

- Pipeline đầy đủ trên mỗi ảnh; `_aggregate_results` tính trung bình toàn tập.
- Difficulty score:
  `difficulty_score = clamp( Σ (weight_i × indicator_i), 0, 1 )` với indicator chuẩn hoá
  `[0,1]`; phân loại `score ≤ 0.4 → easy`, `≤ 0.7 → medium`, else `hard`.
- Đối chiếu JSON hiện có: kiểm tra score từ components đúng công thức weight, và các aggregate
  đúng là trung bình của per-image.

### 6.3 Results

Nguồn: `outputs/analysis/pipeline_evaluation.json` (đã đọc và đối chiếu trực tiếp — mọi con số
khớp).

**Per-image (demo 001, demo 002):**

| Field | pipeline_demo_001.png | pipeline_demo_002.png |
|---|---|---|
| segmentation.num_classes_present | 19 | 18 |
| segmentation.mean_confidence | 0.646535 | 0.676068 |
| depth.mean_inverse_depth | 13.72307 | 13.27886 |
| depth.std_inverse_depth | 9.546155 | 8.537987 |
| depth.min_inverse_depth | 0.432997 | 0.808746 |
| depth.max_inverse_depth | 45.738678 | 32.905846 |
| difficulty.score | **0.541012** | **0.508277** |
| difficulty.level | medium | medium |

**Difficulty components (đối chiếu công thức):**

| Indicator (weight) | demo_001 | demo_002 |
|---|---|---|
| segmentation_uncertainty (0.25) | 0.353465 | 0.323932 |
| depth_variation (0.20) | 0.695628 | 0.642976 |
| scene_complexity (0.20) | 1.0 | 0.947368 |
| foreground_fraction (0.20) | 0.333335 | 0.333335 |
| object_density (0.15) | 0.312353 | 0.283717 |

→ `0.541012` đã được tính lại từ components với weights → khớp với score trong JSON
(`0.25 × 0.353465 + 0.20 × 0.695628 + 0.20 × 1.0 + 0.20 × 0.333335 + 0.15 × 0.312353`).

**Aggregate:**

| Field | Giá trị |
|---|---|
| num_images | 2 |
| average_difficulty_score | **0.524644** (khớp trung bình 2 score) |
| easy_count / medium_count / hard_count | **0 / 2 / 0** |
| average_segmentation_confidence | **0.661301** (khớp trung bình 2 confidence) |
| average_depth_variation | **9.042071** (khớp trung bình std inverse depth) |
| class_occurrence | 18 class xuất hiện trong cả 2 ảnh; `motorcycle` chỉ trong 1 ảnh (demo_001) |

**Scene report highlights (từ JSON):**

- Cả 2 ảnh: `height 352, width 1216`, `analyzed_pixels 428032`,
  `depth_convention "inverse_relative_larger_closer"`, depth terciles near/middle/far ≈ 0.3333.
- demo_001 `num_semantic_classes 19`, `object_class_pixel_ratio 0.312353`,
  `drivable_coverage_ratio 0.131747`, `dynamic_object_count 8`, `nearest_dynamic_class: car (near)`.
- demo_002 `num_semantic_classes 18`, `object_class_pixel_ratio 0.283717`,
  `drivable_coverage_ratio 0.170071`, `dynamic_object_count 7`, `nearest_dynamic_class: car (near)`.
- Mọi thuật ngữ depth đều là relative (larger = closer), không có metric meters.

**Visualization files (đều tồn tại trong repo, cho cả 2 ảnh demo):**

- `outputs/segmentation/pipeline_demo_00{1,2}_original.png`, `_segmentation.png`
- `outputs/depth/pipeline_demo_00{1,2}_depth.png`
- `outputs/analysis/pipeline_demo_00{1,2}_fusion.png`, `_overview.png`, `_scene_report.json`

**Tests:**

| Test | Kết quả |
|---|---|
| `pytest tests/test_evaluate_pipeline.py tests/test_pipeline.py tests/test_scene_analyzer.py tests/test_fusion.py -q` | **117 passed** in 4.73s |
| `pytest -q` (full regression) | **456 passed, 1 skipped** in 22.79s |

### 6.4 Limitations

- Đây là **smoke test / demo pipeline** trên 2 ảnh; chưa phải benchmark thống kê; Easy/Medium/Hard
  hiện chỉ là 0/2/0 trên tập quá nhỏ.
- Difficulty là **rule-based heuristic** từ predicted outputs — không phải ground-truth difficulty,
  không so sánh chéo được giữa các setting.
- MiDaS chỉ cho relative inverse depth (unit-less); không có metric meters trong JSON.
- Raw MiDaS output không qua median scaling (median scaling chỉ là alignment evaluation-time của
  Step 14, không áp dụng vào demo này).
- Confidence cần predictor hỗ trợ `return_confidence=True` (UNetInference của dự án có); nếu
  không, `mean_confidence=null` và uncertainty indicator = 0.5 (trung tính).
- Dùng checkpoint thật (U-Net + MiDaS) trên GPU → kết quả phụ thuộc data và device; không đo FPS.

---

## 7. AI-ASSISTED DEVELOPMENT

### 7.1 Prompt

- File: `prompts/15_pipeline_evaluation.md` (STEP 15 — Full Pipeline Evaluation: Fusion +
  Scene Understanding). Yêu cầu 12 bước: inspect existing implementation; define pipeline
  evaluation (`evaluation/evaluate_pipeline.py`); bảo toàn depth semantics; difficulty
  analysis; dataset handling (hai loại A/B, không pair Cityscapes/KITTI); evaluation output
  (`pipeline_evaluation.json`); visual outputs; CLI; tests
  (`tests/test_evaluate_pipeline.py`); full regression; real-data smoke test `--limit 2`;
  documentation (`prompts/15_pipeline_evaluation.md`).

### 7.2 Generated / Modified Files

**Generated (Step 15):**

| File | Vai trò |
|---|---|
| `evaluation/evaluate_pipeline.py` | Full-pipeline orchestrator + CLI (577 dòng) |
| `evaluation/difficulty_analysis.py` | Rule-based difficulty scoring (222 dòng) |
| `tests/test_evaluate_pipeline.py` | 42 test offline/mock (582 dòng) |
| `outputs/analysis/pipeline_evaluation.json` | Kết quả 2 ảnh demo |

**Generated (cùng commit, cho pipeline chung):** `evaluation/custom_demo.py`,
`tests/test_custom_demo.py`, `data/pipeline/images/{gta5,pipeline_demo_001,pipeline_demo_002,streetest}.png`,
các visualization outputs, `outputs/visualization/smoke/*`.

**Modified:** `configs/pipeline.yaml` (thêm `difficulty`, `pipeline_eval`, `visualization`
block liên quan), `tests/test_config.py` + `tests/test_segmentation_metrics.py` (cập nhật);
`evaluation/` module khác giữ nguyên.
`evaluation/pipeline_metrics.py` = **file empty placeholder (0 byte)** — chưa có nội dung tính
pipeline metrics độc lập (hiện metrics do `segmentation_metrics.py`/`depth_metrics.py` các bước
khác đảm nhiệm).

### 7.3 Testing

- Focused: `pytest tests/test_evaluate_pipeline.py tests/test_pipeline.py
  tests/test_scene_analyzer.py tests/test_fusion.py -q` → **117 passed** in 4.73s.
- Full: `pytest -q` → **456 passed, 1 skipped** in 22.79s.
- Smoke real-input 2 ảnh → JSON hợp lệ, visualization tạo đủ, score deterministic.

### 7.4 Version History

| Commit | Nội dung |
|---|---|
| `47678b0` | "Steps 12-17: unet/midas evaluation, pipeline fusion, custom demo, report" — tạo `evaluation/evaluate_pipeline.py`, `evaluation/difficulty_analysis.py`, `tests/test_evaluate_pipeline.py`, `data/pipeline/images/*`, `docs/15-pipeline-evaluation.md`, visual outputs; sửa `configs/pipeline.yaml`, các test liên quan |

*(Không tạo commit mới trong bước viết báo cáo này — chỉ kiểm chứng git history.)*

### 7.5 Output

- `outputs/analysis/pipeline_evaluation.json` — per-image (2 ảnh) + aggregate.
- `outputs/analysis/pipeline_demo_00{1,2}_scene_report.json` — scene report per-image.
- `outputs/segmentation/…_original.png`, `…_segmentation.png`; `outputs/depth/…_depth.png`;
  `outputs/analysis/…_fusion.png`, `…_overview.png`.
- `outputs/visualization/smoke/*` — smoke visualization sẵn có từ pipeline (scene_segmentation,
  scene_depth, scene_fusion, scene_overview, scene_report).

---

## 8. DISCUSSION

### 8.1 Strengths

- **Reproducible**: CLI chuẩn (`--input-dir`, `--limit`, `--device`, `--save-visualizations`,
  `--output`, `--unet-checkpoint`, `--midas-weights`), JSON + visualization + tests.
- **Tôn trọng contract**: same-image contract được enforce trong `evaluate_single_image` và
  kiểm chứng bằng unit test; spatial alignment rõ ràng (nearest cho IDs, bilinear cho depth).
- **Reuse toàn diện**: fusion, analyzer, model loading, visualization đều dùng module sẵn có.
- **Difficulty minh bạch, configurable** với weights và bins từ `configs/pipeline.yaml`,
  mọi indicator chuẩn hoá về `[0,1]`, deterministic (unit tested).
- **An toàn GPU**: cố định kích thước inference của cả hai model (U-Net nhỏ nhất, MiDaS 384)
  → tránh OOM trên GPU 4 GB với ảnh lớn (như `streetest.png` 3000×2000).

### 8.2 Weaknesses

- `mean_confidence` chỉ có khi predictor hỗ trợ `return_confidence=True`; nếu không, indicator
  uncertainty = 0.5 (trung tính) → score kém nhạy.
- Difficulty là heuristic phụ thuộc dự đoán của model (không so sánh độ khó tuyệt đối được).
- Aggregate hiện rất nhạy với số lượng ảnh nhỏ (JSON chỉ có 2 ảnh demo).

### 8.3 Limitations

- Smoke test 2 ảnh, chưa phải benchmark thống kê; phân bố Easy/Medium/Hard (0/2/0) chỉ mang
  tính minh hoạ.
- MiDaS output là relative inverse depth; không sinh metric meters; không qua median scaling.
- Không có GT cho ảnh demo → không đo độ chính xác tuyệt đối của pipeline; class count/proximity
  mô tả dự đoán của model, không phải ground truth.
- Không đo FPS; không có CI/CD (dự án chạy pytest thủ công).

### 8.4 What This Step Does / Does Not Prove

**Does**: chứng tỏ hai model sẵn có chạy được trên cùng một ảnh RGB, fusion + scene analysis +
difficulty score sinh ra output JSON hợp lệ và visualization đầy đủ; tests đảm bảo
deterministic score, alignment, schema, limit, CPU-safe, không ghép cặp dataset.

**Does NOT**: không chứng minh pipeline "hiểu cảnh" đúng theo ground truth; không phải
benchmark thống kê; không so sánh difficulty chéo; không sinh metric depth tuyệt đối.

---

## 9. CONCLUSION

Step 15 đã:

- Xây dựng full-pipeline evaluation/demo qua `evaluation/evaluate_pipeline.py` (cùng 1 ảnh RGB →
  U-Net → MiDaS → fusion → scene analysis → difficulty) và rule-based difficulty
  `evaluation/difficulty_analysis.py`.
- Tạo 42 test offline (mock/synthetic, CPU) + smoke test real-input 2 ảnh → JSON hợp lệ.
- Kết quả smoke test trên `pipeline_demo_001.png` (score 0.541012) và `pipeline_demo_002.png`
  (score 0.508277) đều **medium**; aggregate avg difficulty 0.524644, medium 2/2.
- Full regression `pytest -q`: **456 passed, 1 skipped**.
- Output: `outputs/analysis/pipeline_evaluation.json` + đủ visualization files.
- Tôn trọng tách biệt dataset: không pair Cityscapes/KITTI; MiDaS vẫn là relative inverse depth;
  model-level metrics giữ nguyên ở Step 13/14.

---

## 10. REFERENCES

1. Prompt Step 15 — `prompts/15_pipeline_evaluation.md`
2. Orchestrator — `evaluation/evaluate_pipeline.py`
3. Difficulty scoring — `evaluation/difficulty_analysis.py`
4. Tests — `tests/test_evaluate_pipeline.py`
5. Fusion — `scene_understanding/fusion.py`
6. Scene analysis — `scene_understanding/analyzer.py`
7. Single-image orchestration helpers — `scene_understanding/pipeline.py`
8. U-Net inference — `models/unet/inference.py`
9. MiDaS inference — `models/midas/inference.py`
10. Visualization — `visualization/{depth,fusion,segmentation,scene,io}.py`
11. Config — `configs/pipeline.yaml`
12. Output kết quả — `outputs/analysis/pipeline_evaluation.json`
13. Smoke visualization — `outputs/visualization/smoke/*`
14. Demo input images — `data/pipeline/images/*.png`
15. Tài liệu kỹ thuật — `docs/15-pipeline-evaluation.md`
16. Báo cáo trước liên quan — `docs/coursework/13_evaluate_unet.md`,
    `docs/coursework/14_evaluate_midas.md`