# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

## Coursework Report — Step 05: System Build Flow

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Tài liệu này mô tả **luồng build của toàn hệ thống** (how the project is actually built, trained, tested, evaluated, and connected), với trọng tâm là deliverable của Step 05 — **U-Net Inference pipeline**.
> Nguồn sự thật: `prompts/05_unet_inference.md`, code thực tế (`models/unet/inference.py`, `training/*`, `scene_understanding/*`, `evaluation/*`, `configs/*.yaml`, `tests/*`), `README.md`, `main.py` và các doc 00–04.
> Kiến trúc chi tiết của U-Net đã nằm ở Step 04 — tài liệu này **không lặp lại**, chỉ trình bày cách các tầng được nối với nhau.

---

## 1. REQUIREMENT

### 1.1 Problem

Step 05 chịu trách nhiệm **U-Net inference pipeline**: nối ảnh RGB đến kết quả semantic segmentation.

```text
Cityscapes RGB image
        ↓
Preprocessing
        ↓
Trained U-Net checkpoint
        ↓
Inference
        ↓
Semantic segmentation prediction
        ↓
Prediction visualization/output
```

Ở bước này model U-Net (Step 04) chưa có ý nghĩa "chất lượng" — nó chỉ verify rằng **đường ống inference chạy đúng về mặt kỹ thuật**: đọc đúng checkpoint, preprocess đúng chuẩn chung, argmax ra class map, resize về đúng resolution gốc. Chất lượng thật sự chỉ có sau khi training/load checkpoint có nghĩa (Step 12/13) và MiDaS (Step 07/08).

### 1.2 Build Objectives

Mục tiêu của luồng build trong dự án (theo mức repository hỗ trợ):

- **Model preparation** — xây U-Net từ config (Step 04), load checkpoint an toàn hoặc tạo model không checkpoint để test.
- **Data preparation** — loader Cityscapes/KITTI của Step 03, `max_samples` cho development mode.
- **Configuration** — config YAML một nguồn sự thật (`configs/unet.yaml`, `midas.yaml`, `pipeline.yaml`), đọc bằng `load_config`.
- **Training** — `UNetTrainer` huấn luyện U-Net trên Cityscapes, chọn best model theo validation mIoU, lưu checkpoint (Step 12).
- **Inference** — U-Net inference (Step 05 — deliverable của doc này) và MiDaS inference (Step 07).
- **Testing** — pytest offline (synthetic fixtures, không cần dataset/download/GPU).
- **Evaluation** — đánh giá U-Net trên Cityscapes val, MiDaS trên KITTI val, pipeline trên ảnh street standalone (Step 13/14/15).
- **Integration into full pipeline** — U-Net + MiDaS → fusion → scene analysis → difficulty → visualization (Step 09–17).

### 1.3 Scope

**Step 05 bao gồm** (theo `prompts/05_unet_inference.md`):

- `models/unet/inference.py` — inference API của U-Net;
- `tests/test_unet_inference.py` — unit tests cho inference;
- cấu hình inference cần thiết (nhóm `inference` trong `configs/unet.yaml`).

**Step 05 KHÔNG thực hiện:** U-Net training, training loop, optimizer/scheduler, segmentation metrics, MiDaS, depth evaluation, fusion, scene analyzer, full pipeline, CI/CD. Những phần đó chỉ được **trình bày** ở đây như các mắt xích của hệ thống (với ghi chú bước triển khai), không phải deliverable của Step 05.

---

## 2. PURPOSE & BUILD FLOW

### 2.1 Purpose

Cần một luồng build có cấu trúc vì:

- Mỗi tầng (data → model → inference → eval) có **dependencies tách biệt**, built/test được **độc lập và offline**;
- Cấu hình nằm trong **config YAML**, không hard-code — thay model/dataset/device không cần sửa code;
- Inference (Step 05) là **cầu nối** giữa model và pipeline: pipeline sau này chỉ cần gọi `predict(image)` mà không biết bên trong là gì.

### 2.2 Overall Build Flow

Flow A-to-Z mà repository thực sự hỗ trợ:

```text
Dataset (Cityscapes / KITTI, Step 03)
  → Preprocessing (ImageTransform / dataset transforms)
  → Configuration (configs/*.yaml — một nguồn sự thật)
  → Model (U-Net từ config / MiDaS pretrained)
  → Training (U-Net) hoặc Pretrained Loading (MiDaS, không training)
  → Checkpoint (unet_cityscapes.pth / dpt_large_384.pt)
  → Inference (models/{unet,midas}/inference.py)
  → Same RGB image cho cả hai model
  → Fusion (rule-based) → Scene Analysis → Difficulty
  → Visualization (segmentation/depth/fusion/overview + scene JSON)
  → Full Pipeline (main.py / evaluation.evaluate_pipeline)
```

Mọi mắt xích trên đều tồn tại trong repository (code + tests verify từng mắt xích).

### 2.3 Component Responsibilities

| Component | Responsibility |
|---|---|
| `preprocessing/` | Dataset loaders Cityscapes (RGB + label trainId, resize bilinear cho image / nearest cho mask) và KITTI (depth mm → mét, valid mask); `ImageTransform` chuẩn hóa ảnh RGB (chuẩn ImageNet) dùng chung cho cả U-Net và MiDaS. |
| `configs/` | Cấu hình YAML theo nhóm (`unet`, `midas`, `pipeline`): tham số model/data/inference/training/eval/fusion/visualization. |
| `models/` | U-Net định nghĩa (Step 04) + U-Net inference (Step 05); MiDaS wrapper + inference (pretrained, relative depth). |
| `training/` | `UNetTrainer`: vòng lặp train/valid, tracking metric, lưu best checkpoint, history JSON; `train_unet.py` là CLI entry point. |
| `evaluation/` | Segmentation metrics (mIoU/pixel accuracy), depth metrics (AbsRel/RMSE/δ…), eval scripts cho U-Net/MiDaS/pipeline, difficulty analysis. |
| `visualization/` | Tô màu segmentation/depth, fusion overlay, overview figure, save scene JSON. |
| `scene_understanding/` | `fusion.py` (rule-based fusion), `analyzer.py` (scene report JSON), `pipeline.py` (orchestration, dependency injection). |
| `tests/` | pytest offline với synthetic fixtures — không cần dataset, pretrained weights, internet hay GPU. |

---

## 3. U-NET BUILD FLOW

### 3.1 Dataset Preparation

Cityscapes đi vào luồng U-Net qua `CityscapesDataset` (Step 03): loader trả mỗi sample `{image, label, image_path, label_path}` — `image` (ảnh RGB đã resize/normalize) và `label` (mask trainId `0..18`, `255` = ignore). Loader gắn cặp ảnh/label theo **stem** của tên file (không bao giờ ghép theo sorted list), tôn trọng `missing_policy`. `max_samples` giới hạn số cặp cho development mode.

### 3.2 Preprocessing

- **Input RGB:** `ImageTransform` (từ Step 03) resize ảnh bằng **bilinear** về `inference.image_size` (mặc định `[512, 1024]`), chuyển HWC→CHW, thêm batch dimension, normalize theo chuẩn của config (`normalization.mean/std` ImageNet). Có một hệ normalization duy nhất — Step 05 không tạo hệ thứ hai.
- **Label mask:** chỉ resize bằng **nearest-neighbour** (không bao giờ bilinear cho class ID).
- Quan trọng (spec Step 05): preprocessing label của Step 03 **không dùng cho input RGB**.

### 3.3 Model Initialization

U-Net được dựng từ config qua `UNet.from_config(config)` (đọc nhóm `model.num_classes/in_channels/base_channels`). Cho inference, `UNetInference` xây model từ `num_classes` khi không truyền model sẵn:

```python
inferencer = UNetInference(model=model, device=device)      # có model
inferencer = UNetInference(num_classes=19, device="cpu")    # không checkpoint (cho test)
inferencer = UNetInference.from_config(cfg, checkpoint=path) # từ YAML + checkpoint
```

Device theo chính sách chung (Step 02): `resolve_device("auto"|"cpu"|"cuda")`; không hard-code `.cuda()`; yêu cầu CUDA khi không có sẽ raise lỗi rõ ràng.

### 3.4 Training Flow

Training nằm ở Step 12 nhưng là mắt xích quan trọng của "build flow" — mô tả đúng code trong `training/train_unet.py` + `training/trainer.py`:

1. **Data loading:** `CityscapesDataset` (train) + (val), `DataLoader` với `shuffle=True` cho train, tuần tự cho val, seed generator, `pin_memory` khi CUDA.
2. **Forward pass:** `model.train()`; logits = `model(images)` trong `torch.autocast(float16)` nếu `mixed_precision` (CUDA-only).
3. **Loss:** `nn.CrossEntropyLoss(ignore_index=255)`; thành phần ảnh không hợp lệ (label 255) bị loại khỏi loss.
4. **Backward + optimizer update:** `float16` dùng `GradScaler.scale(loss).backward() → scaler.step(optimizer) → scaler.update()`; optimizer **Adam**. Batch dict `{image, label}` hoặc `(images, labels)` đều hiểu.
5. **Validation:** mỗi epoch chạy `model.eval()` + `no_grad`, tính loss, pixel accuracy và **mIoU** (tái sử dụng `evaluation/segmentation_metrics.py`, không cài lại công thức).
6. **Checkpoint handling:** model có validation mIoU **tốt hơn best** sẽ được lưu vào `checkpoints/unet_cityscapes.pth` với full state: `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_epoch`, `val_loss`, `val_miou`, `val_pixel_accuracy`, `num_classes`, `ignore_index`, `seed`, `config`. `--resume` đọc lại checkpoint; `--overwrite` cho phép ghi đè file best đã tồn tại. History JSON lưu per-epoch.
7. **Mixed precision:** mặc định tắt; bật bằng `--mixed-precision` hoặc `training.mixed_precision: true` (chỉ có tác dụng trên CUDA).

### 3.5 Checkpoint

Checkpoint đại diện cho **bộ trọng số tốt nhất (theo validation mIoU)** của U-Net. Định dạng chuẩn là dict chứa `model_state_dict` — đúng thứ `models/unet/inference.py` hiểu. Sau này được tái dùng để: inference (Step 05), evaluation (Step 13), và custom demo qua `main.py` (mặc định `checkpoints/unet_cityscapes.pth`). File hiện diện trong repo nhưng là sản phẩm của bước training (Step 12).

### 3.6 U-Net Inference

`models/unet/inference.py` — deliverable của Step 05:

```text
Input image (PIL / HWC RGB array / [3, H, W] tensor)
 → ImageTransform (resize bilinear theo inference.image_size + ImageNet normalize)
 → [1, 3, H', W'] trên device đã resolve
 → U-Net forward (model.eval(), torch.no_grad())
 → logits [1, C, H, W]
 → argmax(dim=1) → integer class map
 → (optional) nearest-neighbour resize về resolution nguồn
 → [H, W] torch.long, trainId 0..18
```

Điểm cốt lõi (đúng spec Step 05):

- **Checkpoint an toàn:** hỗ trợ cả hai format `state_dict` trần và dict có `model_state_dict`; validate đầy đủ (missing/unexpected/shape-mismatched keys) và **raise `InferenceError` rõ ràng** — không bao giờ dùng `strict=False` nuốt lỗi. Model tự dựng khi không có checkpoint (cho test).
- **Không gradient:** toàn bộ predict nằm trong `torch.no_grad()`; `model.eval()` đảm bảo.
- **Class prediction:** chỉ `logits.argmax(dim=1)`; **không sigmoid, không threshold**, không dùng softmax cho class index.
- **Resize:** class map resize về ảnh gốc bằng **nearest-neighbour** (`nearest-exact`); bilinear **cấm** cho class ID.
- **Confidence (optional):** `return_confidence=True` trả thêm bản đồ confidence = `max` của `softmax(logits, dim=1)`, resize về ảnh gốc bằng bilinear; softmax chỉ dùng cho confidence, không thay argmax.
- **Batch:** `predict_batch([B, 3, H, W]) → [B, H, W]`.

---

## 4. MIDAS BUILD FLOW

### 4.1 Pretrained Model Loading

MiDaS DPT-Large là model **pretrained** — dự án **không huấn luyện / fine-tune** nó:

- Kiến trúc được xây qua **torch.hub** (`intel-isl/MiDaS`, variant `dpt_large`) hoặc nạp weights từ checkpoint cục bộ `checkpoints/dpt_large_384.pt`;
- `build_midas_model` là entry point duy nhất có thể chạm torch.hub/cache; tạo `MiDaSModel` wrapper **không bao giờ download**;
- Trọng số MiDaS **không bao giờ được cập nhật** — chỉ `model.eval()` + `no_grad`.

### 4.2 Inference Flow

`models/midas/inference.py`:

```text
Input RGB (PIL / HWC array / [3, H, W] tensor)
 → MiDaS preprocessing (built-in ImageTransform hoặc official MiDaS transform, resize 384)
 → [1, 3, H', W'] trên device
 → DPT-Large forward (eval, no_grad)
 → raw inverse-relative depth map [B, H', W']
 → bilinear resize về resolution nguồn
 → [H, W] float32 relative depth
```

API đối xứng với U-Net: `MidDepthPredictor(model, device, input_size, transform)` + `predict(image)`, `predict_batch(...)`, `from_config(cfg)`.

### 4.3 Depth Semantics

Những câu khẳng định cố định của dự án (phải trình bày đúng):

- **Đầu ra là relative inverse depth** — dạng disparity-like, không có thang đo.
- **Giá trị LỚN hơn = GẦN camera hơn** (larger = closer).
- **KHÔNG phải depth metric theo mét** (`is_metric=False`, `metric_scale=None`); chuyển sang mét chỉ là bước alignment ở đánh giá (Step 08/14), không thuộc model inference.

> MiDaS **không được mô tả là training trong dự án**; nó chỉ được tích hợp (integration), không đào tạo lại.

---

## 5. FULL PIPELINE BUILD FLOW

### 5.1 Same-Image Contract

Pipeline dùng **cùng một object ảnh RGB** đưa vào cả hai predictor (`scene_understanding/pipeline.py` — "The SAME image object is passed to both predictors, honouring the same-image contract (never mix datasets)"). Cityscapes và KITTI **không bao giờ ghép cặp**; ảnh đưa vào pipeline chỉ là ảnh street standalone.

### 5.2 Parallel Model Outputs

```text
              RGB image
                │
   ┌────────────┴────────────┐
   │                         │
   ▼                         ▼
U-Net                     MiDaS
[1,3,H',W']               [1,3,H',W']
→ logits                  → raw inverse relative depth
→ argmax                  → bilinear resize
→ [H,W] trainId mask      → [H,W] relative depth  (larger = closer)
```

Cả hai output được căn chỉnh về **resolution nguồn của ảnh**: segmentation bằng nearest (class ID), depth bằng bilinear.

### 5.3 Fusion

`scene_understanding/fusion.py` — **rule-based/analytical**, không phải mạng nơ-ron fusion:

- Yêu cầu segmentation và depth **cùng kích thước không gian** (shape match) — chính là ràng buộc same-image;
- Pixel depth không hữu hạn (NaN/inf) và pixel void (`255`) bị loại khỏi mọi thống kê (qua `analyzed_mask`);
- **Per-class stats** cho mỗi class hiện diện: `pixel_count`, `pixel_ratio`, `mean_depth`, `median_depth`, `min_depth`, `max_depth`;
- **Depth regions:** ngưỡng suy ra từ chính depth map (default terciles `[0.3333, 0.6667]`): `near` = 1/3 giá trị lớn nhất, `middle`, `far` = 1/3 nhỏ nhất. Quy ước **larger inverse depth = closer**. Ngưỡng không phải mét;
- Kết quả là `FusionResult` (seg mask, depth, valid/analyzed mask, per_class, region_map, region_summary, thresholds).

### 5.4 Scene Analysis

`scene_understanding/analyzer.py` chuyển `FusionResult` thành **scene report JSON**:

- `scene` — kích thước, số class, depth convention;
- `semantic_distribution` — pixel count từng class;
- `depth_distribution` — tỉ lệ near/middle/far;
- `regions` — chi tiết per-class depth;
- `traffic_context` — vehicles, pedestrians, road, `drivable_coverage_ratio`, `drivable_median_depth`, `dynamic_object_count`, `nearest_dynamic_class`;
- `interpretation` — các câu mô tả tiếng Anh (vd *"The nearest dynamic class is 'car' (near relative-depth region)."*).

### 5.5 Difficulty Analysis

`evaluation/difficulty_analysis.py` — **heuristic score**, không phải ground-truth:

```text
difficulty_score = 0.25·U + 0.20·V + 0.20·C + 0.20·F + 0.15·O
```

trong đó `U` = segmentation uncertainty (`1 - mean_confidence`), `V` = depth variation (coefficient of variation), `C` = scene complexity (số class hiện diện / 19), `F` = foreground fraction (near-region ratio), `O` = object density (ROI-class ratio). Score bị clamp về `[0, 1]` và phân loại theo `difficulty.bins`: `score ≤ 0.4` → **easy**; `≤ 0.7` → **medium**; ngược lại → **hard**.

Không bịa thêm thuật toán nào — mọi công thức trên đều là code thật.

---

## 6. TESTING FLOW

### 6.1 Unit Tests

pytest là cổng kiểm tra của dự án (`README` ghi rõ "pytest suite with synthetic fixtures (no dataset downloads needed)"). Mỗi tầng đều có bộ test độc lập: config/device/seed/logger (Step 02), dataset (Step 03), model (Step 04), inference (Step 05), training, metrics, fusion, analyzer, pipeline, evaluation scripts.

### 6.2 Offline Testing

Mọi test dùng dữ liệu **synthetic/mocked**:

- Ảnh RGB nhân tạo `np.random.randint(...)` thay Cityscapes/KITTI;
- Model U-Net khởi tạo ngẫu nhiên (`num_classes`/`base_channels` nhỏ) — **không** cần pretrained weights;
- Dummy predictors (deterministic) cho pipeline — không cần model thật;
- Chạy trên **CPU**; test CUDA bị `skip` khi không có GPU.

### 6.3 Integration / Smoke Testing

- `tests/test_pipeline.py` chạy pipeline với dummy predictors: cùng ảnh vào cả hai predictor, alignment shape, fusion/analyzer chạy đúng, visualization tạo đúng số file;
- Smoke test thật (spec Step 05): load U-Net ngẫu nhiên + ảnh synthetic `(256, 512, 3)` → prediction `(256, 512)` integer — đã chạy xác minh thành công (xem mục 9);
- Import test: `from models.unet.model import UNet; from models.unet.inference import *` — **không lỗi**.

### 6.4 Regression Testing

pytest đảm bảo thay đổi sau không phá tầng trước: Step 05 không được gỡ/làm yếu test cũ, và kết quả kỳ vọng là *all Step 02-04 tests + new inference tests pass*. Test độc lập của từng bước chạy trong cùng một lần để phát hiện hồi quy.

**Số test đo được từ repository hiện tại:**

```text
.venv/bin/python -m pytest tests/ -q
456 passed, 1 skipped in 18.65s
```

- `tests/test_unet_inference.py` (Step 05): **30 passed, 1 skipped** (skip là CUDA).
- Toàn bộ regression: **456 passed, 1 skipped** (skip là CUDA) — đo chính xác ở thời điểm viết doc.

---

## 7. CI/CD

### 7.1 CI Purpose

CI có ích cho coursework này: tự chạy pytest mỗi khi push để bắt nhanh lỗi import, lỗi shape, hồi quy giữa các bước — không cần đụng model hay dataset.

### 7.2 Automated Checks

> **Repo không có cấu hình GitHub Actions / CI** (không tồn tại thư mục `.github/` hay workflow file). Vì vậy được ghi **trung thực**: chưa có CI tích hợp. Nếu sau này thêm, các check phù hợp với project gồm:
> - cài dependency (`pip install -r requirements.txt`);
> - `pytest` (bộ offline, CPU-only — chạy được trong CI);
> - import/syntax check (vd câu lệnh import Step 05);
> - ghi chú: không push formatting/lint vì repo chưa có tooling như `ruff`/`black`.

### 7.3 AI/ML CI Considerations

Những giới hạn thực tế liên quan đúng project nếu làm CI:

- **GPU-dependent model execution** — tầng model/training phụ thuộc CUDA; test trong CI chỉ nên chạy nhánh CPU-offline (bộ hiện tại làm được);
- **Large datasets** — Cityscapes/KITTI không download trong CI; test dataset dùng fixture synthetic;
- **Large checkpoints** — `checkpoints/unet_cityscapes.pth` (~373 MB) và `dpt_large_384.pt` (~1.37 GB) không nên nằm trong CI pipeline thường xuyên;
- **Expensive full evaluation** — đánh giá U-Net/MiDaS trên full val/train splits tốn thời gian và tài nguyên, không chạy trong CI thông thường.

---

## 8. IMPLEMENTATION COMMANDS

Tất cả lệnh dưới đây được **verify từ README / source docstring / config** — không đặt chữ chạy tưởng tượng.

**Testing (README):**

```bash
.venv/bin/python -m pytest -q
```

**Import test (spec Step 05):**

```bash
.venv/bin/python -c "from models.unet.model import UNet; from models.unet.inference import *; print('U-Net inference modules OK')"
```

**U-Net training (Step 12, `training/train_unet.py`):**

```bash
# mặc định: seed 42, batch 1 @ 256x512, Adam lr 1e-4, CE ignore 255, best theo val mIoU
.venv/bin/python -m training.train_unet --config unet

# development smoke-run: giới hạn số mẫu, thiết bị
.venv/bin/python -m training.train_unet --config unet \
    --train-samples 10 --val-samples 10 --device cpu

# tùy chọn đã hỗ trợ: --epochs --batch-size --learning-rate --image-size H W
#   --mixed-precision/--no-mixed-precision --num-workers --seed
#   --resume PATH --overwrite --output-dir PATH
```

**U-Net evaluation (Step 13, `evaluation/evaluate_unet.py`):**

```bash
.venv/bin/python -m evaluation.evaluate_unet
.venv/bin/python -m evaluation.evaluate_unet --config unet \
    --checkpoint checkpoints/unet_cityscapes.pth
# kết quả: outputs/analysis/unet_cityscapes_evaluation.json
# tùy chọn: --output PATH --split val --batch-size 1 --save-predictions
```

**MiDaS evaluation (Step 14, `evaluation/evaluate_midas.py`):**

```bash
.venv/bin/python -m evaluation.evaluate_midas
.venv/bin/python -m evaluation.evaluate_midas \
    --config midas --checkpoint checkpoints/dpt_large_384.pt
# kết quả: outputs/analysis/midas_kitti_evaluation.json (alignment median scaling)
```

**Pipeline execution (Step 15 / Step 17):**

```bash
# smoke 2 ảnh (README)
.venv/bin/python -m evaluation.evaluate_pipeline \
    --input-dir data/pipeline/images --limit 2 --save-visualizations --device auto

# custom-image demo (README, main.py)
.venv/bin/python main.py --input-dir data/pipeline/images
.venv/bin/python main.py --input-dir data/pipeline/images --limit 1

# single-image (main.py, backward-compatible)
.venv/bin/python main.py --image path/to/street.png \
    --unet-checkpoint checkpoints/unet_cityscapes.pth \
    --midas-weights checkpoints/dpt_large_384.pt \
    --device auto
```

`main.py` không bao giờ download; thiếu checkpoint sẽ **fail nhanh với lỗi rõ**.

---

## 9. BUILD VERIFICATION

Cách project xác minh từng tầng hoạt động (chỉ dùng cơ chế có thật trong repo):

| Stage | Verification |
|---|---|
| Data | `tests/test_cityscapes.py`, `tests/test_kitti.py` — ghép cặp theo stem, resize, normalize, valid mask; loaders self-contained (đã verify ở Step 03). |
| U-Net | `tests/test_unet.py` (17 test, Step 04): shape/class/resolution/odd-size/gradient. |
| U-Net inference | `tests/test_unet_inference.py` (30 passed + 1 CUDA skip): tạo wrapper, single/batch, class range, nhiều kích thước, không gradient, eval mode, load checkpoint 2 format, invalid checkpoint error, nearest resize, raw prediction integer. |
| MiDaS | `tests/test_midas.py` — wrapper, variants, predictor, checkpoint/weights safety, depth convention (larger = closer, not metric). |
| Fusion / Analyzer | `tests/test_fusion.py`, `tests/test_scene_analyzer.py` — shape validation, per-class stats, regions, analyzed mask, JSON report. |
| Pipeline | `tests/test_pipeline.py` (dummy predictors): same-image contract, alignment, fusion + analyzer chạy, visualization files. |
| Tests | `pytest` full: **456 passed, 1 skipped**; import test `U-Net inference modules OK`; smoke: prediction `(256, 512)`, min≥0, max≤18, `torch.long`. |

---

## 10. AI-ASSISTED DEVELOPMENT

### 10.1 Prompt

Step 05 được triển khai theo task brief `prompts/05_unet_inference.md` — quy định phạm vi (chỉ U-Net inference), checklist kỹ thuật (safety checkpoint, no-grad, eval mode, nearest resize, device policy, API sạch) và 12 test bắt buộc.

### 10.2 Generated/Modified Scripts

File verify được liên quan Step 05:

| File | Trách nhiệm |
|---|---|
| `models/unet/inference.py` | `UNetInference`, `load_model`, `InferenceError` — API inference U-Net |
| `tests/test_unet_inference.py` | 30+ test inference offline |
| `configs/unet.yaml` (nhóm `inference`) | `device`, `image_size`, `normalize` — cấu hình inference |

*Không có file ngoài những file trên được sinh/thay ở Step 05* (training, eval, MiDaS, fusion đều là các bước khác).

### 10.3 Testing

Code được validate bằng: full pytest (456 pass), import test U-Net inference (OK), smoke inference synthetic (đúng shape/dtype/class-range) — tất cả offline, CPU.

### 10.4 Version History

Git log có các commit gộp tiến độ như `add rp step 1-4`, `add rp step 5` (commit gộp tài liệu); **không có commit riêng biệt duy nhất cho implementation Step 05** để xác minh, nên không mô tả chi tiết commit nào cụ thể. Thông tin dừng ở mức verify được từ `git log`.

---

## 11. DISCUSSION

### 11.1 Strengths

- **API inference sạch, decoupled:** pipeline sau chỉ cần gọi `.predict(image)`; `UNetInference` làm việc với PIL/array/tensor, không dính logic filesystem Cityscapes.
- **Checkpoint safety đúng spec:** hỗ trợ 2 format, validate missing/unexpected/mismatched, raise `InferenceError` rõ ràng — không `strict=False` nuốt lỗi; `num_classes` lệch sẽ không load được.
- **Chuẩn hóa duy nhất:** `ImageTransform` dùng chung cho U-Net và MiDaS (cùng mean/std normalize), không hệ thống song song.
- **Resize đúng luật:** class map luôn nearest-neighbour; depth luôn bilinear; alignment về resolution nguồn bắt buộc, test kỹ (kể cả size lẻ `100×150`).
- **No-grad + eval mode + device policy** được đảm bảo ở inference; test xác minh.
- **Rule-based fusion transparent:** không mạng fusion học — mọi ngưỡng/thống kê đều suy ra minh bạch; difficulty score *không* phải ground-truth (tránh hiểu sai).

### 11.2 Weaknesses

- **Không có chất lượng ở Step 05:** model ngẫu nhiên → prediction không có nghĩa segmentation thật; chỉ verify kỹ thuật (đúng như spec yêu cầu — không bịa claim).
- **MiDaS hub naming issue:** upstream `intel-isl/MiDaS` hubconf giờ export `DPT_Large` (PascalCase), trong khi project normalize về `dpt_large`; `evaluate_midas.py` phải build backend trực tiếp qua `torch.hub.load(..., "DPT_Large", pretrained=False)` rồi load checkpoint — ghi nhận trong docstring module.
- **Inference xử lý từng ảnh tuần tự** trong custom demo; không có batching folder-level (dummy `--limit` giúp smoke nhanh nhưng không phải batch pipeline).

### 11.3 Limitations

- **GPU memory:** config ghi rõ GPU ~4 GB của môi trường bị OOM với `batch 4 @ 512×1024`; U-Net training mặc định `batch 1 @ 256×512` (< 1 GiB). Inference `image_size 512×1024` tốn VRAM tương ứng.
- **Dataset size:** training/eval yêu cầu Cityscapes (có sẵn trên máy, không download); full val split tốn thời gian — dùng `--train-samples/--val-samples` cho smoke.
- **Pretrained MiDaS:** weights lấy từ checkpoint pretrained (`dpt_large_384.pt`), không được đào tạo trong project; output relative, muốn mét cần alignment ở bước đánh giá.
- **Computational cost:** đánh giá full U-Net (Step 13) và MiDaS (Step 14) trên toàn split không nằm trong CI; là các lệnh chạy thủ công.

---

## 12. CONCLUSION

Tóm tắt A-to-Z để trình bày miệng:

*"Từ lúc có dataset đến lúc có kết quả Scene Understanding, điều gì xảy ra?"*

1. **Dataset vào hệ thống:** Cityscapes (ảnh + label trainId) và KITTI (depth mm → mét + valid mask) được nạp bằng loader Step 03, ghép cặp theo **stem**, không ghép theo danh sách.
2. **Preprocessing:** ảnh RGB qua `ImageTransform` (resize bilinear + chuẩn hóa ImageNet) — **một hệ chuẩn hóa duy nhất**; label chỉ resize bằng nearest.
3. **Cấu hình:** mọi tham số (model, data, inference, training, eval, fusion) nằm trong `configs/*.yaml`, đọc qua `load_config`.
4. **Model:** U-Net dựng từ config (Step 04); MiDaS DPT-Large pretrained (Step 07) — MiDaS **không training** trong project.
5. **Training (U-Net, Step 12):** `UNetTrainer` chạy forward → CE loss (ignore 255) → backward → Adam → validation mỗi epoch → lưu **best checkpoint theo val mIoU**.
6. **Checkpoint:** `unet_cityscapes.pth` (segmentation) và `dpt_large_384.pt` (depth) — định dạng `model_state_dict` an toàn cho inference.
7. **Inference (Step 05 là trọng tâm doc này):** ảnh → tensor `[1,3,H,W]` → `model.eval()` + `no_grad` → logits → `argmax` → class map trainId `0..18` → resize **nearest** về ảnh gốc; MiDaS tương tự nhưng output relative depth, resize bilinear.
8. **Cùng một ảnh đưa vào cả hai model** (same-image contract) — segmentation mask và depth cùng resolution nguồn.
9. **Fusion (rule-based):** gộp mask + depth → per-class stats + vùng near/middle/far (ngưỡng terciles suy từ chính depth).
10. **Scene Analysis:** analyzer biến FusionResult thành **scene report JSON** (semantic/depth distribution, traffic context, nearest dynamic class…).
11. **Difficulty:** score heuristic 5 chỉ số (weights `[0.25,0.20,0.20,0.20,0.15]`) → `easy/medium/hard`, **không phải ground-truth**.
12. **Visualization & output:** segmentation/depth/fusion/overview PNG + scene JSON → kết quả Scene Understanding hoàn chỉnh.

*Bonus — "làm sao biết hệ thống đúng?"*: mỗi mắt xích đều có unit test offline (synthetic, CPU, không download). Toàn bộ repo hiện **456 tests passed, 1 skipped** — đó là lưới an toàn để sau này thêm bước mà không phá bước cũ.

---

## 13. REFERENCES

- `prompts/05_unet_inference.md` — task brief Step 05.
- `prompts/00_project_architecture.md`, `01_architecture_specification.md` — kiến trúc tổng thể, quy ước config.
- `docs/coursework/00..04` — tài liệu các bước trước (thống nhất thuật ngữ khi nói U-Net, MiDaS, fusion).
- `README.md` — layout project, dataset constraints, test lệnh, demo lệnh.
- `models/unet/inference.py`, `models/unet/model.py` — U-Net inference + model.
- `models/midas/inference.py`, `models/midas/model.py` — MiDaS predictor + wrapper.
- `training/trainer.py`, `training/train_unet.py` — U-Net training loop và CLI.
- `scene_understanding/pipeline.py`, `fusion.py`, `analyzer.py` — pipeline orchestration, fusion, scene analysis.
- `evaluation/evaluate_unet.py`, `evaluate_midas.py`, `evaluate_pipeline.py`, `difficulty_analysis.py` — các entry point đánh giá và difficulty heuristic.
- `configs/unet.yaml`, `configs/midas.yaml`, `configs/pipeline.yaml` — cấu hình hệ thống.
- `tests/test_unet_inference.py` và full pytest suite — test Step 05 cùng regression.
- `main.py` — demo entry point full pipeline.
- `docs/15-pipeline-evaluation.md` — tài liệu pipeline evaluation (Step 15).