# BÁO CÁO BƯỚC 13 — ĐÁNH GIÁ U-NET TRÊN CITYSCAPES (STEP 13 — EVALUATE U-NET ON CITYSCAPES)

## 1. REQUIREMENT

### 1.1 Problem

Sau Step 12, checkpoint `checkpoints/unet_cityscapes.pth` đã được huấn luyện xong nhưng
chỉ mới có kết quả từ **vòng validation trong lúc training** (best epoch 20, val_mIoU 0.4262,
val_pixel_accuracy 0.9021) ghi trong `outputs/analysis/unet_training_history.json`.

Chưa có một bước **đánh giá độc lập, reproducible** chạy checkpoint đó trên **toàn bộ
500 ảnh validation** của Cityscapes bằng pipeline tái sử dụng (dataset loader + model +
inference + metrics). Step 13 giải quyết khoảng trống này.

### 1.2 Objective

- Tạo module đánh giá chuẩn `evaluation/evaluate_unet.py`.
- Chạy checkpoint `checkpoints/unet_cityscapes.pth` trên **toàn bộ 500 ảnh** Cityscapes `val`.
- Tính metrics bằng **đúng implementation sẵn có** (`evaluation/segmentation_metrics.py`).
- Ghi kết quả ra JSON `outputs/analysis/unet_cityscapes_evaluation.json` kèm per-class
  metrics (IoU / Dice / Accuracy theo 19 class).

### 1.3 Input / Output

| Input | Output |
|---|---|
| `checkpoints/unet_cityscapes.pth` (model_state_dict + metadata) | `outputs/analysis/unet_cityscapes_evaluation.json` |
| `configs/unet.yaml` (`model.num_classes=19`, `data.image_size=[256,512]`) | Pixel Accuracy, mIoU, Mean Dice, per-class IoU/Dice/Accuracy |
| `CityscapesDataset` val split (500 ảnh + GT) | Prediction masks (optional `--save-predictions`) |

### 1.4 Scope

- Đánh giá **CHỈ trên checkpoint đã train sẵn** — không train lại, không cập nhật tham số.
- Tái sử dụng tối đa: `preprocessing/cityscapes.py`, `models/unet/model.py`,
  `models/unet/inference.py`, `evaluation/segmentation_metrics.py`, `utils/config.py`,
  `utils/seed.py`, device utilities.
- `ignore_index=255`, 19 class trainId; class names lấy từ `TRAIN_ID_CLASS_NAMES` của dự án.
- Đánh giá đầy đủ **tất cả 500 ảnh val** (không dùng subset cho kết quả cuối).

### 1.5 Out of Scope

Theo prompt Step 13, **KHÔNG** làm:

- Huấn luyện lại U-Net; huấn luyện hoặc sửa MiDaS.
- Sửa fusion pipeline; đổi kiến trúc U-Net.
- Đổi Cityscapes class mapping; thay thế implementation metrics sẵn có.
- Dùng black-box segmentation library.
- Đánh giá chỉ trên một tập nhỏ cho kết quả cuối.
- Tiến hành các bước sau Step 13.

---

## 2. PURPOSE & SURVEY

### 2.1 Purpose

Xác minh **độc lập** chất lượng mô hình U-Net sau training:

- So sánh kết quả đánh giá độc lập với `training_val_miou` / `training_val_pixel_accuracy`
  lưu trong checkpoint metadata.
- Cung cấp phân tích **per-class** (class nào mạnh, class nào yếu) làm cơ sở cho thảo luận
  hạn chế của mô hình.
- Tạo artifact JSON tái lập được, dùng làm đầu vào cho các bước đánh giá tổng hợp sau.

### 2.2 Technical Background

- **Semantic segmentation metrics**: Pixel Accuracy, IoU per class, mIoU, Dice per class,
  Mean Dice, accuracy per class — đều được tính từ **confusion matrix `(19×19)`** gộp trên
  toàn bộ ảnh.
- `iou = TP / (TP + FP + FN)`; `dice = 2·TP / (2·TP + FP + FN)`; `mean_*` = trung bình trên
  các class thực sự xuất hiện trong ground-truth (không chia cho class vắng mặt).
- Pixel `255` (ignore/void) bị loại hoàn toàn khỏi mọi tính toán — không trở thành class.
- Khác biệt chính giữa kết quả training-time và evaluation-time: vòng validate trong training
  **batch-average** metrics từng batch; Step 13 **gộp một global confusion matrix** trên cả
  500 ảnh rồi mới tính metrics (`aggregate_from_confusion_matrix`).

### 2.3 Technology / Method Survey

| Thành phần | Vai trò |
|---|---|
| `torch.load(weights_only=True)` | Đọc checkpoint; yêu cầu dict chứa `model_state_dict` |
| `models/unet/inference.py::load_model` | Nạp state_dict vào `UNet.from_config`, đặt `model.eval()` |
| `torch.no_grad()` | Forward không tạo gradient, không cập nhật tham số |
| `confusion_matrix` + `aggregate_from_confusion_matrix` | Metrics từ một global matrix (đúng implementation Step 06) |
| `PIL.Image.fromarray(mode="L")` | Lưu prediction mask (chỉ khi `--save-predictions`) |

### 2.4 Why This Approach?

- **Không nhân bản formula**: mọi metric tái sử dụng `evaluation/segmentation_metrics.py` →
  nhất quán với Step 06.
- **An toàn khi đánh giá checkpoint khác**: `_task_params` cross-check `num_classes` /
  `ignore_index` giữa config và checkpoint metadata; nếu lệch (ví dụ checkpoint 8 class mà
  config 19) sẽ **raise**, không chấm điểm lặng lẽ một checkpoint train cho task khác.
- **Tái lập được**: cùng checkpoint + config + split → cùng kết quả JSON.

---

## 3. DATA

Not an independent data step — Step 13 **chỉ đánh giá** trên dữ liệu Cityscapes sẵn có:

- Dataset: `CityscapesDataset` (giữ nguyên từ Step 03), split **`val`** mặc định.
- Số lượng: **500 ảnh + 500 GT** (đã xác minh dataset: 500/500 khớp 100% nhờ `_build_samples`
  ghép theo stem filename).
- `image_size=[256,512]` (H×W) từ `configs/unet.yaml` — khớp với kích thước huấn luyện.
- Label: `gtFine_labelIds.png` → remap sang **trainId (0..18) + 255**.
- Dữ liệu **không được tải xuống** trong bước này; nếu thiếu `data/cityscapes` sẽ fail-fast
  với `FileNotFoundError: "Cityscapes dataset not found at ..."`.---

## 4. IMPLEMENTATION PLAN

### 4.1 Architecture

```
checkpoints/unet_cityscapes.pth ──┐
                                  ↓
configs/unet.yaml ──► UNet.from_config ──► load_model ──► model.eval()
                                  ↓                        (no_grad)
CityscapesDataset(val) ──► DataLoader(batch=1) ──► logits ──► argmax
                                  ↓
                      confusion_matrix (19x19, ignore 255) per batch
                                  ↓
                  aggregate_from_confusion_matrix → metrics
                                  ↓
              outputs/analysis/unet_cityscapes_evaluation.json
```

### 4.2 Components

| File / hàm | Chức năng |
|---|---|
| `evaluation/evaluate_unet.py` | Module đánh giá chính (396 dòng) |
| `evaluate_unet.load_checkpoint` | Load checkpoint; fail nếu thiếu file hoặc không có `model_state_dict` |
| `evaluate_unet._task_params` | Cross-check `num_classes`/`ignore_index` giữa config và checkpoint |
| `evaluate_unet.build_dataset` | Build `CityscapesDataset` split `val` |
| `evaluate_unet.evaluate_dataset` | Loop `no_grad`, gộp confusion matrix, trả `num_samples` + metrics |
| `evaluate_unet.build_results` | Lắp JSON per-class theo tên, kèm metadata checkpoint |
| `evaluate_unet.write_results` | Ghi JSON `indent=2` |
| `evaluate_unet.evaluate_checkpoint` / `main` | Orchestration + CLI |
| `tests/test_evaluate_unet.py` | 17 test synthetic trên CPU |

### 4.3 Configuration

Tái sử dụng `configs/unet.yaml` (không sửa trong Step 13):

- `model.num_classes: 19`, `data.image_size: [256, 512]`
- `data.split_dir.val: val`, `data.cityscapes_root: data/cityscapes`
- `env.seed: 42` (đọc để `set_seed` khi có seed truyền vào)

CLI defaults trong `build_parser`:
`--config unet`, `--checkpoint checkpoints/unet_cityscapes.pth`,
`--output outputs/analysis/unet_cityscapes_evaluation.json`, `--split val`,
`--device auto`, `--batch-size 1`, `--save-predictions` (tắt mặc định).

### 4.4 Processing Flow

1. Load config `unet` → `seed = int(get(config, "env.seed", 42))`.
2. `resolve_device(device)` — CUDA khi có, ngược lại CPU.
3. `load_checkpoint` — verify format `model_state_dict`.
4. `_task_params` → `num_classes=19`, `ignore_index=255`
   (từ `preprocessing.cityscapes.IGNORE_INDEX`).
5. `UNet.from_config(config)` + `load_model(model, checkpoint_path)` (đặt `model.eval()`).
6. `build_dataset(config, split="val")`; `DataLoader(batch_size, shuffle=False, num_workers=0)`.
7. Mỗi batch: `torch.no_grad()` forward → `logits.argmax(1)` →
   `confusion_matrix(...)` cộng dồn vào global matrix.
8. `aggregate_from_confusion_matrix(total_cm)` → metrics, `num_samples`.
9. `build_results` → JSON (kèm `checkpoint_epoch`, `training_val_miou`,
   `training_val_pixel_accuracy` từ checkpoint metadata).
10. `write_results` → lưu JSON, in tóm tắt (best/worst class IoU) ra stdout.

### 4.5 Error Handling

| Tình huống | Xử lý |
|---|---|
| Checkpoint thiếu | `FileNotFoundError("Checkpoint file not found: ...")` → exit 1 |
| Checkpoint sai format (không có `model_state_dict`) | `ValueError` → exit 1 |
| Checkpoint `num_classes` / `ignore_index` lệch config | `ValueError` (không chấm điểm lặng lẽ) |
| Dataset thiếu `data/cityscapes` | `FileNotFoundError("Cityscapes dataset not found at ...")` |
| Batch không có label GT | `ValueError("Evaluation requires ground-truth labels for every image.")` |
| Loader không có sample | `RuntimeError("Evaluation loader produced no samples (empty dataset?).")` |
| Config lỗi | `ConfigError` → in `error: ...`, exit 1 |

---

## 5. SYSTEM BUILD FLOW

### 5.1 Workflow

1. Kiểm tra code sẵn có (model, inference, dataset, metrics, config, trainer) — không
   nhân bản chức năng đã tồn tại.
2. Viết `evaluation/evaluate_unet.py` (load checkpoint → build model/dataset → evaluate
   toàn split → gộp global confusion matrix → JSON).
3. Viết `tests/test_evaluate_unet.py` (dùng data synthetic nhỏ, không chạy 500 ảnh trong test).
4. Chạy test training module + full `pytest` regression.
5. Chạy đánh giá thật trên 500 ảnh val, ghi nhận kết quả.

### 5.2 Integration

- **Reuse**: `evaluate_dataset` dùng đúng `confusion_matrix` + `aggregate_from_confusion_matrix`
  của `evaluation/segmentation_metrics.py` → kết quả Step 13 nhất quán với Step 06.
- **Model loading**: `models/unet/inference.py::load_model` giữ đúng hợp đồng checkpoint
  `{"model_state_dict": ...}` của Step 12 → checkpoint `unet_cityscapes.pth` đọc được.
- **Không sửa** model architecture, class mapping, metric implementation, hay pipeline.
- JSON đầu ra `unet_cityscapes_evaluation.json` là nguồn cho các bước tổng hợp sau.

### 5.3 Testing

`tests/test_evaluate_unet.py` — **17 test** synthetic (data nhân tạo, CPU, không internet,
không chạy 500 ảnh), bao gồm:

1. checkpoint loading (`test_load_checkpoint_roundtrip`), thiếu file, sai format
2. evaluation chạy `no_grad`, weights không đổi (`test_evaluate_dataset_no_grad_and_weights_unchanged`)
3. output dimensions đúng (`iou_per_class`, `dice_per_class`, `class_accuracy` đủ 19)
4. `ignore_index` được tôn trọng (`test_evaluate_dataset_ignores_255_pixels`)
5. JSON result structure (`test_build_results_structure`) + roundtrip + lệch `num_classes`/`ignore_index`
6. CLI argument parsing (`test_build_parser_defaults`, `--save-predictions`)
7. tính nhất quán với `evaluate_segmentation` reference, batch-size invariance,
   save prediction masks
8. fail-fast dataset thiếu

### 5.4 CI/CD

- **KHÔNG CÓ CI/CD** trong dự án — không có `.github/workflows`; mọi kiểm tra chạy thủ
  công qua `pytest`. Bước này không thêm CI/CD.

### 5.5 Execution / Commands

```bash
# 1) Chạy test Step 13 (offline, CPU)
.venv/bin/python -m pytest tests/test_evaluate_unet.py -q

# 2) Chạy toàn bộ regression
.venv/bin/python -m pytest -q

# 3) Đánh giá thật trên 500 ảnh Cityscapes val
python -m evaluation.evaluate_unet --config unet \
    --checkpoint checkpoints/unet_cityscapes.pth

# 4) (Tùy chọn) Lưu thêm prediction masks ra outputs/segmentation/unet_cityscapes/
python -m evaluation.evaluate_unet --config unet --save-predictions
```---

## 6. EVALUATION

### 6.1 Evaluation Objective

- Đánh giá **độc lập** checkpoint `checkpoints/unet_cityscapes.pth` trên **toàn bộ 500 ảnh**
  Cityscapes `val`.
- Xác minh lại các con số từ lúc training (training_val_miou = 0.4262, training_val_pixel_accuracy
  = 0.9021) bằng một pipeline evaluation tách biệt.

### 6.2 Evaluation Method

- Model được đặt `eval()`; toàn bộ inference nằm trong `torch.no_grad()`.
- gộp một global confusion matrix `(19,19)` với `ignore_index=255` trên **tất cả 500 ảnh**,
  sau đó tính: `pixel_accuracy`, `miou`, `mean_dice`, `iou_per_class`, `dice_per_class`,
  `class_accuracy` (từ `aggregate_from_confusion_matrix`).
- Kết quả so sánh trực tiếp với GT trainId; class names lấy từ `TRAIN_ID_CLASS_NAMES`.

### 6.3 Results

Nguồn: `outputs/analysis/unet_cityscapes_evaluation.json` (đã đọc và đối chiếu trực tiếp).

**Tổng quan:**

| Field | Giá trị |
|---|---|
| checkpoint | `checkpoints/unet_cityscapes.pth` |
| split | `val` |
| num_samples | **500** |
| num_classes / ignore_index | 19 / 255 |
| image_size | [256, 512] |
| checkpoint_epoch | **20** |
| pixel_accuracy | **0.9042** |
| miou | **0.4451** |
| mean_dice | **0.5498** |
| (training) training_val_miou | 0.4262 |
| (training) training_val_pixel_accuracy | 0.9021 |

Ghi chú: mIoU độc lập (0.4451) cao hơn một chút so với training_val_miou (0.4262) vì
phương pháp tổng hợp khác (global confusion matrix vs batch-average trong training).

**Per-class (IoU giảm dần):**

| Class | IoU | Dice | Accuracy |
|---|---|---|---|
| road | 0.9522 | 0.9755 | 0.9760 |
| sky | 0.8842 | 0.9385 | 0.9742 |
| vegetation | 0.8630 | 0.9265 | 0.9183 |
| car | 0.8436 | 0.9152 | 0.9375 |
| building | 0.8237 | 0.9033 | 0.9331 |
| sidewalk | 0.6793 | 0.8090 | 0.7810 |
| bicycle | 0.5304 | 0.6932 | 0.6327 |
| person | 0.5280 | 0.6911 | 0.7730 |
| traffic sign | 0.5006 | 0.6672 | 0.5685 |
| terrain | 0.4290 | 0.6004 | 0.6495 |
| pole | 0.3986 | 0.5700 | 0.4713 |
| traffic light | 0.2183 | 0.3584 | 0.2235 |
| wall | 0.1912 | 0.3211 | 0.2649 |
| fence | 0.1881 | 0.3166 | 0.2915 |
| bus | 0.1657 | 0.2842 | 0.1973 |
| train | 0.1307 | 0.2311 | 0.5600 |
| truck | 0.0760 | 0.1412 | 0.0820 |
| motorcycle | 0.0422 | 0.0811 | 0.0450 |
| rider | 0.0115 | 0.0227 | 0.0116 |

- **Best class IoU**: `road` = **0.9522**
- **Worst class IoU**: `rider` = **0.0115**

**Tests:**

| Test | Kết quả |
|---|---|
| `pytest tests/test_evaluate_unet.py -q` | **17 passed** in 1.98s |
| `pytest -q` (full regression) | **456 passed, 1 skipped** in 22.20s |

### 6.4 Limitations

- Metric pixel census tính toán ở resolution `256×512` (không phải resolution gốc 1024×2048)
  → nhạy cảm hơn với class nhỏ (rider, motorcycle, truck).
- Không có augmentation; kết quả chỉ trên Cityscapes val analytic.
- Per-class IoU của các class hiếm/nhỏ rất thấp — minh họa hạn chế về class-imbalance và
  resolution thấp, không phải lỗi script.
- Accuracy per class của `train` (0.5600) không tương xứng IoU (0.1307) — class hiếm được
  dự đoán đúng một phần nhỏ diện tích thật.
- Không đánh giá được tốc độ/FPS (không thuộc phạm vi Step 13).

---

## 7. AI-ASSISTED DEVELOPMENT

### 7.1 Prompt

- File: `prompts/13_evaluate_unet.md` (STEP 13 — Evaluate trained U-Net on Cityscapes validation).
- Yêu cầu cốt lõi: đánh giá checkpoint đã train trên toàn bộ Cityscapes val; tái sử dụng
  dataset loader, model, inference, metrics; tạo `evaluation/evaluate_unet.py` + JSON; test
  synthetic; không retrain; chạy full regression; STOP sau Step 13.

### 7.2 Generated / Modified Files

**Files created** (Step 13):
- `evaluation/evaluate_unet.py` — module đánh giá
- `tests/test_evaluate_unet.py` — 17 test
- `outputs/analysis/unet_cityscapes_evaluation.json` — kết quả đánh giá thật

**Files modified**: không có (module/config/metrics/architecture giữ nguyên).

### 7.3 Testing

- `pytest tests/test_evaluate_unet.py` → **17 passed**
- `pytest -q` → **456 passed, 1 skipped**
- Thực thi thành công lệnh đánh giá thật; JSON chứa đúng 500 samples.

### 7.4 Version History

| Commit | Nội dung liên quan Step 13 |
|---|---|
| `d4a21f6` | "unet module" — nền model/inference |
| `47678b0` | "Steps 12-17: unet/midas evaluation, pipeline fusion, custom demo, report" — chứa `evaluation/evaluate_unet.py`, `tests/test_evaluate_unet.py`, JSON kết quả |
| `970c3e4` | "add rp step 12" — report 12 (bối cảnh trước Step 13) |

*(Không tạo commit mới trong bước viết báo cáo này — chỉ kiểm chứng.)*

### 7.5 Output

- `outputs/analysis/unet_cityscapes_evaluation.json`
- Chưa lưu prediction masks mặc định (`--save-predictions` off); `outputs/segmentation/unet_cityscapes/`
  hiện không tồn tại trong repo.

---

## 8. DISCUSSION

### 8.1 Strengths

- **Đánh giá độc lập & reproducible**: cùng checkpoint + config + split → cùng JSON; tách khỏi
  vòng validation trong training.
- **Tái sử dụng tối đa**: không viết lại metrics/formula; tận dụng `models/unet/inference.py`
  và `evaluation/segmentation_metrics.py`.
- **An toàn**: `no_grad`, `eval()`, cross-check `num_classes`/`ignore_index` → không chấm sai
  một checkpoint của task khác.
- **Per-class rõ ràng**: JSON liệt kê 19 class theo tên thật của Cityscapes.

### 8.2 Weaknesses

- Chạy toàn bộ 500 ảnh trên CPU tốn thời gian (không có CUDA trong môi trường test);
  script hỗ trợ CUDA khi có.
- `--save-predictions` chưa được bật mặc định nên chưa có prediction masks lưu sẵn.
- Metric chỉ ở resolution 256×512 thấp hơn native Cityscapes.

### 8.3 Limitations

- Kết quả chỉ có nghĩa trên Cityscapes val analytic; không phải benchmark trên camera thật.
- Các class hiếm (rider, motorcycle, truck) có IoU rất thấp — thể hiện giới hạn của model +
  resolution, không phải lỗi đo lường.
- Không đo timing/FPS.

### 8.4 What This Step Does / Does Not Prove

**Does prove**: checkpoint `unet_cityscapes.pth` tái lập được kết quả segmentation khả dụng
trên 500 ảnh val với pixel accuracy 0.9042, mIoU 0.4451, mean dice 0.5498; class lớn chiếm
ưu thế, class nhỏ/hiếm yếu.

**Does NOT prove**: không nói model tốt tuyệt đối, không khẳng định chất lượng ngoài
Cityscapes val, không khẳng định real-time/hardware nào cả. Cũng không chứng minh checkpoint
là "tốt nhất có thể" — chỉ đánh giá mô hình đã train.

---

## 9. CONCLUSION

Step 13 hoàn thành mục tiêu **đánh giá độc lập U-Net trên toàn bộ Cityscapes val**:

- Module `evaluation/evaluate_unet.py` + 17 test synthetic (offline, CPU, không internet).
- Kết quả thật trên 500 ảnh: **pixel accuracy 0.9042, mIoU 0.4451, mean dice 0.5498**;
  best class IoU `road` 0.9522, worst `rider` 0.0115.
- Kết quả JSON: `outputs/analysis/unet_cityscapes_evaluation.json`.
- Full regression: **456 passed, 1 skipped**.
- Vòng validation training: training_val_miou 0.4262 — phù hợp, độ lệch do phương pháp tổng hợp.

Không tiến hành gì ngoài phạm vi Step 13; không retrain, không sửa pipeline/MiDaS/metrics.

---

## 10. REFERENCES

1. Prompt Step 13 — `prompts/13_evaluate_unet.md`
2. Evaluation module — `evaluation/evaluate_unet.py`
3. Step 13 tests — `tests/test_evaluate_unet.py`
4. Metrics — `evaluation/segmentation_metrics.py`
5. Checkpoint — `checkpoints/unet_cityscapes.pth`
6. Evaluation output — `outputs/analysis/unet_cityscapes_evaluation.json`
7. U-Net model — `models/unet/model.py`
8. U-Net inference — `models/unet/inference.py`
9. Dataset — `preprocessing/cityscapes.py`
10. Config — `configs/unet.yaml`
11. Báo cáo trước có liên quan — `docs/coursework/12_train_unet.md`, `docs/coursework/06_evaluation.md`