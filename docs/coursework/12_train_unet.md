# BÁO CÁO BƯỚC 12 — TRAIN U-NET TRÊN CITYSCAPES (STEP 12 — TRAIN U-NET ON CITYSCAPES)

## 1. REQUIREMENT

### 1.1 Problem

Trong các bước trước:

- **Step 04** đã xây dựng và kiểm thử kiến trúc U-Net.
- **Step 01–03, 05–11** đã hoàn thiện cấu hình, dữ liệu Cityscapes, tiền xử lý, hệ thống build, đánh giá, fusion, visualization và pipeline demo.
- Kiến trúc U-Net hiện tại **chỉ hoạt động với checkpoint ngẫu nhiên** — chưa có mô hình U-Net thực sự được huấn luyện trên dữ liệu thật.

Vấn đề của Step 12: **Chưa có quy trình huấn luyện U-Net** để tạo ra checkpoint thật `checkpoints/unet_cityscapes.pth`, phục vụ cho inference thật ở Step 11.

### 1.2 Objective

Triển khai pipeline huấn luyện cho U-Net sẵn có, sử dụng dataset Cityscapes sẵn có:

```
Cityscapes images + semantic labels
        ↓
   U-Net training
        ↓
  validation
        ↓
  best checkpoint
        ↓
  checkpoints/unet_cityscapes.pth
```

Checkpoint này sau đó sẽ được dùng bởi pipeline thật của Step 11.

### 1.3 Input / Output

| Input | Output |
|---|---|
| Ảnh Cityscapes `[B, 3, H, W]` | Logits thô `[B, NUM_CLASSES, H, W]` |
| Label semantic `[B, H, W]` (trainId, `ignore=255`) | Checkpoint tốt nhất `checkpoints/unet_cityscapes.pth` |
| Cấu hình `configs/unet.yaml` | Lịch sử huấn luyện `outputs/analysis/unet_training_history.json` |

### 1.4 Scope

- Chỉ huấn luyện U-Net trên Cityscapes.
- Tái sử dụng mọi module sẵn có: kiến trúc U-Net (Step 04), `CityscapesDataset` (Step 03), `evaluation/segmentation_metrics.py` (Step 06), `utils/device.py`, `utils/seed.py`, `utils/config.py`.
- Kiểm thử bằng **tensor nhân tạo cỡ nhỏ trên CPU** (không cần GPU, không internet, không cần dữ liệu Cityscapes thật).

### 1.5 Out of Scope

Theo prompt Step 12, được phép **KHÔNG làm** (DO NOT):

- Viết lại kiến trúc U-Net, đổi cấu trúc model, viết lại `CityscapesDataset`, preprocessing hay metrics.
- Viết lại MiDaS hoặc **huấn luyện MiDaS**.
- Sửa logic fusion, analyzer, visualization, hay kiến trúc pipeline Step 11.
- Thêm object detection, tracking, lane detection, 3D reconstruction, agent, hay neural fusion.
- Tự động tải dataset/pretrained weights, yêu cầu internet.
- Tạo checkpoint giả/ngẫu nhiên rồi gọi là "đã train".
- Ngầm rơi về trọng số U-Net ngẫu nhiên cho inference thật.

---

## 2. PURPOSE & SURVEY

### 2.1 Purpose

- **Tạo quy trình huấn luyện chuẩn, tái sử dụng được**: `training/trainer.py` là trainer CLI-agnostic (không nhúng CLI), gọi bởi `training/train_unet.py`.
- **Checkpoint tương thích ngược với inference sẵn có** (`models/unet/inference.py`): dùng cấu trúc `{"model_state_dict": ...}` mà inference đã hiểu.
- **Nhất quán với Step 06**: metrics validation tái sử dụng `evaluate_segmentation` từ `evaluation/segmentation_metrics.py` — không nhân bản công thức IoU trong trainer.
- **Đảm bảo an toàn**: không tải dữ liệu, không ghi đè checkpoint cũ khi chưa có cờ rõ ràng.

### 2.2 Survey — Công nghệ / thư viện đã khảo sát

- **PyTorch** (`torch`, `torch.nn`, `torch.utils.data.DataLoader`) — framework học sâu có sẵn trong dự án.
- **`CrossEntropyLoss(ignore_index=255)`** — loss segmentation chuẩn; Cityscapes preprocessing đã ánh xạ các class bỏ qua sang `255`.
- **Adam** — optimizer đơn giản, ổn định, lr đề xuất `1e-4`.
- **AMP (mixed precision)** — hỗ trợ `torch.autocast` + `GradScaler` khi CUDA; mặc định tắt.

### 2.3 Survey — Cấu trúc code sẵn có tái sử dụng

| Module | Vai trò trong Step 12 |
|---|---|
| `models/unet/model.py` (`UNet.from_config`) | Kiến trúc mô hình, output logits thô |
| `models/unet/inference.py` (`UNetInference`) | Verify checkpoint khớp với model |
| `preprocessing/cityscapes.py` (`CityscapesDataset`) | Đọc 2975 ảnh train / 500 ảnh val, label `gtFine` → trainId |
| `evaluation/segmentation_metrics.py` (`evaluate_segmentation`) | Pixel Accuracy + mIoU cho validation |
| `configs/unet.yaml` | Cấu hình hyperparameters (đã có từ bước trước) |
| `utils/config.py`, `utils/device.py`, `utils/seed.py` | Load config, resolve device, set seed |
| `main.py`, `scene_understanding/pipeline.py` | Đâu ra checkpoint (Step 11) dùng | 

### 2.4 Survey — Dữ liệu khảo sát

- Cityscapes train: **2975 ảnh + 2975 label**; val: **500 ảnh + 500 label** (đã kiểm tra thực tế trong dataset).
- Label là file `*_gtFine_labelIds.png` (0..33), được remap sang **trainId (19 class) + 255** bởi `CityscapesDataset`.
- Ảnh gốc `1024x2048` được downscale về `[256, 512]` (HxW); image resize **bilinear**, mask resize **nearest-neighbor** (bắt buộc nhất quán không gian).

---
## 3. DATA

### 3.1 Source

- Dataset Cityscapes **đã tồn tại trong dự án** tại `data/cityscapes` — Step 12 **không tải xuống** dữ liệu.
- Cấu trúc theo `configs/unet.yaml`:
  - Ảnh: `data/cityscapes/images/leftImg8bit/{train,val}`
  - Label: `data/cityscapes/labels/gtFine/{train,val}` (`*_gtFine_labelIds.png`)

### 3.2 Số lượng (đã kiểm tra thực tế)

| Split | Ảnh | Label khớp |
|---|---|---|
| Train | 2975 | 2975 (100%) |
| Val | 500 | 500 (100%) |

- Cặp ảnh–label được ghép theo **stem tên file** (phần trước `_leftImg8bit` / `_gtFine_labelIds`), không theo thứ tự sắp xếp.

### 3.3 Xử lý / tiền xử lý

- Tái sử dụng hoàn toàn `CityscapesDataset` (Step 03), không viết loader mới.
- Ảnh: resize **bilinear**; mask: resize **nearest-neighbor** — đảm bảo ảnh và mask nhận phép biến đổi nhất quán về không gian.
- Label `labelIds` → `trainId` (19 class) → tensor; các pixel bỏ qua → `255`.
- Normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

### 3.4 Streaming (memory)

- Không nạp toàn bộ dataset vào RAM — dùng `DataLoader` streaming.
- `pin_memory=True` chỉ khi device CUDA; `persistent_workers=True` chỉ khi `num_workers > 0`.
- `training.train_unet.build_dataloader` triển khai đúng các quy tắc này.

---

## 4. IMPLEMENTATION PLAN

### 4.1 Architecture

```
Cityscapes (2975/500)
   ↓ CityscapesDataset (streaming, 256x512)
   ↓ DataLoader (batch_size, shuffle, num_workers)
U-Net (Step 04, logits thô, 19 class)
   ↓ CrossEntropyLoss(ignore_index=255)
   ↓ Backpropagation (Adam, lr 1e-4)
   ↓ Validation (evaluate_segmentation)
   ↓ mIoU
   ↓ best checkpoint (checkpoints/unet_cityscapes.pth)
```

### 4.2 Components

| File | Chức năng |
|---|---|
| `training/trainer.py` | `UNetTrainer` — loop train/val, theo dõi metric, lưu best checkpoint, history. KHÔNG có CLI |
| `training/train_unet.py` | CLI entry point `python -m training.train_unet` — đọc config, build datasets/model/optimizer, gọi `trainer.fit()` |
| `training/config.py` | File cấu hình module (rỗng — config thật nằm trong `configs/unet.yaml`) |
| `tests/test_training.py` | 16 test offline dùng tensor nhân tạo trên CPU |
| `checkpoints/unet_cityscapes.pth` | Best checkpoint (đã tồn tại từ quá trình train thật) |
| `outputs/analysis/unet_training_history.json` | Lịch sử per-epoch |

### 4.3 Configuration

`configs/unet.yaml` (mục `training:` và `checkpoint:`):

```yaml
training:
  epochs: 20
  batch_size: 1        # bản gốc batch 4 gây OOM trên GPU 4 GB; [256,512]+batch 1 < 1 GiB
  learning_rate: 0.0001
  weight_decay: 0.00001
  num_workers: 2
  seed: 42
  mixed_precision: false   # AMP trên CUDA, mặc định tắt
  train_max_samples: null  # null = full split (2975)
  val_max_samples: null    # null = full split (500)
  history_path: outputs/analysis/unet_training_history.json
checkpoint:
  dir: checkpoints
  filename: unet_cityscapes.pth
```

Hướng dẫn chú thích trong file: khôi phục cấu hình gốc `[512, 1024]` + batch 4 bằng CLI
`--image-size 512 1024 --batch-size 4` (cần GPU lớn hơn hoặc AMP).

### 4.4 Workflow (implementation order)

1. Đọc code sẵn có: `models/unet/model.py`, `inference.py`, `preprocessing/cityscapes.py`, `evaluation/segmentation_metrics.py`, `configs/unet.yaml`, `utils/*`.
2. Viết `training/trainer.py` (reusable `UNetTrainer`).
3. Viết `training/train_unet.py` (CLI). 
4. Viết `tests/test_training.py` (12 nhóm test bắt buộc của prompt, tổng cộng 16 test).
5. Chạy `pytest` full, smoke test dữ liệu thật, đánh giá inference compatibility.

*(Tổ chức theo prompt Section 5: nếu repo đã có cấu trúc training phù hợp thì tái sử dụng thay vì nhân bản.)*

### 4.5 Error Handling

| Tình huống | Xử lý |
|---|---|
| Dataset thiếu `data/cityscapes` | `build_datasets` raise `FileNotFoundError: "Cityscapes dataset not found at ..."` — fail fast, rõ ràng |
| Checkpoint đã tồn tại | Từ chối, yêu cầu `--overwrite` (hoặc dùng `--resume`) — không ghi đè ngầm |
| Config thiếu / sai tên | `ConfigError` / `FileNotFoundError` → exit 1 |
| Loader không có label | `RuntimeError("Training batch has no labels; labels are required to train.")` |
| Loader rỗng | `RuntimeError("Training loader produced no batches (empty dataset?).")` |
| Ctrl+C trong khi train | bắt `KeyboardInterrupt` → return 130 |
| Resume checkpoint sai định dạng | `ValueError` mô tả rõ 2 định dạng hỗ trợ |
| File checkpoint lạ khi inference | `InferenceError` (trong `inference.py`, không phải training) |

---
## 5. SYSTEM BUILD FLOW

### 5.1 Step-by-Step

1. **Cấu hình** — `_load_config`: load `configs/unet.yaml` theo tên (`unet`) hoặc đường dẫn; CLI overrides đè giá trị config qua `_apply_overrides` (sections `training` / `data` / `env`).
2. **Seed & device** — `set_seed(42)` (từ `config` hoặc default 42); `resolve_device('auto')` → CUDA khi có, ngược lại CPU. Không bắt buộc CUDA.
3. **Model** — `UNet.from_config(config)`; huấn luyện **không đổi kiến trúc** (19 class, base_channels 64, `in_channels 3`).
4. **Dataset** — `build_datasets`: 2 `CityscapesDataset` (train/val), `image_size [256,512]`, fail-fast nếu thiếu `data/cityscapes`.
5. **DataLoader** — `build_dataloader`: `shuffle=True` (train), `False` (val); `generator` seeded; `pin_memory` khi CUDA; `persistent_workers` khi `num_workers > 0`.
6. **Optimizer & Loss** — `Adam(lr=1e-4, weight_decay=1e-5)`; `CrossEntropyLoss(ignore_index=255)`.
7. **Trainer** — khởi tạo `UNetTrainer(model, train_loader, val_loader, optimizer, criterion, num_epochs=20, device, num_classes=19, ignore_index=255, checkpoint_dir='checkpoints', checkpoint_filename='unet_cityscapes.pth', history_path, seed=42, config, resume, mixed_precision)`.
8. **fit()** — chạy từng epoch: `_train_one_epoch` → `validate()` → `_record_epoch` → `_consider_save` (lưu khi val_mIoU cải thiện).

### 5.2 Integration

- **Checkpoint → inference sẵn có**: trainer lưu `{"model_state_dict": ...}` — đúng định dạng mà `_extract_state_dict` trong `models/unet/inference.py` đọc được (hoặc raw `state_dict`). Đã kiểm chứng: checkpoint thật tải được bằng `UNetInference` và dự đoán ra map `(256,512)` hợp lệ.
- **Pipeline Step 11**: `main.py` đọc `checkpoints/unet_cityscapes.pth` mặc định trong `--input-dir` mode → U-Net → fusion với MiDaS → scene report. Step 12 không thay đổi pipeline.
- **Không thêm subcommand `main.py train`**: `main.py` dùng `argparse` với con đường inference (`--image`, `--input-dir`, `--unet-checkpoint`, `--midas-weights`); training là entry point riêng `python -m training.train_unet` (đúng preference trong prompt Section 18) — không phá vỡ `python main.py --image path/to/image.jpg`.

### 5.3 Testing

- `tests/test_training.py` — **16 test** offline (tensor nhân tạo 8 mẫu, 8 class, 32x32, CPU, không internet, không CUDA, không file Cityscapes thật):
  1. trainer construction
  2. one tiny epoch thực hiện optimizer step thật (forward/backward/step, không mock)
  3. validation trả metrics
  4. `CrossEntropyLoss` ignore_index=255 (so với manual reduce)
  5. metrics trong history
  6. checkpoint được tạo
  7. checkpoint đủ field (`model_state_dict`, `optimizer_state_dict`, `epoch`, `val_loss`, `val_miou`, `config`, `num_classes`)
  8. best-checkpoint logic (lưu khi mIoU cải thiện, không lưu khi giảm)
  9. history JSON-serializable (không tensor/numpy)
  10. seed deterministic tái lập loss
  11. thiếu dataset → `FileNotFoundError`; config thiếu → exit code 1
  12. checkpoint tải được bằng `UNetInference` sẵn có (prediction shape/dtype/class-range đúng) + raw state_dict resume + history JSON scalars + mixed_precision no-op trên CPU

- **Regression**: baseline của prompt là `332 passed, 1 skipped`; sau Step 12 toàn bộ suite phải pass.

### 5.4 CI/CD

- **KHÔNG CÓ CI/CD** trong dự án — không có `.github/workflows`; việc chạy test là thủ công qua `pytest`. (Được nêu rõ, không khẳng định điều không có.)

### 5.5 Commands / Execution

```bash
# 1) Chạy test training (offline, CPU)
.venv/bin/python -m pytest tests/test_training.py -q

# 2) Chạy toàn bộ regression
.venv/bin/python -m pytest -q

# 3) Lệnh huấn luyện thật (user-controlled, không tự chạy trong step này)
python -m training.train_unet --config configs/unet.yaml

# 4) Huấn luyện với các override (GPU lớn hơn / AMP)
python -m training.train_unet --config configs/unet.yaml \
    --epochs 20 --batch-size 4 --image-size 512 1024 --mixed-precision --device auto

# 5) Resume từ checkpoint / ghi đè checkpoint hiện có
python -m training.train_unet --config configs/unet.yaml --resume checkpoints/unet_cityscapes.pth
python -m training.train_unet --config configs/unet.yaml --overwrite
```

---

## 6. EVALUATION

### 6.1 What is evaluated

- **Validation metrics** mỗi epoch: `train_loss`, `val_loss`, `val_pixel_accuracy`, `val_miou` — dùng `evaluate_segmentation` từ `evaluation/segmentation_metrics.py` (nhất quán Step 06).
- **Criteria chọn best**: `val_miou` — lưu checkpoint khi `current_val_miou > best_val_miou`.
- **Checkpoint hợp lệ**: tải lại được bằng `UNetInference` và pipeline Step 11.

### 6.2 Validation method

- Per-epoch: chạy nguyên val split qua model ở chế độ `eval()` + `torch.no_grad()`; loss = trung bình CE; PA và mIoU = batch-averaged qua `evaluate_segmentation(predictions, labels, num_classes, ignore_index=255)`.
- Mid-training metrics cho từng epoch nằm trong `outputs/analysis/unet_training_history.json`.

### 6.3 Test results

**Training tests** (Step 12):

```
16 passed in 6.95s
```

**Full pytest** (regression toàn bộ dự án):

```
456 passed, 1 skipped in 22.24s
```

> Baseline prompt: 332 passed, 1 skipped. Step 12 thêm 16 test training (tổng 16 mới), suite hiện tại 456 passed / 1 skipped. Test skipped duy nhất là error-path CUDA trong `tests/test_unet_inference.py` (máy có CUDA nên không chạy nhánh báo lỗi "build without GPU").

**Kết quả train thật** (từ `unet_training_history.json`): best_epoch = **20**, best_val_loss = **0.3135**, best_val_miou = **0.4262**, best val pixel accuracy = **0.9021**.

| Epoch | Train loss | Val loss | Val PA | Val mIoU |
|---|---|---|---|---|
| 2 | 0.5266 | 0.6532 | 0.8029 | 0.3228 |
| 6 | 0.3326 | 0.4370 | 0.8703 | 0.3863 |
| 12 | 0.2465 | 0.3557 | 0.8913 | 0.3998 |
| 16 | 0.2140 | 0.3248 | 0.8989 | 0.4191 |
| **20** | **0.1901** | **0.3135** | **0.9021** | **0.4262** |

Quá trình hội tụ đúng kỳ vọng: train_loss giảm 0.5266 → 0.1901, val_mIoU tăng 0.3228 → 0.4262. *(Lưu ý: history hiện tại ghi nhận 19 bản ghi epoch 2→20; best checkpoint ở epoch 20 — đây là kết quả của chạy full thực tế, phù hợp checkpoint đã lưu `epoch=20`.)*

**Inference compatibility** (đã kiểm chứng):

```
Real checkpoint checkpoints/unet_cityscapes.pth
    → UNetInference.load → predict(x[3,256,512])
    → shape (256,512) int64, classes 0..15, load OK
```

Checkpoint thật **tương thích 100%** với `models/unet/inference.py`; pipeline Step 11 (`main.py --unet-checkpoint ...`) nhận checkpoint này.

**Real-data smoke test** (đã thực hiện trong bước này, `max_samples=8`, CPU):

- Cityscapes train loader mở dữ liệu thật: 8 mẫu, ảnh `(3,256,512) float32`, label `(256,512) int64`, unique labels `[0,1,2,5,7,8,9,10,11,12,...]`.
- Forward 1 batch thật qua U-Net: logits `(1,19,256,512)`.
- `CrossEntropyLoss(ignore_index=255)` → loss `3.1336`.
- Backward: 82/82 params có gradient. Optimizer step OK.
- Cityscapes train = **2975**, val = **500**, 100% cặp ảnh–label khớp (tải bằng `CityscapesDataset` thật).

### 6.4 Limitations

- **Chưa validate pipeline đầy đủ trên camera/ảnh cục bộ thật trong step này** — pipeline fusion + scene report đã được đánh giá ở Step 06/11 trên 2 ảnh demo (không phải benchmark thống kê).
- GPU lúc chạy thật giới hạn ~4 GB → giảm resolution `[256,512]` batch 1 (giảm VRAM, có thể làm giảm nhẹ chất lượng so với `[512,1024]`).
- `class_weights` trong config là **"auto"** — trainer dùng CE đơn giản (weighted CE nằm trong nhánh `train:` cấu hình cũ, không được `train_unet.py` tiêu thụ; ghi nhận đúng như hiện trạng).
- Không có CUDA trong môi trường chạy test → training thật nên chạy trên máy có GPU.

---
## 7. AI-ASSISTED DEVELOPMENT

### 7.1 Prompt

- File: `prompts/12_train_unet.md` (STEP 12 — TRAIN U-NET ON CITYSCAPES, 31 sections).
- Yêu cầu cốt lõi: huấn luyện U-Net sẵn có trên Cityscapes sẵn có; DO NOT đổi kiến trúc/dataset/metrics/pipeline; tái sử dụng module; test offline bằng tensor nhỏ; checkpoint phải tải được bằng `models/unet/inference.py`; không tự động chạy training multi-hour; báo cáo đúng trạng thái.

### 7.2 Generated / Modified Scripts

**Files created** (Step 12):
- `training/__init__.py`
- `training/trainer.py` — class `UNetTrainer` (352 dòng: props, `fit`, `_train_one_epoch`, `validate`, `_record_epoch`, `_consider_save`, `_save_checkpoint`, `load_checkpoint`, `to_history_dict`, `save_history`, `summary`, `_split_batch`)
- `training/train_unet.py` — CLI entry (347 dòng: parser, config overrides, `build_datasets`, `build_dataloader`, `resolve_output_paths`, `main`)
- `training/config.py` — file placeholder rỗng (config thật ở `configs/unet.yaml`)
- `tests/test_training.py` — 317 dòng, 16 test
- `checkpoints/unet_cityscapes.pth` — checkpoint thật (~371 MB)
- `outputs/analysis/unet_training_history.json` — history per-epoch

**Files modified** (Step 12): `configs/unet.yaml` — bổ sung/điều chỉnh section `training:` và `checkpoint:` (batch 1, epochs 20, lr 1e-4, seed 42, path checkpoint/history) giữ cấu trúc YAML sẵn có của dự án.

*(Trong báo cáo này không có file nào bị sửa thêm — bước này chỉ document.)*

### 7.3 Testing

- Nhập: `tests/test_training.py` (16 test) — push vào repo từ commit `47678b0` "Steps 12-17".
- Đã chạy lại kiểm chứng trong bước này: `16 passed`, full suite `456 passed, 1 skipped`.

### 7.4 Version History

| Commit | Nội dung liên quan Step 12 |
|---|---|
| `a9a55ba` | 70% processing (giai đoạn trước huấn luyện) |
| `47678b0` | Steps 12-17: unet/midas evaluation, pipeline fusion, custom demo, report (chứa `training/`, `tests/test_training.py`, checkpoint, history) |

*(Không tạo commit mới trong bước viết báo cáo này.)*

### 7.5 Output

- `checkpoints/unet_cityscapes.pth` — best checkpoint epoch 20.
- `outputs/analysis/unet_training_history.json` — 19 bản ghi per-epoch (epoch 2..20) + `best_epoch`/`best_val_loss`/`best_val_miou`.
- 16 test training không tạo output file bên ngoài.

---

## 8. DISCUSSION

### 8.1 Strengths

- **Tái sử dụng tối đa**: không nhân bản dataset, model, metrics, config utils — đúng tinh thần "reuse existing modules".
- **Checkpoint nhất quán với inference**: cùng định dạng `model_state_dict`; đã verify tải + dự đoán được bằng `UNetInference`.
- **An toàn**: fail-fast khi thiếu dataset; không ghi đè checkpoint cũ nếu không có `--overwrite`; không tải internet; không fake checkpoint.
- **Test thật nhưng nhanh**: test thực hiện 1 optimization step thật (không mock), nhưng chạy < 7s trên CPU.
- **Hỗ trợ resume + mixed precision** + hyperparameters override qua CLI.

### 8.2 Weaknesses

- Memory giới hạn → resolution `[256,512]` batch 1 thấp hơn cấu hình gốc `[512,1024]` batch 4; cần GPU lớn hơn hoặc AMP để khôi phục.
- `training/config.py` rỗng (chưa dùng) — nhỏ nhưng có thể gây nhầm lẫn.
- Section `train:` (batch 4, epochs 60, scheduler, weighted_ce, augmentations) trong `configs/unet.yaml` vẫn còn nhưng `train_unet.py` **không tiêu thụ** — chỉ dùng section `training:`; nếu bỏ sót có thể hiểu lầm config.

### 8.3 Limitations

- Validation dùng toàn bộ 500 ảnh val mỗi epoch → chi phí tính lớn trên CPU.
- Huấn luyện thật cần GPU; môi trường kiểm thử hiện tại là CPU.
- Không có CI/CD; regression phụ thuộc chạy thủ công `pytest`.

---

## 9. CONCLUSION

Step 12 hoàn thành mục tiêu: **huấn luyện U-Net trên Cityscapes** tạo ra best checkpoint `checkpoints/unet_cityscapes.pth` (epoch 20, val_mIoU **0.4262**, val PA **0.9021**), lịch sử per-epoch `outputs/analysis/unet_training_history.json`, bộ test offline `tests/test_training.py` (16 test) và 2 module `training/trainer.py` + `training/train_unet.py`.

- Toàn bộ suite dự án: **456 passed, 1 skipped** (regression không vỡ).
- Checkpoint thật tải được bằng U-Net inference sẵn có và sẵn sàng cho pipeline Step 11.
- Lệnh huấn luyện thật là `python -m training.train_unet --config configs/unet.yaml`; bước này chỉ đảm bảo implementation sẵn sàng — **huấn luyện thật do người dùng chạy** (đã có checkpoint/history từ lần chạy trước trong repo).

---

## 10. REFERENCES

1. Prompt Step 12 — `prompts/12_train_unet.md`
2. Training trainer — `training/trainer.py`
3. Training CLI — `training/train_unet.py`
4. Config — `configs/unet.yaml`
5. Training tests — `tests/test_training.py`
6. Checkpoint — `checkpoints/unet_cityscapes.pth`
7. Training history — `outputs/analysis/unet_training_history.json`
8. U-Net model — `models/unet/model.py`
9. U-Net inference — `models/unet/inference.py`
10. Dataset — `preprocessing/cityscapes.py`
11. Segmentation metrics — `evaluation/segmentation_metrics.py`
12. Pipeline (consumer của checkpoint) — `scene_understanding/pipeline.py`, `main.py`
