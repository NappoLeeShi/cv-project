# BÁO CÁO BƯỚC 14 — ĐÁNH GIÁ MIDAS TRÊN KITTI (STEP 14 — EVALUATE MIDAS ON KITTI)

## 1. REQUIREMENT

### 1.1 Problem

Trong các bước trước:

- **Step 04** đã tích hợp mô hình độ sâu MiDaS **DPT-Large** (pretrained, không huấn luyện).
- **Step 13** đã đánh giá độc lập U-Net trên toàn bộ Cityscapes val.
- Dataset KITTI `data/kitti/` gồm 1000 ảnh RGB và 1000 depth map GT đã sẵn sàng.

Vấn đề của Step 14: **chưa có đánh giá độc lập, reproducible** toàn bộ 1000 ảnh KITTI cho
MiDaS bằng chính các module sẵn có của dự án (model loader, KITTI loader, depth metrics).

### 1.2 Objective

- Tạo module `evaluation/evaluate_midas.py` chạy MiDaS DPT-Large trên **toàn bộ 1000 ảnh
  KITTI**, không train (evaluation + inference only).
- Tái sử dụng `models/midas/model.py`, `models/midas/inference.py`, `preprocessing/kitti.py`,
  `evaluation/depth_metrics.py`.
- Báo cáo các depth metrics chuẩn: **RMSE, MAE, AbsRel, δ1, δ2, δ3** với median scaling ở
  evaluation time.
- Ghi kết quả ra JSON `outputs/analysis/midas_kitti_evaluation.json`.

### 1.3 Input / Output

| Input | Output |
|---|---|
| `checkpoints/dpt_large_384.pt` (pretrained DPT-Large, ~1.38 GB) | `outputs/analysis/midas_kitti_evaluation.json` |
| `data/kitti/images/` — 1000 ảnh RGB | RMSE, MAE, AbsRel, δ1, δ2, δ3 (1000 ảnh) |
| `data/kitti/depth/` — 1000 depth map GT (16-bit mm) | Prediction maps `.npy` (optional `--save-predictions`) |
| `configs/midas.yaml` | `mean_scale`, `median_scaling`, metadata |

### 1.4 Scope

- Đánh giá **MiDaS DPT-Large pretrained** trên **toàn bộ 1000 KITTI validation samples** —
  không dùng subset cho kết quả cuối.
- Tái sử dụng tối đa module sẵn có; không viết lại depth metric, preprocessing, hay model loader.
- Tôn trọng hợp đồng ngữ nghĩa: MiDaS output là **relative inverse depth** (`larger = closer`);
  median scaling chỉ là alignment ở evaluation time, Không phải training, và MiDaS **không**
  sinh metric depth (meters).
- Tôn trọng valid-depth mask của KITTI (pixel depth `0` / vượt `depth_cap_m` là invalid).

### 1.5 Out of Scope

Theo prompt Step 14, **KHÔNG** làm:

- Huấn luyện MiDaS; sửa model architecture; chuyển sang DPT/BEiT khác.
- Sửa U-Net, sửa U-Net checkpoint, sửa fusion pipeline.
- Tải checkpoint khác từ internet.
- Dùng KITTI anonymous test set (không có GT).
- Đánh giá chỉ trên một tập nhỏ cho kết quả cuối.
- Mô tả raw MiDaS output như metric depth.
- Tiến hành các bước sau Step 14.

---

## 2. PURPOSE & SURVEY

### 2.1 Purpose

- Đo chất lượng ước lượng độ sâu tương đối của MiDaS so với GT metric của KITTI sau
  median alignment, trên trọn bộ 1000 ảnh.
- Nối tiếp Step 13: đây là "nửa còn lại" của đánh giá model-level — U-Net (segmentation)
  và MiDaS (depth) được đánh giá độc lập trước khi gộp vào pipeline (Step 15).
- Xác minh checkpoint `dpt_large_384.pt` load được và tạo prediction khớp không gian
  với GT KITTI.

### 2.2 Technical Background

- **MiDaS DPT-Large**: mô hình monocular depth pretrained, Dense Prediction Transformer.
  Output là **relative inverse depth** — giá trị lớn = gần camera. Không phải metric meters.
- **Vì relative, không thể so trực tiếp với depth GT metric**: cần **alignment** ở evaluation.
  Median scaling: `scale = median(gt_valid) / median(pred_valid)`; `scaled_pred = pred * scale`.
  Mỗi ảnh được align độc lập với một scale riêng (không dùng chung một scale toàn tập).
- **KITTI depth GT**: PNG 16-bit đơn kênh tính theo millimetre; loader chia cho
  `scale_mm=1000.0` để ra mét. Depth `0` đánh dấu pixel invalid (do surface sparsity của
  Velodyne) — được đưa vào `valid_mask`.
- Các metric: `rmse`, `mae`, `abs_rel`, và `delta_accuracy` với các ngưỡng
  `1.25^1, 1.25^2, 1.25^3` (δ1, δ2, δ3). Threshold accuracy:
  `mean(max(pred/gt, gt/pred) < threshold)` trên pixel valid có prediction dương.
- Vì mỗi ảnh align riêng, metric tổng = **trung bình per-image** (không thể gộp một global
  statistic đơn cho toàn tập; khác với U-Net dùng global confusion matrix ở Step 13).

### 2.3 Technology / Method Survey

| Thành phần | Vai trò |
|---|---|
| `models/midas/model.py` | `MiDaSModel`, `MiDaSError`, `_load_weights`; `MODEL_SOURCE="intel-isl/MiDaS"` |
| `models/midas/inference.py` | `MidDepthPredictor` — transform + forward + resize về resolution gốc |
| `preprocessing/kitti.py` | `KittiDepthDataset` — pair ảnh/depth theo stem, `valid_mask` |
| `evaluation/depth_metrics.py` | `evaluate_depth` — RMSE/MAE/AbsRel/δ1..δ3; `median_scale` |
| `torch.hub.load(MODEL_SOURCE, "DPT_Large", pretrained=False)` | Dựng architecture (không tải weights network) |
| `utils/device.py`, `utils/seed.py`, `utils/config.py` | Device/seed/config chuẩn dự án |

### 2.4 Why This Approach?

- **Không nhân bản**: model loading (qua `_load_weights` sẵn có), preprocessing (qua
  `MidDepthPredictor`), và metric formulas (qua `evaluate_depth`) đều tái sử dụng.
- **Median scaling per-image**: đúng đặc tính relative của MiDaS; mỗi ảnh có scale riêng
  giúp alignment công bằng, không phụ thuộc thứ tự dữ liệu.
- **Loại pixel invalid**: chỉ tính trên valid pixels (GT > 0, trong `depth_cap_m`), không
  coi pixel `0` của KITTI là GT hợp lệ.

---

## 3. DATA

Step 14 sử dụng dữ liệu KITTI đã chuẩn bị sẵn trong repo (không tải xuống, không sửa đổi):

- `data/kitti/images/` — **1000 ảnh RGB** (đã kiểm tra: 1000 file `.png`).
- `data/kitti/depth/` — **1000 depth map GT** (đã kiểm tra: 1000 file `.png`).
- **Pairing theo stem** (không theo thứ tự sắp xếp): loader dựng `image_map` và `depth_map`
  bằng `_build_stem_map`, ghép cặp theo cùng key. Vì tên file KITTI khác nhau giữa ảnh
  (`_sync_image_<frame>_<camera>`) và depth (`_sync_groundtruth_depth_<frame>_<camera>`),
  `evaluation/evaluate_midas.py` truyền `stem_key=_kitti_stem_key` để chuẩn hoá:
  `_sync_image_` / `_sync_groundtruth_depth_` → `_sync_`. Kết quả: **1000/1000 cặp khớp**.
- Không dùng KITTI anonymous test set (không có GT).
- Không có file `splits/val.txt` → loader dùng mọi stem xuất hiện ở cả hai thư mục (1000 mẫu).
- GT depth: PNG 16-bit millimetre → mét (chia `scale_mm=1000.0`); pixel `0` invalid →
  `valid_mask`; áp `depth_cap_m=80` từ config (pixel xa hơn 80 m bị đánh invalid).
- Dataset builder của evaluate_midas dùng `image_size=None` để giữ resolution gốc KITTI —
  prediction resize về đúng resolution GT trước khi so sánh.

---

## 4. IMPLEMENTATION PLAN

### 4.1 Architecture

```
data/kitti/images (1000 RGB)  ──┐
                    paired by stem (via _kitti_stem_key)
data/kitti/depth (1000 GT mm) ──┴──► KittiDepthDataset (valid_mask, metres)
                 ↓
configs/midas.yaml ──► input_size=384 ──► MidDepthPredictor(MiDaSModel DPT-Large)
                 ↓                        (torch.no_grad, model.eval())
prediction per image (relative inverse depth)
                 ↓              median scale (per image, eval-time)
evaluate_depth(pred, gt, valid, align="median") → rmse, mae, abs_rel, δ1..δ3
                 ↓
            mean over 1000 images + mean_scale
                 ↓
    outputs/analysis/midas_kitti_evaluation.json
```

### 4.2 Components

| File / hàm | Chức năng |
|---|---|
| `evaluation/evaluate_midas.py` | Module đánh giá chính (466 dòng) |
| `_build_backend` | Dựng DPT-Large từ `torch.hub.load(MODEL_SOURCE, "DPT_Large", pretrained=False)` |
| `load_checkpoint` | Build backend + `_load_weights` + bọc `MiDaSModel`, đặt `model.eval()` |
| `_kitti_stem_key` | Chuẩn hoá stem ghép cặp ảnh/depth KITTI |
| `build_dataset` | `KittiDepthDataset` (image_size=None, scale_mm=1000, depth_cap_m, stem_key) |
| `evaluate_dataset` | Loop inference, median-align per image, cộng dồn metric, tính trung bình |
| `evaluate_checkpoint` | Orchestration: load model → dataset → evaluate → build_results |
| `build_results` / `write_results` | Lắp JSON chuẩn + ghi `indent=2` |
| `tests/test_evaluate_midas.py` | 26 test synthetic/mock trên CPU |

### 4.3 Configuration

`configs/midas.yaml` (tái sử dụng, không sửa trong Step 14):

- `model.variant: dpt_large`, `model.source: hub`, `model.repo: intel-isl/MiDaS`
- `preprocessing.input_size: 384`
- `depth.representation: relative`, `depth.metric_alignment: median`
- `eval.depth_cap_m: 80`
- `env.seed: 42`

CLI defaults (`build_parser`): `--config midas`,
`--checkpoint checkpoints/dpt_large_384.pt`,
`--output outputs/analysis/midas_kitti_evaluation.json`, `--split val`,
`--device auto`, `--batch-size 1`, `--align` (median|none), `--save-predictions` (off).

### 4.4 Processing Flow

1. Load config `midas`; `resolve_device(device)`.
2. `_build_backend` — dựng architecture DPT-Large (không tải weights từ mạng).
3. `_load_weights(backend, checkpoint, device)` — nạp `dpt_large_384.pt` khớp chính xác.
4. `MiDaSModel(backend=..., variant="dpt_large").to(device).eval()`.
5. `build_dataset` — KITTI val (1000 mẫu), giữ resolution gốc.
6. Với mỗi ảnh: mở RGB → `MidDepthPredictor.predict(pil)` (transform, forward `no_grad`,
   bilinear resize về resolution gốc) → `evaluate_depth(pred, gt, valid, align="median")`.
7. Cộng dồn `rmse/mae/abs_rel/δ1/δ2/δ3` và các `scale` (median) per image.
8. `metrics = sums / num_samples`; `mean_scale` = trung bình các scale.
9. `build_results` → JSON kèm `relative_inverse_depth=true`,
   `larger_value_is_closer=true`, `median_scaling=true`, `depth_cap_m`, `notes`.
10. `write_results` → lưu JSON; in RMSE/MAE/AbsRel/δ1/δ2/δ3 + median scale ra stdout.

### 4.5 Error Handling

| Tình huống | Xử lý |
|---|---|
| Checkpoint thiếu | `FileNotFoundError("Checkpoint file not found: ...")` → exit 1 |
| Không dựng được DPT_Large từ hub | `MiDaSError` → exit 1 |
| Device/config lỗi | `ConfigError`/`ValueError`/`RuntimeError`/`MiDaSError` → in `error: ...`, exit 1 |
| Ảnh mở lỗi / prediction lỗi / metric lỗi | Log warning `Skipping ...`, tăng `skipped`, không làm dừng toàn bộ |
| Metric fail (giá trị không dương) | `ValueError` từ `median_scale`/`delta_accuracy` → skip ảnh đó |
| Loader rỗng / toàn bộ fail | `RuntimeError("Evaluation produced no samples (empty dataset or all failures?).")` |
| Không có valid pixels để align | `ValueError("no valid pixels with positive prediction for median scaling")` |

---

## 5. SYSTEM BUILD FLOW

### 5.1 Workflow

1. Inspect code sẵn có: `models/midas/model.py`, `inference.py`, `preprocessing/kitti.py`,
   `evaluation/depth_metrics.py`, configs, utilities — không nhân bản.
2. Verify dataset: `data/kitti/images` = 1000, `data/kitti/depth` = 1000, pairing theo stem
   qua `_kitti_stem_key` (xác minh lại: `len(dataset)=1000`, sample khớp ảnh/depth).
3. Verify checkpoint: `dpt_large_384.pt` tồn tại và load được bằng MiDaS implementation.
4. Viết `evaluation/evaluate_midas.py`.
5. Viết `tests/test_evaluate_midas.py` (synthetic + mock, CPU, không cần checkpoint 1.4 GB).
6. Chạy focused tests, rồi full `pytest`, rồi đánh giá thật 1000 ảnh.

### 5.2 Integration

- **Reuse**: model loading qua `_load_weights`, dự đoán qua `MidDepthPredictor`, dataset qua
  `KittiDepthDataset`, metrics qua `evaluate_depth` — không viết lại formula.
- **Hub-name note** (từ docstring module): upstream MiDaS hub giờ export `DPT_Large`
  (PascalCase) trong khi `validate_variant` chuẩn hoá về `dpt_large`; vì thế module dựng
  backend trực tiếp bằng `torch.hub.load(MODEL_SOURCE, "DPT_Large", pretrained=False)` rồi
  bọc `MiDaSModel` — khớp đúng key với checkpoint, không nhân bản loading logic.
- **Không thay đổi** MiDaS model, metrics, preprocessing, hay pipeline.
- JSON đầu ra phục vụ Step 15 (pipeline evaluation) và báo cáo tổng hợp.

### 5.3 Testing

`tests/test_evaluate_midas.py` — **26 test** synthetic/mock (không chạy 1000 ảnh, không cần
checkpoint 1.4 GB trừ khi đánh dấu integration), bao gồm:

1. checkpoint path handling (`missing_raises`), model ở eval mode, weights được nạp
2. no-gradient inference + weights không đổi + gradient là `None`
3. prediction/GT shape alignment (predictor trả shape đúng GT)
4. valid depth mask được tôn trọng (pixel invalid bị loại)
5. median scaling `scale = median(gt)/median(pred)`, aligned → perfect
6. metric keys chuẩn (`rmse, mae, abs_rel, delta1, delta2, delta3`)
7. `evaluate_dataset` structure, perfect-when-constant, save `.npy` predictions, empty-raises
8. `build_results` structure + no-alignment variant
9. JSON roundtrip
10. CLI argument parsing (defaults, `--save-predictions`, `--align`)
11. integration `evaluate_checkpoint` với data giả + stub model

### 5.4 CI/CD

- **KHÔNG CÓ CI/CD** trong dự án — không có `.github/workflows`; mọi kiểm tra chạy thủ
  công qua `pytest`. Bước này không thêm CI/CD.

### 5.5 Execution / Commands

```bash
# 1) Focused tests (theo prompt Step 12)
.venv/bin/python -m pytest tests/test_evaluate_midas.py \
    tests/test_depth_metrics.py tests/test_config.py -q

# 2) Full regression
.venv/bin/python -m pytest -q

# 3) Đánh giá thật trên 1000 ảnh KITTI val
python -m evaluation.evaluate_midas \
    --config midas --checkpoint checkpoints/dpt_large_384.pt

# 4) (Tùy chọn) Lưu prediction maps `.npy` ra outputs/depth/midas_kitti/
python -m evaluation.evaluate_midas --config midas --save-predictions
```

---

## 6. EVALUATION

### 6.1 Evaluation Objective

Đo chất lượng ước lượng độ sâu của MiDaS DPT-Large trên **toàn bộ 1000 ảnh KITTI val**,
so với GT metric, sau median scaling per-image (alignment đánh giá — không phải training).

### 6.2 Evaluation Method

- Model `eval()`; toàn bộ inference trong `torch.no_grad()`.
- Mỗi ảnh: predict → median scale `scale = median(gt_valid)/median(pred_valid)` →
  `scaled = pred * scale` → `evaluate_depth(scaled, gt, valid)`.
- Metric tổng = **trung bình per-image** trên 1000 ảnh (vì mỗi ảnh align với scale riêng;
  không gộp global statistic đơn — khác với Step 13 U-Net dùng global confusion matrix).
- Chỉ tính trên valid pixels; pixel GT `0` / xa hơn `depth_cap_m=80` bị loại.

### 6.3 Results

Nguồn: `outputs/analysis/midas_kitti_evaluation.json` (đã đọc và đối chiếu trực tiếp).

**Tổng quan:**

| Field | Giá trị |
|---|---|
| model / dataset / split | MiDaS DPT-Large / KITTI / val |
| checkpoint | `checkpoints/dpt_large_384.pt` |
| num_samples / num_expected / skipped | **1000 / 1000 / 0** |
| relative_inverse_depth / larger_value_is_closer | true / true |
| metric_alignment / median_scaling | median / true |
| depth_cap_m | 80 |
| mean_scale | **0.2217** |

**Depth metrics (median-aligned):**

| Metric | Giá trị |
|---|---|
| RMSE | **4.2561 m** |
| MAE | **3.0182 m** |
| AbsRel | **0.8489** |
| δ1 (1.25^1) | **0.1671** |
| δ2 (1.25^2) | **0.3275** |
| δ3 (1.25^3) | **0.4781** |

Giải thích: RMSE/MAE là sai số tuyệt đối theo mét sau alignment (nhạy với outlier lớn); AbsRel
là sai số tương đối trung bình; δ1..δ3 là tỉ lệ pixel valid có
`max(pred/gt, gt/pred)` dưới từng ngưỡng — giá trị thấp (δ1 = 0.1671) cho thấy phần lớn error
tương đối vượt 25%.

**Tests:**

| Test | Kết quả |
|---|---|
| `pytest tests/test_evaluate_midas.py tests/test_depth_metrics.py tests/test_config.py -q` | **66 passed** in 1.58s |
| `pytest -q` (full regression) | **456 passed, 1 skipped** in 21.16s |

Ghi chú: kết quả thực tế đo được (measured), ĐẦY ĐỦ 1000 ảnh — không phải smoke test.

### 6.4 Limitations

- **Đây là đánh giá model-level depth, chưa phải pipeline-level** (Step 15).
- Median scaling chỉ căn chỉnh scale toàn cục của mỗi ảnh, không sửa được sai lệch tương đối
  cục bộ → δ1 thấp.
- Metric tính ở resolution gốc KITTI sau khi resize prediction về GT; không đánh giá timing/FPS.
- Độ chính xác giảm ở vùng xa (depth cap 80 m, sparsity của Velodyne); không có camera thật.
- Raw MiDaS output là relative inverse depth — tuyệt đối không mô tả là meters.

---

## 7. AI-ASSISTED DEVELOPMENT

### 7.1 Prompt

- File: `prompts/14_evaluate_midas.md` (STEP 14 — Evaluate MiDaS DPT-Large on KITTI).
- Yêu cầu cốt lõi: đánh giá pretrained MiDaS trên toàn bộ KITTI val; tái sử dụng model,
  KITTI loader, depth metrics; median scaling như evaluation-time alignment; JSON kết quả;
  test synthetic; không train MiDaS; full regression; STOP sau Step 14.

### 7.2 Generated / Modified Files

**Files created** (Step 14):
- `evaluation/evaluate_midas.py` — module đánh giá
- `tests/test_evaluate_midas.py` — 26 test
- `outputs/analysis/midas_kitti_evaluation.json` — kết quả thật 1000 ảnh

**Files modified**: không có (model/config/metrics/pipeline giữ nguyên).

### 7.3 Testing

- Focused: `pytest tests/test_evaluate_midas.py tests/test_depth_metrics.py tests/test_config.py -q`
  → **66 passed**
- Full: `pytest -q` → **456 passed, 1 skipped**
- Đã chạy đánh giá thật 1000 ảnh; JSON chứa đúng num_samples=1000, skipped=0.

### 7.4 Version History

| Commit | Nội dung liên quan Step 14 |
|---|---|
| `d4a21f6` | "unet module" — nền model/inference |
| `47678b0` | "Steps 12-17: unet/midas evaluation, pipeline fusion, custom demo, report" — chứa `evaluation/evaluate_midas.py`, `tests/test_evaluate_midas.py`, `midas_kitti_evaluation.json` |

*(Không tạo commit mới trong bước viết báo cáo này — chỉ kiểm chứng.)*

### 7.5 Output

- `outputs/analysis/midas_kitti_evaluation.json`
- Prediction maps chỉ được lưu khi bật `--save-predictions`
  (ra `outputs/depth/midas_kitti/*_pred.npy`) — mặc định tắt; thư mục hiện chưa tồn tại
  trong repo.

---

## 8. DISCUSSION

### 8.1 Strengths

- **Toàn tập 1000 ảnh**, tái lập được; reuse tối đa module sẵn có (không nhân bản formula).
- **Median scaling per-image** đúng chuẩn đánh giá depth relative; ghi rõ trong JSON rằng
  alignment là ở evaluation time và MiDaS không sinh metric depth.
- **An toàn dữ liệu**: pairing theo stem (không ghép theo thứ tự), bỏ pixel invalid, không
  dùng anonymous test set.
- Test không cần checkpoint 1.4 GB hay GPU; 26 test focused nhanh (< 2s).

### 8.2 Weaknesses

- Metric là per-image average; khác biệt nhỏ giữa các chiến lược aggregation (trung bình
  per-image so với gộp global statistic) — phù hợp với alignment per-image nhưng cần đọc kỹ.
- Trên CPU, predict MiDaS DPT-Large cho 1000 ảnh tốn thời gian; CUDA được hỗ trợ khi có.

### 8.3 Limitations

- Model-level evaluation, chưa đánh giá pipeline (Step 15).
- Độ sâu tương đối chỉ có nghĩa sau alignment; δ1 thấp cho thấy error tương đối lớn.
- Không đo FPS/timing; không đánh giá ngoài KITTI val.

### 8.4 What This Step Does / Does Not Prove

**Does prove**: checkpoint `dpt_large_384.pt` hoạt động với MiDaS implementation của dự án,
khớp không gian với GT KITTI, và có thể đo lường được trên 1000 ảnh với median scaling
(RMSE 4.2561 m, AbsRel 0.8489, δ1 0.1671).

**Does NOT prove**: không chứng minh MiDaS đạt độ chính xác metric depth tuyệt đối, không
khẳng định chất lượng real-time hay trên camera thật, không so sánh với các mô hình khác,
không chứng minh pipeline tổng hợp.

---

## 9. CONCLUSION

Step 14 hoàn thành mục tiêu **đánh giá độc lập MiDaS DPT-Large trên toàn bộ 1000 ảnh KITTI**:

- Module `evaluation/evaluate_midas.py` + 26 test synthetic (offline, CPU).
- Kết quả thật (median-aligned): **RMSE 4.2561 m, MAE 3.0182 m, AbsRel 0.8489,**
  **δ1 0.1671, δ2 0.3275, δ3 0.4781**, mean_scale 0.2217.
- JSON: `outputs/analysis/midas_kitti_evaluation.json` (num_samples 1000, skipped 0).
- Full regression: **456 passed, 1 skipped**.
- Không train MiDaS, không sửa model/pipeline; ngữ nghĩa relative inverse depth được giữ nguyên.

---

## 10. REFERENCES

1. Prompt Step 14 — `prompts/14_evaluate_midas.md`
2. Evaluation module — `evaluation/evaluate_midas.py`
3. Step 14 tests — `tests/test_evaluate_midas.py`
4. Depth metrics — `evaluation/depth_metrics.py`
5. MiDaS model — `models/midas/model.py`
6. MiDaS inference — `models/midas/inference.py`
7. KITTI dataset — `preprocessing/kitti.py`
8. Config — `configs/midas.yaml`
9. Checkpoint — `checkpoints/dpt_large_384.pt`
10. Evaluation output — `outputs/analysis/midas_kitti_evaluation.json`
11. Báo cáo trước có liên quan — `docs/coursework/13_evaluate_unet.md`,
    `docs/coursework/06_evaluation.md`