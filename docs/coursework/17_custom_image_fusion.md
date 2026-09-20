# Step 17 — Custom Image Fusion Demo

> Quan trọng: Step 17 là bước **demo / inference chỉ**, mục tiêu tạo một CLI thân thiện để người
> dùng bỏ ảnh street-scene RGB bất kỳ vào `data/pipeline/images/` rồi chạy toàn bộ pipeline có sẵn
> mà **không cần ground truth, không retrain, không sửa architecture, không tải model**. Kết quả
> mỗi ảnh được ghi vào `outputs/custom/` (visualization + JSON scene report).

---

## 1. REQUIREMENT

### 1.1 Problem

Sau các bước:

- **Step 11** xây dựng pipeline single-image (chạy 1 ảnh khi truyền đường dẫn ảnh cụ thể).
- **Step 13** đánh giá model-level **U-Net** trên Cityscapes GT.
- **Step 14** đánh giá model-level **MiDaS** trên KITTI GT.
- **Step 15** đánh giá full-pipeline trên 2 ảnh demo qua `evaluation/evaluate_pipeline.py`.

Tất cả các cách dùng này đều cần chỉ định đường dẫn ảnh cụ thể, hoặc được thiết kế cho benchmark.
Chưa có một **workflow dành cho người dùng cuối**: đưa một *thư mục* ảnh street-scene RGB bất kỳ
vào, tự động phát hiện ảnh, chạy cả U-Net + MiDaS trên **cùng một ảnh RGB**, rồi xuất ra báo cáo
cảnh + điểm difficulty mà người dùng không cần viết nhãn hay biết đường dẫn model.

Vấn đề của Step 17: thêm một **custom-image folder demo** (inference-only) tích hợp vào `main.py`,
theo đúng mô hình same-image contract đã có, xuất ra `outputs/custom/`.

### 1.2 Objective

1. Cho phép người dùng bỏ ảnh RGB street-scene vào `data/pipeline/images/` (không cần chỉ định
   từng ảnh).
2. Cung cấp CLI:

   ```
   python main.py --input-dir data/pipeline/images
   python main.py --input-dir data/pipeline/images --limit 1
   python main.py --input-dir data/pipeline/images --device auto
   ```

3. Tự động phát hiện ảnh (`.png/.jpg/.jpeg/.bmp/.webp`…), xử lý nhiều ảnh trong thư mục.
4. Với mỗi ảnh: **cùng một ảnh RGB** đi qua U-Net → segmentation, MiDaS → relative inverse depth,
   rồi fusion → scene analysis → difficulty analysis → visualization → JSON scene report.
5. Ghi kết quả vào `outputs/custom/<stem>_*.png` và `outputs/custom/<stem>_scene_report.json`.
6. Không retrain, không sửa U-Net/MiDaS, không tải weight tự động, không đổi metric.

### 1.3 Input

- **Thư mục đầu vào cố định:** `data/pipeline/images/` (mặc định).
- Ảnh RGB street-scene bất kỳ, các định dạng được hỗ trợ trong `evaluation/custom_demo.py`:
  `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` (kế thừa Step 15) **cộng thêm** `.webp`
  (`SUPPORTED_EXTENSIONS = STEP15 + ".webp"`).
- **Không cần** Cityscapes labels, KITTI depth, annotation hay mask.
- Checkpoint mặc định: `checkpoints/unet_cityscapes.pth` (U-Net) và
  `checkpoints/dpt_large_384.pt` (MiDaS DPT-Large).

### 1.4 Output

Với mỗi input `data/pipeline/images/<stem>.<ext>`, ghi vào `outputs/custom/`:

```
<stem>_original.png
<stem>_segmentation.png
<stem>_depth.png
<stem>_fusion.png
<stem>_overview.png
<stem>_scene_report.json
```

Ngoài ra `main.py` còn in tóm tắt (input dir, số ảnh, từng ảnh + difficulty level, output dir) ra stdout.

### 1.5 Scope

- Demo/inference folder-based trên ảnh tùy ý (custom image).
- Validation đầu vào rõ ràng (thư mục thiếu, rỗng, ảnh hỏng, ảnh grayscale…).
- Tự động tạo `outputs/custom/`, giữ nguyên cấu trúc Step 11 / Step 15.
- Tests không load MiDaS checkpoint 1.4GB (dùng predictor giả / mocked, deterministic).

### 1.6 Out of Scope

- Retrain U-Net / MiDaS, thay đổi architecture, hyperparameter.
- Tính metric supervised (mIoU/RMSE/…) trên ảnh custom — không có GT.
- Đổi difficulty formula, thay fusion network, hay đổi semantic convention.
- Sửa hành vi Step 11 (single-image), Step 13/14/15 evaluation.

---

## 2. PURPOSE & SURVEY

### 2.1 Purpose

Cho người dùng (hoặc người demo) một cách dùng **nhanh và không cần kỹ thuật**: đặt ảnh vào một
thư mục cố định, chạy một lệnh duy nhất, nhận báo cáo toàn cảnh (segmentation + relative depth +
fusion + scene analysis + difficulty). Step 17 biến pipeline vốn chỉ có trong benchmark thành một
**sản phẩm demo dùng được**.

### 2.2 Custom Image Inference

`evaluation/custom_demo.py` phối hợp (không chứa bất kỳ neural-network logic hay metric formula
nào): phát hiện ảnh → validate → gọi `evaluation.evaluate_pipeline.evaluate_single_image` (reuse
Step 15 core) → lắp ráp JSON scene report bằng `build_scene_report` → ghi 5 visualization + 1 JSON
bằng `save_custom_visualizations` (chỉ gọi các hàm có sẵn trong `visualization/`).

### 2.3 Why Custom Images?

- Steps 13–15 đánh giá trên dataset chuẩn (Cityscapes/KITTI) có GT → đo **độ chính xác model**.
- Custom image **không có GT** → chỉ minh họa toàn bộ pipeline chạy hợp lý trên ảnh ngoài phân
  phối dataset, không khẳng định độ chính xác.

### 2.4 Relation to Dataset Evaluation

| Phạm vi | Dataset evaluation (13–15) | Custom-image demo (17) |
|---|---|---|
| Ảnh đầu vào | Cityscapes / KITTI (có GT) | Ảnh RGB tùy ý (không GT) |
| Mục đích | Đo metric model-level/pipeline | Minh họa end-to-end |
| Ground truth | Có (semantic / depth) | Không |
| Output | JSON metric | Visualization + scene report |
| Ghi chú | Cityscapes & KITTI không ghép cặp | Same-image contract (1 ảnh → 2 model) |

---

## 3. DATA

### 3.1 Custom Image Input

Input mặc định: `data/pipeline/images/`. Hiện tại chứa **4 ảnh RGB street-scene**:

| File | Kích thước ảnh (H×W) |
|---|---|
| `gta5.png` | 900 × 1600 |
| `pipeline_demo_001.png` | 352 × 1216 |
| `pipeline_demo_002.png` | 352 × 1216 |
| `streetest.png` | 2000 × 3000 |

Kích thước lấy từ field `image.height`/`image.width` của 4 `*_scene_report.json` trong
`outputs/custom/` (ảnh chưa qua resize — pipeline chạy ở resolution gốc, xem §3.3).

### 3.2 Image Format

- Định dạng hỗ trợ (mã nguồn `evaluation/custom_demo.py`): `.png`, `.jpg`, `.jpeg`, `.bmp`,
  `.tiff`, `.webp` (Set của Step 15 + `.webp`).
- Phát hiện qua đuôi file (`suffix.lower()`); các file khác bị **bỏ qua (không xử lý)**.
- Ảnh được đọc bằng `PIL.Image` và chuyển sang **RGB** (`convert("RGB")`).

### 3.3 Input Resolution

Pipeline chạy ở **resolution gốc của từng ảnh**. `evaluate_single_image` đọc kích thước thật và
`_resize_nearest_ids` (segmentation) / `_resize_bilinear` (depth) căn chỉnh về đúng kích thước
đó. Riêng U-Net inference dùng `_safe_unet_inference_size` chọn resolution an toàn với VRAM rồi
upscale về kích thước gốc sau đó (bounding inference resolution).

### 3.4 Preprocessing

- Ảnh được đọc 1 lần duy nhất → cùng một mảng `np_image` đi vào cả U-Net lẫn MiDaS
  (photographic same-image contract, §4.3).
- Preprocessing model-specific được ủy thác cho `models/unet` và `models/midas` (reuse, không
  viết lại trong `main.py`).
- Grayscale (`L`, `I`, `I;16`, `F`) → chuyển RGB kèm **warning**, theo convention dự án.
- Không có ground-truth nào được đọc.

### 3.5 Dataset vs Custom Image

- Cityscapes/KITTI dùng cho **đánh giá model** (Steps 13–14), có GT, có metric.
- Custom images dùng cho **minh họa pipeline** (Step 17), không GT, không metric.
- Không có mapping/ghép cặp nào giữa custom images và Cityscapes/KITTI.

---

## 4. IMPLEMENTATION PLAN

### 4.1 Pipeline Architecture

```
custom RGB image
        |
        +----> U-Net (checkpoints/unet_cityscapes.pth)
        |        |
        |        └──> semantic segmentation (trainId, 19 lớp)
        |
        +----> MiDaS DPT-Large (checkpoints/dpt_large_384.pt)
                 |
                 └──> relative inverse depth (larger = closer)
                          |
                          v
                    Alignment (resize về kích thước ảnh gốc)
                          |
                          v
                 Fusion (rule-based, KHÔNG phải network học)
                          |
                          v
                 Scene Understanding (analyzer)
                          |
                          v
                 Difficulty Analysis (heuristic)
                          |
                          v
                 Visualization (5 ảnh) + JSON scene report
```

### 4.2 U-Net Inference

- Model wrapper `models.unet.inference.UNetInference.from_config`, checkpoint
  `checkpoints/unet_cityscapes.pth` (mặc định), device theo `--device`.
- Inference được đảm bảo ở chế độ eval, qua hàm `_load_unet` (reuse Step 15) — sử dụng
  `_safe_unet_inference_size` để giới hạn resolution trong lúc forward.
- Không thay đổi architecture, không retrain.

### 4.3 MiDaS Inference

- `models.midas.model.build_midas_model(variant=...)` + `models.midas.inference.MidDepthPredictor`,
  checkpoint `checkpoints/dpt_large_384.pt` (mặc định).
- Output là **relative inverse depth** — convention dự án:
  `DEPTH_CONVENTION = "inverse_relative_larger_closer"` (giá trị lớn = gần). **Không bao giờ**
  mô tả raw MiDaS output là depth mét.
- Không tải weight từ mạng; checkpoint thiếu → lỗi rõ tên file.

### 4.4 Fusion

- Reuse `scene_understanding.fusion.fuse` (rule-based, không phải learned network); không đổi.
- Depth map chia 3 region `far/middle/near` bằng tercile (`depth_thresholds.low/high` trong JSON);
  informed bởi class semantic từ segmentation.

### 4.5 Scene Understanding

- Reuse `scene_understanding.analyzer.analyze_fusion(...)` (via `evaluate_single_image`):
  semantic distribution, depth distribution, per-region depth stats, traffic_context
  (vehicles/pedestrians/road, `nearest_dynamic_class`, `dynamic_object_count`,
  `drivable_coverage_ratio`, `drivable_median_depth`) + câu `interpretation` sinh ngôn ngữ tự
  nhiên.

### 4.6 Difficulty Analysis

- Reuse `evaluation.difficulty_analysis.compute_difficulty` — heuristic tổ hợp tuyến tính các
  indicator: `segmentation_uncertainty`, `depth_variation`, `scene_complexity`,
  `foreground_fraction`, `object_density`; trọng số/ngưỡng từ `configs/pipeline.yaml`
  (`easy` ≤ 0.4, `medium` ≤ 0.7, còn lại `hard`).
- Điểm difficulty là **heuristic**, **không phải** ground-truth human difficulty; không đổi công
  thức.

### 4.7 Visualization

Reuse các hàm có trong `visualization/`:

| Output | Hàm reuse |
|---|---|
| `<stem>_original.png` | `save_visualization(array)` |
| `<stem>_segmentation.png` | `colorize_segmentation` |
| `<stem>_depth.png` | `colorize_depth` |
| `<stem>_fusion.png` | `create_fusion_overlay(np_image, segmentation, depth, alpha_seg, alpha_depth)` |
| `<stem>_overview.png` | `create_full_visualization` + `save_figure` |
| `<stem>_scene_report.json` | `save_scene_report` |

Title depth: các figure dùng đúng thuật ngữ "Relative inverse depth (larger = closer)".

### 4.8 Output Handling

- Output mặc định `outputs/custom/` (`DEFAULT_OUTPUT_DIR`); tự động `mkdir(parents=True,
  exist_ok=True)`.
- Tên file theo `<stem>_<kind>`; không overwrite kết quả của input file khác nhau.
- `--output-dir` cho phép đổi nơi ghi (optional). `--no-visualization` chỉ ghi JSON scene report.
- Summary trả về JSON-serializable (input_dir, output_dir, num_images, từng ảnh + paths + score).

---

## 5. SYSTEM BUILD FLOW

### 5.1 End-to-End Workflow

`main.py` → `_run_batch_demo` → `evaluation.custom_demo.run_custom_demo`:

1. `validate_checkpoints` — kiểm tra 2 checkpoint tồn tại (lỗi nêu rõ file thiếu).
2. `resolve_path(input_dir)` + kiểm tra thư mục tồn tại.
3. `discover_images(input_dir, limit)` — tìm file có đuôi hợp lệ, **sort**, cắt theo `--limit`.
4. Rỗng hoặc không có ảnh hỗ trợ → `CustomDemoError` rõ ràng.
5. `custom_out.mkdir(parents=True, exist_ok=True)`.
6. Build predictors (thật hoặc dummy từ test), `set_seed(42)`.
7. Với từng ảnh: `load_rgb_image` (validate) → `evaluate_single_image` (same image → 2 model) →
   `build_scene_report` → `save_custom_visualizations`.
8. In summary, trả exit code 0; mọi lỗi pipeline bị bắt → `error: ...` + exit 1.

### 5.2 CLI Execution

```
.venv/bin/python main.py --input-dir data/pipeline/images
.venv/bin/python main.py --input-dir data/pipeline/images --limit 1
.venv/bin/python main.py --input-dir data/pipeline/images --device auto
.venv/bin/python main.py --input-dir data/pipeline/images --output-dir PATH --no-visualization
```

Single-image backward-compat (Step 11): `--image/-i`, `--unet-checkpoint`, `--midas-weights`.

### 5.3 Folder Processing

- Nhiều ảnh: xử lý lần lượt tất cả ảnh hợp lệ trong thư mục.
- `--limit N`: dừng sau N ảnh (chỉ dùng test/demo nhanh).
- Mỗi ảnh ra đúng bộ 6 file với stem riêng; không overwrite chéo.

### 5.4 Error Handling

- Thiếu thư mục input → "input directory not found: ...".
- Rỗng / chỉ có file không hỗ trợ → "no supported images found in ... (supported extensions: ...)".
- Ảnh hỏng / không đọc được → `cannot read image file <path>: <exc>`.
- Shape sai (ndim ≠ 3, channels ≠ 3), ảnh zero-size → `CustomDemoError` tên file.
- Grayscale → chuyển RGB + cảnh báo (không im lặng xử lý sai).
- Checkpoint thiếu → tên file cụ thể; lỗi pipeline bọc trong `failed to process <name>: ...`.
- Không tải weight từ mạng trong trường hợp nào.

### 5.5 Testing

- **Focused:** `tests/test_custom_demo.py` (516 dòng, 28 tests) dùng `DummySegPredictor` /
  `DummyDepthPredictor` (không load MiDaS 1.4GB). Kết quả: **28 passed**.
- **Full regression:** `pytest -q` → **456 passed, 1 skipped**.
- Phạm vi test: CLI parse (input-dir/image/both/neither), thư mục thiếu/rỗng/unsupported-only,
  discovery, `--limit 1`, nhiều ảnh, output-dir auto-create, tên file đúng, cùng 1 image object
  vào 2 predictor, scene-report JSON, grayscale convert, corrupt file, checkpoint thiếu, regression
  Step 11 + Step 15.

### 5.6 CI/CD

**CI/CD không được triển khai** trong dự án. Mọi kiểm tra chạy thủ công qua pytest cục bộ.

---

## 6. EVALUATION / DEMONSTRATION

### 6.1 Demonstration Objective

Minh họa **qualitative end-to-end** (không phải benchmark): ảnh street-scene bất kỳ đi qua
segmentation + relative depth + fusion + scene analysis + difficulty, xuất artifacts đầy đủ.
Không tính accuracy — không có ground truth cho ảnh custom.

### 6.2 Input Images

4 ảnh hiện có trong `data/pipeline/images/`: `gta5.png` (900×1600), `pipeline_demo_001.png`
(352×1216), `pipeline_demo_002.png` (352×1216), `streetest.png` (2000×3000). Đây là
**smoke test/demonstration trên 4 ảnh**, không phải benchmark thống kê.

### 6.3 Generated Outputs

`outputs/custom/` có **24 file** (6 file × 4 ảnh): với mỗi ảnh đủ `_original`, `_segmentation`,
`_depth`, `_fusion`, `_overview`, `_scene_report.json`.

### 6.4 Observed Results

Số liệu theo đúng `outputs/custom/*_scene_report.json` (không GT; depth là **relative inverse
depth**, không phải mét):

| Ảnh | Classes | mean_conf | depth mean (inv) | depth std | depth min | depth max | Difficulty score | Level | Nearest dynamic |
|---|---|---|---|---|---|---|---|---|---|
| gta5 (900×1600) | 16 | 0.6435 | 12.5216 | 9.9814 | 0.0 | 33.7422 | 0.576852 | medium | rider |
| pipeline_demo_001 (352×1216) | 19 | 0.74121 | 13.7231 | 9.5462 | 0.4330 | 45.7387 | 0.501527 | medium | person |
| pipeline_demo_002 (352×1216) | 18 | 0.7423 | 13.2789 | 8.5380 | 0.8087 | 32.9058 | 0.47926 | medium | car |
| streetest (2000×3000) | 16 | 0.6996 | 11.7819 | 9.2399 | 0.0 | 35.9845 | 0.55035 | medium | person |

Difficulty components (heuristic):

- `gta5`: {seg_uncert 0.3565, depth_var 0.7971, complexity 0.8421, fg_fraction 0.3333, obj_density 0.6213}
- `pipeline_demo_001`: {0.25879, 0.69563, 1.0, 0.3333, 0.20691}
- `pipeline_demo_002`: {0.2577, 0.6430, 0.9474, 0.3333, 0.2006}
- `streetest`: {0.3004, 0.7842, 0.8421, 0.3333, 0.5554}

Depth regions mọi ảnh chia ~1/3–1/3–1/3: far/middle/near đều ≈ 0.3333 do
ngưỡng tercile dẫn xuất từ chính depth map mỗi ảnh. `traffic_context` (pipeline_demo_001):
`dynamic_object_count` 8, `nearest_dynamic_class` = person (near), `drivable_coverage_ratio`
0.1685. Cả 4 ảnh xếp **medium** — đây chỉ là nhận định heuristic của hệ thống.

### 6.5 Limitations

- **Qualitative demo trên 4 ảnh**, không phải benchmark thống kê; không suy luận accuracy.
- Raw MiDaS = relative inverse depth, **không** thể hiện khoảng cách mét.
- Difficulty là heuristic có trọng số do con người chọn; không phải ground-truth difficulty.
- Kết quả phụ thuộc checkpoint hiện có; U-Net yếu ở các lớp hiếm (mô tả Step 13).

---

## 7. AI-ASSISTED DEVELOPMENT

### 7.1 Prompt

`prompts/17_custom_image_fusion.md` (506 dòng) — đặc tả yêu cầu: input folder cố định
`data/pipeline/images/`, CLI `--input-dir`, `--limit`, `--device auto`, các checkpoint có sẵn,
same-image contract, image validation, naming output `outputs/custom/<stem>_*`, tôn trọng
"Relative inverse depth (larger = closer)", scene report JSON, difficulty heuristic giữ nguyên,
tests không tải MiDaS, integration vào `main.py`, cập nhật README, regression safety.

### 7.2 Generated / Modified Scripts

- Tạo mới: `evaluation/custom_demo.py` (365 dòng), `tests/test_custom_demo.py` (516 dòng),
  `prompts/17_custom_image_fusion.md`.
- Modified: `main.py` (thêm `--input-dir`, `--limit`, `_run_batch_demo`, backward-compat),
  `README.md` (section "## Custom Image Fusion Demo (Step 17)").
- Reuse (không sửa): `evaluation/evaluate_pipeline.py`, `scene_understanding/`,
  `evaluation/difficulty_analysis.py`, `visualization/`, `models/unet`, `models/midas`.

### 7.3 Testing

- Focused `tests/test_custom_demo.py`: **28 passed** (7.09s).
- Full `pytest -q`: **456 passed, 1 skipped** (21.22s).
- Không có test nào load checkpoint MiDaS thật.

### 7.4 Version History

- `47678b0` "Steps 12-17: unet/midas evaluation, pipeline fusion, custom demo, report" — chứa
  toàn bộ milestone implementation (kể cả custom demo).
- `827ab02` "add rp step 15-16" — nhóm các báo cáo step.
- (không có commit mới cho riêng Step 17 trong phiên này — chỉ tạo report)

### 7.5 Output

`outputs/custom/` (24 file) + summary stdout; tài liệu `prompts/17_custom_image_fusion.md`,
README section, test suites.

---

## 8. DISCUSSION

### 8.1 Strengths

- Không cần đường dẫn ảnh/GT — người dùng chỉ bỏ ảnh vào thư mục rồi chạy 1 lệnh.
- Reuse toàn bộ module có sẵn (evaluate_pipeline, visualization, analyzer, difficulty) → không
  trùng lặp logic, không lệch metric.
- Same-image contract được giữ chặt (1 mảng RGB duy nhất vào cả 2 model).
- Validation rõ ràng, thông báo lỗi tên file cụ thể, grayscale xử lý theo convention.
- Tests deterministic, không tải weight nặng, giữ regression 456 passed.

### 8.2 Weaknesses

- Chỉ hỗ trợ folder-based (`--input-dir`) cho custom demo; không có `--image` cho đường dẫn tùy ý
  (cố ý theo yêu cầu, single-image vẫn dùng `--image` riêng).
- Với ảnh lớn (streetest 3000×2000) resolution gốc có thể chậm; U-Net bị bound bởi
  `_safe_unet_inference_size` trước khi upscale.
- Không có progress/ETA; log chỉ dạng text từng ảnh.

### 8.3 Limitations

- Không có ground truth cho ảnh custom → không thể nói pipeline "đúng" bao nhiêu.
- Difficulty là heuristic thay đổi theo trọng số con người.
- 4 ảnh demo không đại diện thống kê cho mọi street scene.

### 8.4 What This Step Demonstrates

Cách dùng pipeline trong thực tế: một ảnh RGB duy nhất sinh ra segmentation, relative inverse
depth, fusion, scene report và difficulty — end-to-end, không cần annotation, tái lập được qua
1 CLI lệnh.

### 8.5 What This Step Does Not Prove

Không chứng minh độ chính xác của U-Net/MiDaS trên ảnh custom (không GT); không chứng minh
difficulty khớp difficulty thật của con người; không phải benchmark.

---

## 9. CONCLUSION

Step 17 đã biến pipeline scene-understanding thành một **custom-image folder demo** dùng được:
người dùng bỏ ảnh RGB street-scene vào `data/pipeline/images/` và chạy
`python main.py --input-dir data/pipeline/images`; hệ thống tự phát hiện ảnh, chạy U-Net →
segmentation và MiDaS → relative inverse depth trên **cùng một ảnh**, thực hiện fusion →
scene analysis → difficulty analysis, ghi 24 artifact (5 visualization + 1 JSON × 4 ảnh) vào
`outputs/custom/`. Toàn bộ logic reuse module sẵn có, không retrain, không tải weight, không đổi
metric. Focused tests 28 passed, full suite 456 passed / 1 skipped. Step 17 chứng tỏ khả năng
**đóng gói và trình diễn** pipeline — bổ sung cho các đánh giá định lượng từ Steps 13–15.

---

## 10. REFERENCES

1. Prompt Step 17: `prompts/17_custom_image_fusion.md`.
2. Custom demo: `evaluation/custom_demo.py`; core pipeline: `evaluation/evaluate_pipeline.py`.
3. CLI: `main.py` (`--input-dir`, `--limit`, `--device`, `--output-dir`, `--no-visualization`).
4. Model wrappers: `models/unet/inference.py`, `models/midas/inference.py`,
   `models/midas/model.py`.
5. Scene understanding: `scene_understanding/fusion.py`, `scene_understanding/analyzer.py`;
   difficulty: `evaluation/difficulty_analysis.py`.
6. Visualization reuse: `visualization/{io,segmentation,depth,fusion,scene}.py`.
7. Outputs thực tế: `outputs/custom/*_scene_report.json` và các PNG tương ứng.
8. Tests: `tests/test_custom_demo.py`; README section "Custom Image Fusion Demo (Step 17)".
9. Cross-reference: Step 13 (U-Net eval), Step 14 (MiDaS eval), Step 15 (pipeline eval).