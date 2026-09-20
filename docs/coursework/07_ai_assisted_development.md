# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

## Coursework Report — Step 07: AI-Assisted Development

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Tài liệu này mô tả **cách dự án sử dụng AI để hỗ trợ phát triển** (AI-assisted development): từ prompt → OpenCode/AI triển khai → code/tài liệu → test → validation → Git history / output.
> Nguồn sự thật: 18 file prompt `prompts/*.md`, bộ báo cáo Markdown `docs/coursework/*.md` (00–07), Git history 15 commit, `README.md`, bộ pytest hiện tại (456 passed, 1 skipped) và các output thực tế trong `outputs/`.
> Không có thông tin nào trong tài liệu này được bịa đặt; mọi nội dung không verify được được ghi rõ **"not explicitly available in repository"**.

---

## 1. REQUIREMENT

### 1.1 Purpose

AI-assisted development được ghi chép trong coursework vì:

- Cần chứng minh **tính minh bạch**: AI được dùng như công cụ hỗ trợ, không phải thay thế vai trò của sinh viên;
- Cần một **quy trình tái lập được**: đầu vào (prompt) → quá trình (AI + người) → đầu ra (code/tài liệu/test/output) có thể kiểm tra chéo qua Git và filesystem;
- Giúp sinh viên trình bày được **AI đã hỗ trợ gì** và **con người chịu trách nhiệm gì** trong từng bước.

### 1.2 Development Requirement

Yêu cầu phát triển được thể hiện qua từng file prompt trong `prompts/`:

- `prompts/00` … `prompts/17` — 18 task brief, mỗi file là **một bước phát triển rõ ràng** (00 kiến trúc → 17 custom demo);
- Mỗi prompt nêu **scope (làm gì)** và **anti-scope (cấm làm gì)** — ví dụ Step 06 yêu cầu "Implement evaluation metrics… Do NOT implement: U-Net training, MiDaS, depth… full pipeline, CI/CD".

Yêu cầu truy xuất (traceability) được đối chiếu với repository:

| Loại | Bằng chứng trong repository |
|---|---|
| Prompt | `prompts/00_*.md` … `prompts/17_*.md` (18 file) |
| Generated/modified scripts | `models/`, `preprocessing/`, `scene_understanding/`, `evaluation/`, `visualization/`, `utils/`, `training/`, `configs/` |
| Testing | `tests/` (20 file test + `conftest.py`) |
| Validation | Kết quả pytest, file JSON `outputs/analysis/*.json`, hình ảnh `outputs/*` |
| Version history | Git log (15 commit), `git show --stat` |
| Documentation | `docs/coursework/` (7 báo cáo) + `docs/report/coursework_report.md` |

### 1.3 Traceability

Mối quan hệ tổng quát, được hỗ trợ bởi chính cấu trúc repository:

```text
Prompt
  ↓
AI-assisted implementation (OpenCode + review của sinh viên)
  ↓
Source Code / Documentation
  ↓
Testing (unit / smoke / full suite)
  ↓
Validation (so kết quả với kỳ vọng, pytest xanh)
  ↓
Git History / Output
```

Không phải mọi bước đều có commit riêng (xem §6.2). Với bước code, bằng chứng chính là **file nguồn + test + output JSON**; với bước tài liệu (00, 01, 16), đầu ra là **Markdown**, không có test đi kèm.

---

## 2. AI-ASSISTED DEVELOPMENT WORKFLOW

### 2.1 Overall Workflow

Quy trình phát triển (đúng như cách các prompt tổ chức):

```text
Requirement
    ↓
Prompt (task brief trong prompts/)
    ↓
OpenCode / AI-assisted implementation
    ↓
Source Code / Documentation
    ↓
Testing (pytest)
    ↓
Validation (kiểm tra kết quả, smoke test, import test)
    ↓
Git History / Output (JSON, PNG, report)
```

Giải thích từng bước:

- **Requirement** — yêu cầu tổng thể của coursework (phân đoạn + depth → scene understanding), mô tả trong `README.md` và `docs/coursework/00`–`01`.
- **Prompt** — mỗi bước được gói thành một task brief trong `prompts/`, nêu rõ mục tiêu, file cần tạo, phạm vi bị cấm, và cách kiểm chứng.
- **OpenCode / AI-assisted implementation** — AI đề xuất/viết code theo prompt; sinh viên **xem lại và điều chỉnh** trước khi chấp nhận (xem §2.2).
- **Source Code / Documentation** — đầu ra là mã nguồn (bước kỹ thuật) hoặc tài liệu Markdown (bước báo cáo).
- **Testing** — bộ `tests/` chạy được offline, không cần dataset/GPU/internet.
- **Validation** — import test, smoke inference, kiểm tra ranh giới (ignore 255, undefined class…), đối chiếu con số JSON với code.
- **Git History / Output** — kết quả được commit và/hoặc ghi ra `outputs/` (JSON đánh giá, PNG visualization, scene report).

### 2.2 Role of the Student

AI là **công cụ hỗ trợ (assistance tool)**, không phải người quyết định. Sinh viên chịu trách nhiệm:

- **Định nghĩa requirement** — viết/tinh chỉnh prompt, phạm vi in-scope / out-of-scope;
- **Xem lại code AI tạo ra** — `git diff`, đọc lại logic, đảm bảo đúng convention (không đổi design, không thêm thư viện thừa);
- **Kiểm tra tính đúng đắn** — đối chiếu công thức (IoU/Dice/median scaling…), đối chiếu số liệu JSON;
- **Chạy test** — `pytest`, smoke test, import test;
- **Validate kết quả** — xác nhận khớp với yêu cầu coursework specification;
- **Quyết định** — chấp nhận thay đổi của AI hay yêu cầu sửa, trước khi commit.

**Không có khẳng định nào "AI tự động phát triển toàn bộ"** — repository không lưu bản ghi AI làm việc độc lập; mọi thay đổi đi qua review và commit của con người.

### 2.3 Role of OpenCode

Dựa trên `prompts/`, `docs/coursework/` và Git history, OpenCode đã tham gia các hoạt động verify được:

- **Đọc/khảo sát repository** — prompt yêu cầu đọc các step trước và module hiện hữu (Step 06 mục 1 "Read Existing Specifications");
- **Triển khai module theo yêu cầu** — viết `models/unet/model.py`, `evaluation/segmentation_metrics.py`, `evaluation/depth_metrics.py`, …;
- **Viết test** — tạo các file `tests/test_*.py` với dữ liệu synthetic nhỏ;
- **Sửa bug / cải tiến** — sửa logic khi review (ghi nhận chung trong các báo cáo trước; repository không có log bug riêng của OpenCode);
- **Sinh tài liệu** — tạo các báo cáo Markdown `docs/coursework/00`–`06`;
- **Chạy lệnh validation** — pytest, import test, smoke inference, đọc/trình bày kết quả JSON.

Không có yêu cầu nào vượt quá những gì prompt và lịch sử repo cho phép khẳng định.

---

## 3. PROMPT STRUCTURE

### 3.1 Prompt Design

Các prompt trong `prompts/` chia làm **nhóm kỹ thuật** (`02`–`15`, `17`) và **nhóm tài liệu** (`00`, `01`, `16`). Riêng `prompts/16_coursework_report.md` tự mô tả: *"File này gộp cả Prompt cho OpenCode và Report Template"*.

Các thành phần recurring (chỉ những thành phần thực sự hiện diện trong prompt):

| Thành phần | Ví dụ thực tế |
|---|---|
| **Objective / Goal** | "Implement U-Net semantic segmentation model…" (Step 04); "Implement depth evaluation for the MiDaS + KITTI pipeline" (Step 08) |
| **Scope / Implementation Location** | "Implement segmentation metrics in `evaluation/segmentation_metrics.py`" (Step 06) |
| **Anti-scope (Do NOT)** | Step 06: "Do NOT implement: U-Net training, MiDaS, depth… CI/CD"; Step 00: "Do NOT write implementation code" |
| **Read Existing Specifications** | Step 06 mục 1: "Read prompts/01..05" trước khi làm |
| **Testing / Tests** | Mỗi prompt kỹ thuật liệt kê test tối thiểu (Step 06: Test 1..13); yêu cầu "Run `pytest`, all previous tests must continue to pass" |
| **Validation** | "Run… import test", "smoke test", "Do not return NaN unexpectedly", "manually calculable example" |
| **Output / Report** | Mỗi prompt kết thúc bằng yêu cầu "Files created/modified, Test results, Problems…" |
| **Final Response / STOP** | "STOP after Step 06. Do NOT continue to Step 07." (Step 06) |

Không phải mọi prompt đều giống nhau: nhóm tài liệu cấm viết code; nhóm kỹ thuật buộc viết code + test; Step 16 chỉ định rõ cấu trúc report.

### 3.2 Step-by-Step Prompting

Dự án được phát triển **theo từng bước (incremental)**, không yêu cầu AI làm mọi thứ cùng lúc. Trình tự các prompt hiện có:

```text
Batch tài liệu:   00 kiến trúc → 01 architecture spec → 16 report template
Batch kỹ thuật:   02 config/utils → 03 dataset → 04 unet → 05 unet inference
                  → 06 seg metrics → 07 midas → 08 depth metrics
                  → 09 fusion/scene → 10 visualization → 11 demo pipeline
                  → 12 train unet → 13 eval unet → 14 eval midas
                  → 15 pipeline eval → 17 custom demo
```

Mỗi step kỹ thuật **phụ thuộc step trước** (đọc 01/02/03 trước khi làm 04, đọc 01..05 trước khi làm 06…) và có **anti-scope** để không trùng lặp công việc của step khác. Không có prompt cho Step 18 trong repository (18 file = Step 00–17).

### 3.3 Example Prompt

Lấy **Step 06** (`prompts/06_segmentation_evaluation.md`) làm ví dụ đại diện:

| Phần | Nội dung thực tế (rút gọn) |
|---|---|
| Title | `# Step 06 — Semantic Segmentation Evaluation` |
| Objective | "Implement evaluation metrics for the U-Net semantic segmentation task" |
| Do NOT list | training / MiDaS / depth / fusion / scene / visualization / pipeline / CI-CD |
| Mục 1 | **Read Existing Specifications**: đọc prompts 01..05, inspect `evaluation/`, `models/unet/`, `tests/`, `configs/` |
| Mục 2 | **Implementation Location**: `evaluation/segmentation_metrics.py` + `tests/test_segmentation_metrics.py` |
| Mục 3–15 | Metric đặc tả: Pixel Accuracy, IoU, mIoU, Dice, ignore 255, undefined-class convention, **reusable confusion matrix**, API, input validation, Tensor/NumPy, numerical stability, 13 test bắt buộc |
| Mục 16–18 | Manual verification, "pytest — all previous tests must continue to pass", import test |
| Mục 20 | **Report**: files created/modified, metric API, test results, assumptions |
| Cuối | "STOP after Step 06. Do NOT continue to Step 07." |

Ví dụ thứ hai là **Step 00** (`prompts/00_project_architecture.md`) — nhóm tài liệu: cấm viết code, yêu cầu đưa ra spec nhiều mục (Project Overview → Final Architecture Recommendation), dùng tables + Mermaid. Đây là 2 đại diện cho 2 loại prompt của dự án.

---

## 4. GENERATED / MODIFIED SCRIPTS

### 4.1 Implementation Outputs

Các file code dưới đây hiện diện trong repository và khớp với **phạm vi của từng prompt**:

| Bước (prompt) | File chính | Trách nhiệm |
|---|---|---|
| 02 | `utils/config.py`, `utils/seed.py`, `utils/logger.py`, `utils/device.py`; `configs/{unet,midas,pipeline}.yaml` | Cấu hình + tiện ích dùng chung |
| 03 | `preprocessing/{cityscapes,kitti,transforms,errors}.py` | Dataset loaders + transforms (stem-pair, không ghép sai ảnh) |
| 04 | `models/unet/model.py` | U-Net: DoubleConv/Down/Up + `from_config` |
| 05 | `models/unet/inference.py` | `UNetInference` wrapper (resize, quantize, logits raw) |
| 06 | `evaluation/segmentation_metrics.py` | Pixel Acc / IoU / mIoU / Dice từ reusable confusion matrix |
| 07 | `models/midas/{model,inference}.py` | MiDaS DPT-Large backend + `MidDepthPredictor` |
| 08 | `evaluation/depth_metrics.py` | RMSE/MAE/AbsRel/δ1..3 + `median_scale` |
| 09 | `scene_understanding/{fusion,analyzer}.py` | Fusion rule-based + scene analysis |
| 10 | `visualization/{segmentation,depth,fusion,scene,io}.py` | Colorize depth/seg + overlay + overview + save |
| 11 | `scene_understanding/pipeline.py`, `main.py` | Pipeline tổng + CLI orchestrator |
| 12 | `training/{trainer,train_unet}.py` | Training loop + checkpoint |
| 13 | `evaluation/evaluate_unet.py` | Eval U-Net trên Cityscapes val → JSON |
| 14 | `evaluation/evaluate_midas.py` | Eval MiDaS trên KITTI val → JSON |
| 15 | `evaluation/{evaluate_pipeline,difficulty_analysis,pipeline_metrics}.py` | Pipeline eval + difficulty score heuristic |
| 17 | `evaluation/custom_demo.py`, demo nhánh `main.py` | Custom street-image demo → `outputs/custom/` |

**Lưu ý trung thực:** việc gán file cho bước dựa trên **nội dung prompt + file hiện có + commit message**; repository không lưu log "AI đã tạo file X lúc Y". Các file do prompt có phạm vi rõ ràng sinh ra (`segmentation_metrics.py`…) được công nhận là output của bước tương ứng; các file nền tảng như `utils/` cũng có thể đã được viết lại nhiều lần trong quá trình phát triển.

### 4.2 Documentation Outputs

Tài liệu Markdown coursework nằm trong `docs/coursework/` (7 báo cáo từ bước 00–06). Ngoài ra còn `docs/15-pipeline-evaluation.md` (tài liệu kỹ thuật Step 15) và `docs/report/coursework_report.md` (draft báo cáo tổng). Các báo cáo coursework là **artifact riêng cho từng bước**, được commit trong các commit tách biệt (xem §6.2).

### 4.3 Generated Analysis Outputs

Các output thực tế từ các bước chạy (verified tồn tại):

| Loại | File |
|---|---|
| JSON đánh giá | `outputs/analysis/unet_cityscapes_evaluation.json`, `outputs/analysis/midas_kitti_evaluation.json`, `outputs/analysis/pipeline_evaluation.json` |
| JSON training history | `outputs/analysis/unet_training_history.json` (19 epochs, best_epoch 20, best val_miou 0.4262) |
| Scene report | `outputs/analysis/pipeline_demo_00{1,2}_scene_report.json`, `outputs/custom/*_scene_report.json` |
| Visualization | `outputs/analysis/*_fusion.png`, `*_overview.png`; `outputs/segmentation/*`, `outputs/depth/*`, `outputs/custom/*` |
| Smoke visualization | `outputs/visualization/smoke/` (scene_segmentation/depth/fusion/overview.png + report) |

`docs/coursework/06_evaluation.md` trình bày chi tiết các con số trong khối JSON này — tại đây chỉ xác nhận chúng là output thực tế của workflow AI-assisted.

---

## 5. TESTING

### 5.1 Unit Tests

Bộ test `tests/` (20 file test + `conftest.py`) validate code được tạo/sửa bởi AI. Các test dùng **synthetic fixtures** (không tải Cityscapes/KITTI, không cần weights, GPU hay internet) — đúng yêu cầu của prompt.

Số lượng test collected theo file (đo từ `pytest --collect-only`):

| Test file | Count | Test file | Count |
|---|---|---|---|
| test_visualization.py | 52 | test_kitti.py | 18 |
| test_evaluate_pipeline.py | 42 | test_config.py | 18 |
| test_midas.py | 34 | test_unet.py | 17 |
| test_unet_inference.py | 31 | test_evaluate_unet.py | 17 |
| test_depth_metrics.py | 29 | test_cityscapes.py | 17 |
| test_custom_demo.py | 28 | test_training.py | 16 |
| test_pipeline.py | 27 | test_logger.py | 8 |
| test_fusion.py | 27 | test_device.py | 6 |
| test_segmentation_metrics.py | 26 | test_seed.py | 4 |
| test_scene_analyzer.py | 21 | test_evaluate_midas.py | 19 |

Tổng thu được: **457 test collected** (456 passed, 1 skipped).

### 5.2 Integration / Smoke Tests

Các smoke/integration test verified:

- **Import test** — `python -c "from evaluation.segmentation_metrics import *; print('Segmentation metrics OK')"` (prompt 06 yêu cầu);
- **Synthetic inference** — `tests/test_unet_inference.py` (31 test) chạy forward trên tensors nhỏ; smoke inference trong `docs/coursework/05` (prediction `(256,512)`, min 3, max 18, `torch.long`);
- **Pipeline smoke** — `outputs/visualization/smoke/` và `outputs/custom/` là bằng chứng pipeline chạy thật trên ảnh (`data/pipeline/images/`);
- **Evaluation scripts** — `tests/test_evaluate_unet.py`, `test_evaluate_midas.py`, `test_evaluate_pipeline.py` test logic eval offline.

### 5.3 Full Test Suite

Kết quả toàn suite (**chạy lại tại thời điểm viết doc này**):

```text
.venv/bin/python -m pytest -q
456 passed, 1 skipped in 22.71s
```

(1 skip = test phụ thuộc CUDA, không có trên máy chạy.)

### 5.4 Regression Testing

Các prompt kỹ thuật đều yêu cầu **"Run pytest — all previous tests must continue to pass"** và **"Do not remove or weaken existing tests"** (ví dụ Step 06 mục 17). Nhờ đó mỗi bước mới phải giữ các test cũ (Step 02..05 trở đi) vẫn xanh — số test tăng dần theo bước mà không có test cũ bị xóa/yếu đi.

---

## 6. VERSION HISTORY

### 6.1 Git Usage

Repository dùng Git để:

- **Theo dõi thay đổi** — 15 commit từ 2026-08-17 đến 2026-09-20;
- **Xem lại lịch sử** — `git log`, `git show --stat` để xác định file thay đổi từng commit;
- **Nhận diện file modified** — phân biệt commit code và commit tài liệu;
- **Lưu lại các giai đoạn** — code, dataset demo, output, report.

### 6.2 Step-Based Development

Lịch sử commit thực tế (`git log`, mới → cũ):

| Commit | Ngày | Message | Nội dung chính (`git show --stat`) |
|---|---|---|---|
| `f0f3897` | 2026-09-20 | add rp step 6 -7 | Thêm `docs/coursework/05_system_build_flow.md` + `06_evaluation.md` |
| `94bdcc5` | 2026-09-20 | add rp step 5 | Thêm `docs/coursework/04_model_implementation.md` |
| `a9d3f25` | 2026-09-20 | add rp step 1-4 | Thêm `docs/coursework/00..03`; xóa các doc gốc ở root; chỉnh `prompts/00` |
| `1018391` | 2026-09-14 | Merge remote 02-features-output tweak | Merge |
| `47678b0` | 2026-09-14 | Steps 12-17: unet/midas evaluation, pipeline fusion, custom demo, report | Thêm prompts 12–17, `evaluation/evaluate_*.py`, `difficulty_analysis.py`, `custom_demo.py`, `docs/15-pipeline-evaluation.md`, `docs/report/coursework_report.md`, ảnh demo |
| `fa79c81` | 2026-09-14 | Add section 02 features and output mapping | Sửa `02-features-output.md` |
| `a9a55ba` | 2026-09-14 | 70% processing | Khối lớn: 66 file, +14250 dòng — models, preprocessing, scene_understanding, visualization, utils, main, configs, metrics (bao trùm Step 02–11) |
| `f5b6667` | 2026-09-14 | 03-solution-tech-ai.md Done | Tài liệu |
| `8fc1f26` | 2026-09-13 | 01-problem-definition.md Done | Tài liệu |
| `d4a21f6` | 2026-09-07 | unet module | `models/unet/`, `preprocessing/cityscapes.py`, `config.py`, `visualization/segmentation.py`, metrics ban đầu |
| `63839ea` | 2026-09-07 | updated | Di chuyển cấu trúc `scene-understanding/` → root |
| `4965a76` | 2026-09-07 | updated architure | Đổi cấu trúc thư mục |
| `17ae3e3`/`a1ec88d`/`f03cf2d` | 2026-08-17 | upadtemd / update / add 1.2md | Tài liệu giai đoạn đầu |

**Nhận xét trung thực về traceability:** không có **một commit cho một step** với mọi step:

- Code bước 02–11 gộp phần lớn trong commit `a9a55ba` "70% processing";
- Code bước 12–17 gộp trong `47678b0`;
- Commit tài liệu coursework (`a9d3f25`, `94bdcc5`, `f0f3897`) tách bạch và gần đây;
- Step 00/01/16 vốn là tài liệu — outputs là Markdown, không phải commit code.

### 6.3 Change Traceability

Độ truy xuất thực tế của dự án:

```text
Prompt → Source change → Test → Commit → Output
```

| Bước | Prompt → Code | Test xanh | Commit riêng? | Output |
|---|---|---|---|---|
| 02–11 | Có (prompt file) | Có (tests hiện hữu) | Không riêng (gộp `a9a55ba`) | Có (modules) |
| 12 | Có | `test_training.py` | Gộp | `unet_cityscapes.pth` + history JSON |
| 13–15 | Có | `test_evaluate_*.py` | Gộp (`47678b0`) | JSON + PNG đánh giá |
| 17 | Có | `test_custom_demo.py` | Gộp | `outputs/custom/` |
| 00/01/16 | Có (prompt là doc) | Không test | Có / đóng góp vào report | Markdown |

Mối liên hệ **Prompt → File → Test → Output** thể hiện rõ qua repository; mối liên hệ **File → Commit riêng theo step** **không** luôn có (code các step bị gộp commit), nên không thể khẳng định mức ảnh hưởng của từng commit tới từng step một cách đầy đủ.

---

## 7. OUTPUT

### 7.1 Step-Level Outputs

Bảng dưới đây dựa trên **bằng chứng repository** — giá trị không xác định được ghi "Not explicitly available in repository".

| Step | Prompt | Main Output (implementation) | Validation |
|---|---|---|---|
| 00 | `prompts/00_project_architecture.md` | Architecture spec (nội dung trong `prompts/01_architecture_specification.md` và `docs/coursework/00`) | Design doc, không code |
| 01 | `prompts/01_architecture_specification.md` | `docs/coursework/01_architecture_specification.md` | Design doc |
| 02 | `prompts/02_config_utils.md` | `utils/*.py` + `configs/*.yaml` | `test_config.py`, `test_seed.py`, `test_logger.py`, `test_device.py` |
| 03 | `prompts/03_dataset.md` | `preprocessing/{cityscapes,kitti,transforms,errors}.py` | `test_cityscapes.py`, `test_kitti.py` |
| 04 | `prompts/04_unet.md` | `models/unet/model.py` | `test_unet.py` (17) |
| 05 | `prompts/05_unet_inference.md` | `models/unet/inference.py` | `test_unet_inference.py` (31) |
| 06 | `prompts/06_segmentation_evaluation.md` | `evaluation/segmentation_metrics.py` | `test_segmentation_metrics.py` (26) |
| 07 | `prompts/07_midas.md` | `models/midas/{model,inference}.py` | `test_midas.py` (34) |
| 08 | `prompts/08_depth_evaluation.md` | `evaluation/depth_metrics.py` | `test_depth_metrics.py` (29) |
| 09 | `prompts/09_fusion_scene_understanding.md` | `scene_understanding/{fusion,analyzer}.py` | `test_fusion.py`, `test_scene_analyzer.py` |
| 10 | `prompts/10_visualization.md` | `visualization/*.py` | `test_visualization.py` (52) |
| 11 | `prompts/11_demo_pipeline.md` | `scene_understanding/pipeline.py` + `main.py` | `test_pipeline.py` (27) |
| 12 | `prompts/12_train_unet.md` | `training/{trainer,train_unet}.py`; `checkpoints/unet_cityscapes.pth` | `test_training.py` (16); history JSON (19 epochs) |
| 13 | `prompts/13_evaluate_unet.md` | `evaluation/evaluate_unet.py` + `outputs/analysis/unet_cityscapes_evaluation.json` | `test_evaluate_unet.py` (17) |
| 14 | `prompts/14_evaluate_midas.md` | `evaluation/evaluate_midas.py` + `outputs/analysis/midas_kitti_evaluation.json` | `test_evaluate_midas.py` (19) |
| 15 | `prompts/15_pipeline_evaluation.md` | `evaluation/evaluate_pipeline.py` + `outputs/analysis/pipeline_evaluation.json` + `docs/15-pipeline-evaluation.md` | `test_evaluate_pipeline.py` (42) |
| 16 | `prompts/16_coursework_report.md` | `docs/report/coursework_report.md` (draft) | Không test (báo cáo) |
| 17 | `prompts/17_custom_image_fusion.md` | `evaluation/custom_demo.py` + demo nhánh `main.py` + `outputs/custom/*` | `test_custom_demo.py` (28) |

Với các bước tài liệu (00/01/16) không có test đi kèm. Việc lưu trữ "implementation JSON tại thời điểm step chạy" riêng theo commit là **not explicitly available in repository**.

### 7.2 Coursework Documentation

Các báo cáo Markdown theo bước hiện có trong `docs/coursework/`:

```text
docs/coursework/00_project_architecture.md        → Step 00
docs/coursework/01_architecture_specification.md  → Step 01
docs/coursework/02_config_utils.md                → Step 02
docs/coursework/03_data_and_preprocessing.md      → Step 03
docs/coursework/04_model_implementation.md        → Step 04
docs/coursework/05_system_build_flow.md           → Step 05
docs/coursework/06_evaluation.md                  → Step 06
docs/coursework/07_ai_assisted_development.md     → Step 07 (tài liệu này)
```

Tài liệu này (07) thuộc nhóm "phản tư về quá trình" — không phải output kỹ thuật có test, khác với các step 02–15.

### 7.3 Final Report Relationship

Các báo cáo theo bước là **nguồn tài liệu** cho báo cáo tổng `docs/report/coursework_report.md`: kiến trúc (00/01), dữ liệu (03), model (04), build (05), evaluation (06), quy trình phát triển (07 — tài liệu này). Báo cáo tổng tổng hợp nội dung từ các bước này thay vì viết lại từ đầu — như `prompts/16_coursework_report.md` mô tả (gộp prompt + report template).

---

## 8. DISCUSSION

### 8.1 Strengths

- **Incremental development** — 18 prompt, mỗi prompt một phạm vi nhỏ + anti-scope, giảm rủi ro làm hỏng module đã có *(được quan sát từ nội dung prompt — interpretation)*;
- **Automated testing** — 457 test offline (456 pass) chạy trong ~23s, không cần dataset/GPU/internet;
- **Reproducibility** — tests synthetic + JSON metadata đầy đủ (checkpoint, split, num_samples, convention) giúp tái lập kết quả;
- **Documentation** — mỗi giai đoạn có báo cáo Markdown riêng trong `docs/coursework/`;
- **Faster iteration** — việc kiểm chứng (pytest + smoke + import test) gắn liền mỗi prompt giúp phát hiện lỗi sớm *(nhận định mang tính đánh giá, không phải con số đo được)*.

### 8.2 Weaknesses

- **AI-generated code vẫn cần review** — không có log kiểm tra riêng cho từng thay đổi của AI trong repository;
- **Giả định sai có thể lan truyền nếu không kiểm tra** — mọi con số trong các doc này phải đối chiếu lại `outputs/*.json` và code;
- **Repository là nguồn sự thật** — khi doc cũ và JSON hiện tại lệch, phải ưu tiên trạng thái hiện tại;
- **Tài liệu sinh ra có thể chứa lệch lạc nếu không validate** — quá trình viết 06/07 đã kiểm chứng từng con số trước khi ghi.

### 8.3 Lessons Learned

- Đừng tin số liệu AI đưa ra — hãy đọc lại JSON và chạy lại test;
- Prompt có anti-scope rõ ràng giúp mỗi bước nhỏ, dễ review;
- Tách commit tài liệu và commit code giúp lịch sử dễ đọc (thực tế repo đã tách các commit report); tuy nhiên để truy xuất tốt hơn, mỗi step nên có commit riêng để `git log` phản ánh đúng thứ tự phát triển *(khuyến nghị dựa trên quan sát git history)*;
- Ghi metadata (checkpoint, split, convention) vào mọi output JSON để sau này validate được.

---

## 9. CONCLUSION

Tóm tắt:

- AI được dùng như **development assistant** — viết code/test/tài liệu theo prompt, có sự review và quyết định của sinh viên;
- Phát triển được chia thành **các bước rõ ràng** (18 prompt từ Step 00–17);
- Mỗi bước dùng một **structured prompt** (objective, scope, anti-scope, testing, validation, output, STOP);
- Implementation luôn được theo sau bởi **testing và validation** (pytest 456 passed / 1 skipped, smoke/import test, đối chiếu JSON);
- **Repository artifacts và Git history** (15 commit) cung cấp khả năng truy xuất theo mức độ verified;
- **Sinh viên chịu trách nhiệm cuối cùng** trong việc review và validate mọi sản phẩm AI tạo ra.

---

## 10. REFERENCES

- `prompts/00_project_architecture.md` … `prompts/17_custom_image_fusion.md` — các task brief (18 prompt).
- `prompts/06_segmentation_evaluation.md` — ví dụ prompt kỹ thuật (structured, anti-scope, 13 test).
- `prompts/16_coursework_report.md` — ví dụ prompt tài liệu (gộp prompt + report template).
- `docs/coursework/00`–`06` — các báo cáo coursework trước, cung cấp ngữ cảnh và thuật ngữ thống nhất.
- `docs/15-pipeline-evaluation.md`, `docs/report/coursework_report.md` — tài liệu kỹ thuật và draft báo cáo tổng.
- `README.md` — cách dùng pipeline, demo custom image, đường dẫn test.
- Git history: `git log`, `git show --stat` cho các commit từ `f03cf2d` (2026-08-17) đến `f0f3897` (2026-09-20).
- `tests/` — 20 file test + `conftest.py`; kết quả `pytest -q` = **456 passed, 1 skipped** (đo 2026-09-20).
- `outputs/analysis/*.json`, `outputs/custom/*`, `outputs/visualization/smoke/*` — các output đã verify.
