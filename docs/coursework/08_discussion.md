# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

## Coursework Report — Step 08: Discussion

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Tài liệu này **thảo luận về kết quả** của project — không lặp lại chi tiết triển khai (đã có trong docs 04–06).
> Mọi con số được đối chiếu với `outputs/analysis/*.json` hiện tại. Những nhận định nguyên nhân mà repository không đo trực tiếp được sẽ được ghi rõ là **interpretation**.

---

## 1. REQUIREMENT

### 1.1 Purpose

Phần Discussion có mục đích:

- **Giải thích ý nghĩa của kết quả** — số liệu nói lên điều gì, chứ không chỉ trình bày lại số liệu;
- **Nối kết các module lại với nhau** — U-Net, MiDaS, Fusion, Scene Understanding, Difficulty Analysis cùng đóng góp gì vào cả hệ thống;
- **Chỉ ra giới hạn** — điều gì được phép kết luận và điều gì KHÔNG được kết luận từ thí nghiệm thực tế.

### 1.2 Discussion Questions

Tài liệu trả lời các câu hỏi:

- U-Net làm được tốt điều gì? (mIoU 0.4451, Pixel Accuracy 0.9042 — chi tiết §2.1)
- U-Net gặp khó ở đâu? (class nhỏ/hiếm: rider 0.0115, motorcycle 0.0422)
- Kết quả MiDaS nói lên điều gì? (relative inverse depth, AbsRel 0.8489 sau alignment — §2.2)
- Fusion thêm được gì? (kết hợp WHAT + WHERE — §5)
- Pipeline chứng minh điều gì? (chạy trọn vẹn end-to-end trên 2 ảnh — §2.3)
- Những giới hạn chính là gì? (§9)
- Kết luận nào KHÔNG được thí nghiệm chứng minh? (§10.2)

---

## 2. OVERALL RESULT INTERPRETATION

### 2.1 U-Net

Kết quả trên Cityscapes val (500 ảnh, checkpoint best epoch 20) — nguồn `outputs/analysis/unet_cityscapes_evaluation.json`:

| Metric | Giá trị | Ý nghĩa |
|---|---|---|
| Pixel Accuracy | **0.9042** | ~90% pixel hợp lệ được phân loại đúng |
| Mean IoU (mIoU) | **0.4451** | Trung bình cộng IoU của 19 class |
| Mean Dice | **0.5498** | Trung bình Dice (F1-like) của 19 class |

Class mạnh nhất (per-class IoU):

```text
road 0.9522 · sky 0.8842 · vegetation 0.8630 · car 0.8436 · building 0.8237
```

Class yếu nhất (per-class IoU):

```text
rider 0.0115 · motorcycle 0.0422 · truck 0.0760 · fence 0.1881 · wall 0.1912
```

**Cảnh báo quan trọng — không đồng nhất Pixel Accuracy với mIoU.** Hai con số này đo hai thứ khác nhau:

- **Pixel Accuracy** chỉ chia "số pixel đúng / tổng pixel hợp lệ" — một scene chủ yếu là road + vegetation sẽ cho accuracy cao chỉ vì hai class này chiếm nhiều pixel;
- **mIoU** là **trung bình cộng IoU từng class**, mỗi class (kể cả rider nhỏ) đóng góp như nhau. Khi các class nhỏ/hiếm bị phân đoạn kém, mIoU bị kéo xuống dù Pixel Accuracy vẫn cao.

Vì thế **mIoU 0.4451 phản ánh trung thực hơn** chất lượng tổng thể của U-Net trên 19 class.

### 2.2 MiDaS

Kết quả trên KITTI val (1000 ảnh, median scaling, depth cap 80 m) — nguồn `outputs/analysis/midas_kitti_evaluation.json`:

| Metric | Giá trị |
|---|---|
| RMSE | **4.2561 m** |
| MAE | **3.0182 m** |
| AbsRel | **0.8489** |
| δ1 / δ2 / δ3 | **0.1671 / 0.3275 / 0.4781** |
| Mean scale | 0.2217 |

**Điều cần nhớ rất rõ:** MiDaS raw output là **Relative Inverse Depth**, KHÔNG phải metric depth bằng mét.

- Giá trị lớn hơn → gần camera hơn; giá trị nhỏ hơn → xa hơn;
- Dãy giá trị không có đơn vị mét;
- KITTI ground truth là depth thực tế tính bằng mét (sau khi quy từ millimetre).

Do đó, các con số "mét" (RMSE/MAE) **chỉ có ý nghĩa SAU bước median scaling tại thời điểm evaluation** — đây là căn chỉnh thang đo để so sánh, không phải output native của MiDaS. AbsRel 0.8489 và δ1 0.1671 là kết quả yếu theo chuẩn depth benchmark: đánh giá MiDaS trên toàn bộ ảnh KITTI (không crop trung tâm) + pretrained model không sinh sẵn thang mét khiến việc so sánh số mét kém thuận lợi. *(Phần giải thích nguyên nhân này là interpretation.)*

### 2.3 Full Pipeline

Nguồn `outputs/analysis/pipeline_evaluation.json`:

- Số ảnh: **2** (`pipeline_demo_001.png`, `pipeline_demo_002.png`);
- Difficulty: **2 medium** (score 0.541012 / 0.508277), không easy, không hard;
- Mean segmentation confidence: 0.6613; average depth variation: 9.0421.

**Đây là "2-image smoke test" — KHÔNG phải benchmark có ý nghĩa thống kê.** Không thể dùng 2 ảnh để đánh giá chất lượng pipeline cho toàn bộ dataset.

Điều smoke test chứng minh được (các bước chạy thành công trên cùng một ảnh RGB):

- **Model loading** — U-Net checkpoint + MiDaS DPT-Large weights load và chạy được;
- **Same-image execution** — cùng một ảnh đi qua cả hai model (same-image contract);
- **Output alignment** — segmentation resize nearest, depth resize bilinear về resolution ảnh;
- **Fusion** — phối hợp semantic + relative depth tạo thống kê per-class/region;
- **Scene Analysis** — sinh semantic distribution, depth regions, traffic context, nearest dynamic object;
- **Difficulty Analysis** — tính difficulty score/level theo công thức heuristic;
- **Visualization/output generation** — sinh JSON + PNG (`outputs/analysis/`, `outputs/custom/`).

---

## 3. U-NET DISCUSSION

### 3.1 Strengths

U-Net mạnh ở các class diện tích lớn, phổ biến và có cấu trúc tương đối ổn định trong cảnh đường phố:

- **road** (0.9522) — region trải rộng, ít biến thiên nhãn;
- **sky** (0.8842), **vegetation** (0.8630), **building** (0.8237) — đặc trưng màu/ngữ cảnh khá đồng nhất;
- **car** (0.8436) — object thường xuất hiện với kích thước không quá nhỏ.

Đây là các class **vừa chiếm nhiều pixel vừa đồng đều về hình dạng**, nên dễ học hơn. *(Nhận định "class lớn/common dễ hơn class nhỏ/rare" là interpretation hợp lý dựa trên mẫu per-class IoU, không phải là một thí nghiệm riêng đo nguyên nhân.)*

### 3.2 Weaknesses

Các class yếu: **rider (0.0115), motorcycle (0.0422), truck (0.0760), fence (0.1881), wall (0.1912), traffic light (0.2183), bus (0.1657), train (0.1307)**.

Không chỉ liệt kê — lý do hợp lý cho các class này, được ghi rõ là **interpretation**:

- **Small object size** — rider/motorcycle/traffic light chỉ chiếm vài pixel, dễ bị mất khi downscale;
- **Class imbalance** — các class này xuất hiện hiếm trong Cityscapes so với road/vegetation; config có `class_weights: auto` (`configs/unet.yaml`) cho thấy project đã chủ động bù imbalance nhưng hiệu quả trên class rất hiếm vẫn hạn chế;
- **Low training resolution** — training ở `image_size: [256, 512]` (`configs/unet.yaml`), thấp hơn nhiều so với native 1024×2048, làm mất chi tiết nhỏ;
- **Limited training epochs** — train 20 epochs (`configs/unet.yaml`, `training.epochs: 20`) — số epoch ít, chưa hội tụ triệt để trên class khó.

Các yếu tố này là **cấu hình verify được trong config**, còn **việc gán nguyên nhân** là interpretation.

### 3.3 Practical Meaning

Trong hệ thống hoàn chỉnh, U-Net đóng vai trò **cung cấp "WHAT"** — gán cho mỗi pixel một class ngữ nghĩa. Với các class chính (road, vegetation, sky, car, building) U-Net đủ tin cậy để cung cấp input ổn định cho Fusion và Scene Understanding. Điểm yếu tập trung ở class nhỏ/hiếm — quan trọng cho an toàn giao thông (rider, motorcycle) nhưng lại là nơi U-Net yếu nhất.

---

## 4. MIDAS DISCUSSION

### 4.1 Strengths

- **Monocular depth** — MiDaS dự đoán độ sâu từ **một ảnh RGB**, không cần stereo/LiDAR;
- **Relative ordering** — output là relative inverse depth nên vẫn diễn tả đúng **thứ tự near/far** (giá trị lớn = gần hơn), đủ dùng cho phân vùng region;
- **No training needed** — dùng pretext pretrained DPT-Large (`configs/midas.yaml`, `variant: dpt_large`, source: hub), coursework không phải huấn luyện MiDaS;
- MiDaS được load bằng `_load_weights` và chạy qua `MidDepthPredictor`, đóng gói đúng convention của project.

### 4.2 Weaknesses

Diễn giải kết quả KITTI:

- MiDaS dự đoán **relative inverse depth**;
- KITTI cung cấp **metric depth ground truth** (mét);
- **Median scaling** căn chỉnh thang đo cho evaluation: `scale = median(gt_valid)/median(pred_valid)`, `aligned = pred × scale`;
- Median scaling **là evaluation-time alignment** — KHÔNG train MiDaS, KHÔNG cập nhật weights/bias.

Vì hai thang đo khác nhau, con số "mét" chỉ hợp lệ sau alignment. AbsRel 0.8489 và RMSE 4.2561 m cao chủ yếu vì model pretrained không align sẵn với phân phối KITTI và đánh giá trên toàn bộ ảnh; giá trị này **không nên được trình bày như chất lượng metric-depth tuyệt đối của MiDaS**. *(Giải thích nguyên nhân là interpretation.)*

### 4.3 Practical Meaning

Relative depth vẫn hữu ích:

- **Near/far classification** — chia vùng near/middle/far theo quantile (pipeline) để biết object nào gần xe;
- **Depth variation** — số đo std inverse depth dùng trong difficulty (`average_depth_variation 9.0421`);
- **Kết hợp với class** — gán mỗi class một phân phối depth (vd road, person, vegetation ở giữa cảnh; sky/far ở vùng far) để đọc hiểu cảnh mà không cần thang mét tuyệt đối.

---

## 5. FUSION AND SCENE UNDERSTANDING DISCUSSION

### 5.1 Why Fusion Helps

- **Segmentation alone:** biết WHAT (class gì);
- **Depth alone:** biết để biết gần/xa nhưng không biết đó là gì;
- **Fusion:** kết hợp WHAT + WHERE/NEARNESS.

Dùng output thực tế: `outputs/analysis/pipeline_evaluation.json` (scene) cho thấy Fusion gán mỗi class một bộ thống kê depth (mean/median/min/max), từ đó xác định vd: **person** xuất hiện ở near region (median 19.618), **truck/bus/train** ở far region — thông tin mà segmentation đơn thuần không cung cấp.

### 5.2 Rule-Based Fusion

Ưu điểm (verify được từ `scene_understanding/fusion.py` là rule-based, config `roi_classes`/`drivable_classes`):

- **Interpretable** — luật rõ ràng, sinh viên giải thích được;
- **Easy to debug** — đầu ra JSON per-class/region dễ soi;
- **Deterministic** — cùng input → cùng output, tái lập được;
- **Không cần dataset training thêm** — không học tham số fusion.

Nhược điểm:

- **Phụ thuộc luật viết tay** — chất lượng tùy luật định nghĩa;
- **Not learned from data** — không tối ưu tự động;
- **May not generalize** — luật tốt cho scene này có thể không tốt scene khác;
- **Depth thresholds relative từng ảnh** — vùng near/far tính theo quantile của chính ảnh đó (config `region_quantiles: [0.3333, 0.6667]`), không so sánh tuyệt đối giữa các ảnh.

### 5.3 Scene Understanding

Output thực tế (từ scene report `pipeline_demo_001`):

- **Semantic distribution** — vd road 56076 px, vegetation 98434 px, sky 37310 px, person 49262 px, car 42131 px (^19 class hiện diện);
- **Depth regions** — depth_distribution ~0.333 ở mỗi vùng near/middle/far (proxy tercile);
- **Drivable coverage** — `drivable_coverage_ratio 0.1317` (demo 001), 0.1701 (demo 002);
- **Dynamic object count** — `dynamic_object_count 8` (demo 001), 7 (demo 002);
- **Nearest dynamic class** — car ở near region (cả hai ảnh).

Không bịa giá trị: các số trên đúng từ JSON hiện tại.

---

## 6. DIFFICULTY ANALYSIS DISCUSSION

### 6.1 Difficulty Formula

Công thức thật từ `configs/pipeline.yaml` (`difficulty.factor_weights`) và `evaluation/difficulty_analysis.py`:

```text
difficulty_score = 0.25·ind_seg_uncertainty
                 + 0.20·ind_depth_variation
                 + 0.20·ind_scene_complexity
                 + 0.20·ind_foreground_fraction
                 + 0.15·ind_object_density
```

Ngưỡng (`difficulty.bins`): `easy ≤ 0.4`, `medium ≤ 0.7`, còn lại `hard`.

### 6.2 Interpretation

Đóng góp của từng indicator:

- **segmentation_uncertainty** = `1 − mean_confidence` — model phân đoạn càng lưỡng lự, scene càng khó (weight 0.25, cao nhất);
- **depth_variation** = coeficient of variation (std/mean) của inverse depth — càng nhiều biên depth càng khó;
- **scene_complexity** = số class hiện diện / 19 — càng nhiều class càng phức tạp;
- **foreground_fraction** = tỉ lệ pixel vùng near — càng nhiều tiền cảnh gần càng khó;
- **object_density** = tỉ lệ pixel thuộc class động (person..bicycle) — càng nhiều object càng khó.

Với 2 ảnh đã chạy: demo_001 score 0.541012, demo_002 score 0.508277 — cả hai đều **medium**. Mức độ khó không chênh lệch lớn và nằm khoảng giữa.

### 6.3 Limitations

- Difficulty là **heuristic score** — KHÔNG phải ground-truth difficulty của cảnh;
- Là tổng có trọng số của các indicator được chọn thủ công;
- Chỉ đánh giá trên **2 ảnh** → phân phối easy/medium/hard (0/2/0) **KHÔNG đại diện** cho toàn bộ dataset (không có cơ sở để nói "scene trung bình là medium").

---

## 7. STRENGTHS OF THE PROJECT

Các điểm mạnh **được repository hỗ trợ**:

- **Clear dataset separation** — Cityscapes (semantic) và KITTI (depth) tách biệt, không ghép cặp sai (README + docs 03/06);
- **Same-image pipeline design** — U-Net + MiDaS cùng nhận một ảnh RGB (same-image contract, docs 05/06);
- **Interpretable Fusion** — rule-based, deterministic, output JSON per-class/region;
- **Reproducible CLI** — `main.py --input-dir`, `evaluation.evaluate_*` với `--config`/`--checkpoint`; kết quả đã sinh trong `outputs/`;
- **Automated testing** — 457 test (456 passed, 1 skipped) chạy offline, synthetic fixtures;
- **Model-level + pipeline-level evaluation** — đánh giá U-Net/Cityscapes, MiDaS/KITTI riêng, pipeline behavior riêng (docs 06);
- **Visualization** — segmentation/depth/fusion/overview PNG (outputs/);
- **Explicit limitations** — các docs 06/07 đã nêu rõ giới hạn (smoke test, heuristic, relative depth).

---

## 8. WEAKNESSES OF THE PROJECT

Các điểm yếu **verify được**:

- **U-Net performance on rare classes** — rider 0.0115, motorcycle 0.0422;
- **Limited training resolution** — `[256, 512]` (config) so với native 1024×2048;
- **Limited training epochs** — 20 epochs (config `training.epochs`) — hội tụ chưa triệt để;
- **MiDaS relative-depth nature** — không sinh metric depth native;
- **No metric-depth output** — mọi giá trị "mét" chỉ sau evaluation-time median alignment;
- **Limited pipeline smoke-test size** — 2 ảnh (pipeline_evaluation.json);
- **Heuristic difficulty** — không phải GT;
- **No paired pipeline ground truth** — ảnh pipeline stand-alone không có GT segmentation/depth cho pipeline;
- **Computational/VRAM constraints** — GPU ~4 GB (ghi chú `configs/unet.yaml`), buộc batch_size 1, image_size [256,512] trong training; pipeline giới hạn U-Net inference để MiDaS cùng chạy (docs 05/06).

---

## 9. LIMITATIONS

Phân biệt rõ limitations (giới hạn mang tính cấu trúc) với weaknesses (điểm yếu theo kết quả):

### 9.1 Dataset Limitation

Cityscapes và KITTI là **hai dataset riêng biệt, không paired**: không có cặp segmentation–depth ground truth trên cùng một ảnh. Do đó project **không thực hiện paired semantic-depth evaluation** trên cùng hình.

### 9.2 Model Limitation

U-Net (semantic segmentation, 19 class) và MiDaS (monocular relative depth) là hai task khác nhau với output semantics khác nhau — không thể so sánh trực tiếp chất lượng hai model với nhau.

### 9.3 Pipeline Evaluation Limitation

Pipeline hiện chỉ được đánh giá trên **2 ảnh** (smoke test) — **không đủ** cho statistical benchmarking. Không có kết luận định lượng về chất lượng pipeline ở quy mô lớn.

### 9.4 Difficulty Limitation

Difficulty là **heuristic**, không phải GT; không có GT difficulty để đo độ đúng của score.

### 9.5 Computational Limitation

Bằng chứng từ repository (GHI CHÚ `configs/unet.yaml`): GPU ~4 GB, batch 4 @[512,1024] gây OOM, batch 1 @[256,512] < 1 GiB; `evaluate_pipeline.py` giới hạn U-Net inference theo config nhỏ nhất để chừa VRAM cho MiDaS. Đây là căn cứ nêu hạn chế tài nguyên, **không được bịa thêm thông số phần cứng** ngoài mức ghi trong repo.

---

## 10. WHAT THE PROJECT CAN AND CANNOT CONCLUDE

### 10.1 What Can Be Concluded

- U-Net có hiệu năng semantic segmentation đo được trên Cityscapes val (mIoU 0.4451, Pixel Acc 0.9042 — 500 ảnh);
- MiDaS cung cấp prediction **relative inverse depth** (larger = closer);
- **Median scaling** cho phép evaluation so sánh với KITTI metric depth (evaluation-time only, không train);
- **Fusion** kết hợp semantic + relative depth thành rule-based scene features;
- **Full pipeline** chạy thành công end-to-end trên 2 ảnh thử nghiệm (smoke test).

### 10.2 What Cannot Be Concluded

- **Không** thể khẳng định MiDaS cho metric depth chính xác bằng mét;
- **Không** thể khẳng định smoke test 2 ảnh đại diện hiệu năng pipeline tổng quát;
- **Không** thể khẳng định difficulty score là ground-truth;
- **Không** thể khẳng định Cityscapes và KITTI cung cấp paired GT;
- **Không** thể khẳng định Fusion "cải thiện độ chính xác model" — chưa có thí nghiệm định lượng riêng so sánh có/không có Fusion.

Sự phân biệt này rất quan trọng để tránh overclaim trong báo cáo.

---

## 11. DISCUSSION SUMMARY

| Component | Main Strength | Main Weakness | Main Limitation |
|---|---|---|---|
| U-Net | mIoU 0.4451; mạnh class lớn (road 0.9522, sky 0.8842) | Yếu class nhỏ/hiếm (rider 0.0115, motorcycle 0.0422) | Train ở [256,512], 20 epochs trên GPU ~4 GB |
| MiDaS | Monocular relative depth, pretrained, không cần train | AbsRel 0.8489, δ1 0.1671 sau alignment trên KITTI | Output relative inverse depth, không phải mét; scale chỉ ở eval-time |
| Fusion | Rule-based, interpretable, deterministic | Không học từ dữ liệu, luật viết tay | Threshold relative theo từng ảnh |
| Scene Understanding | Sinh per-class/region stats, traffic context, nearest dynamic object | Phụ thuộc chất lượng 2 model | Không có GT scene để đối chiếu |
| Difficulty Analysis | Công thức minh bạch, trọng số cấu hình được | Heuristic, chỉ số tự chọn | Không phải GT; mới 2 ảnh (0 easy/2 medium/0 hard) |

---

## 12. CONCLUSION

Kết luận trình bày được trước hội đồng:

Dự án chứng minh tính khả thi của việc kết hợp **Semantic Segmentation (U-Net) + Relative Depth (MiDaS)** cho bài toán **rule-based Scene Understanding**: cùng một ảnh RGB, hệ thống phân đoạn cảnh, ước lượng relative depth, fusion, phân tích scene và đánh giá difficulty — chạy được end-to-end, tái lập và có kiểm thử.

Bên cạnh đó, phải thừa nhận các giới hạn lớn: đánh giá pipeline mới ở mức 2-ảnh smoke test; MiDaS chỉ cho relative depth (chưa phải mét); U-Net yếu trên class nhỏ/hiếm; difficulty là heuristic; Cityscapes và KITTI không paired nên không có đánh giá semantic–depth trên cùng ảnh. Kết luận định tính về tính khả thi là vững, còn mọi kết luận định lượng về chất lượng pipeline tuyệt đối đều **chưa được chứng minh**.

---

## 13. REFERENCES

- `docs/coursework/00_project_architecture.md` … `07_ai_assisted_development.md` — ngữ cảnh thiết kế, triển khai, đánh giá trước đó.
- `outputs/analysis/unet_cityscapes_evaluation.json` — kết quả U-Net (source of truth).
- `outputs/analysis/midas_kitti_evaluation.json` — kết quả MiDaS (source of truth).
- `outputs/analysis/pipeline_evaluation.json` — kết quả pipeline smoke test (source of truth).
- `outputs/analysis/pipeline_demo_{001,002}_scene_report.json` — scene analysis per ảnh.
- `configs/unet.yaml`, `configs/midas.yaml`, `configs/pipeline.yaml` — training/eval/difficulty cấu hình, ghi chú VRAM.
- `evaluation/depth_metrics.py`, `evaluation/segmentation_metrics.py`, `evaluation/difficulty_analysis.py` — công thức metric & difficulty.
- `scene_understanding/fusion.py`, `scene_understanding/analyzer.py`, `evaluation/evaluate_pipeline.py` — logic fusion/scene/pipeline.
- `README.md`, `tests/` (457 test: 456 passed, 1 skipped).
