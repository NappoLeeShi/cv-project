# Scene Understanding Using Semantic Segmentation and Monocular Depth Estimation

## Coursework Report — Step 09: Conclusion

> Môn học: Computer Vision — Dự án: **CV-PROJECT**
> Tài liệu này **kết luận toàn bộ project** — đề cập ngắn gọn, không lặp lại chi tiết triển khai (docs 01–07) và chi tiết thảo luận (docs 08).
> Mọi con số được đối chiếu với `outputs/analysis/*.json` hiện tại.

---

## 1. CONCLUSION

### 1.1 Project Summary

**Bài toán.** Từ một **ảnh RGB duy nhất**, làm rõ **"vật gì ở đâu, gần hay xa"** trong cảnh giao thông — kết hợp nhận diện ngữ nghĩa (*semantic segmentation*) với ước lượng độ sâu đơn mắt (*monocular depth estimation*).

**Đầu vào.** Một ảnh RGB — **cùng một ảnh** được đưa vào cả hai model (same-image contract).

**Luồng tổng thể A-to-Z:**

```text
RGB image
   U-Net  ─────────────> Segmentation map  (class cho mỗi pixel: road, car, person, ...)
   MiDaS ──────────────> Relative inverse depth map (giá trị lớn = gần hơn)
   Fusion (rule-based) ─> thống kê depth per-class / per-region
      Scene Understanding ─> semantic distribution, depth regions, traffic context,
                            nearest dynamic object, drivable coverage
      Difficulty Analysis ─> difficulty score + level (easy/medium/hard)
      Visualization ───────> ảnh chồng segmentation / depth / fusion
```

Tóm tắt bằng tiếng Việt đơn giản:

- **U-Net** nhận ảnh RGB, trả về **bản đồ phân đoạn** 19 class của Cityscapes (road, car, person, vegetation, ...) — trả lời câu hỏi **"WHAT" (cái gì)**.
- **MiDaS DPT-Large** (pretrained) nhận **cùng ảnh đó**, trả về **relative inverse depth map** — trả lời câu hỏi **"WHERE" (ở đâu, gần hay xa)**. Giá trị lớn hơn nghĩa là gần camera hơn.
- **Fusion** là đoạn **logic quy tắc (rule-based)**, không phải mạng học — ghép bản đồ class với bản đồ depth để biết **mỗi class nằm ở vùng depth nào**, mỗi vùng gần/xa chứa class nào.
- **Scene Understanding** chuyển số liệu thành **mô tả cảnh** cụ thể: thành phần ngữ nghĩa, vùng near/middle/far, object động gần nhất, vùng lái được (drivable).
- **Difficulty Analysis** gán cho mỗi cảnh một **điểm khó** theo công thức heuristic có trọng số, với các mức easy/medium/hard.
- **Evaluation** tách hai mức: **model-level** (U-Net trên Cityscapes, MiDaS trên KITTI) và **pipeline-level** (smoke test 2 ảnh tuỳ chỉnh).

### 1.2 Main Achievements

Các thành quả **được repository hỗ trợ**:

- **U-Net semantic segmentation pipeline** — train, infer, eval chạy được qua CLI (`evaluate_unet`, `infer_unet`, …);
- **MiDaS monocular depth inference** — tích hợp pretrained DPT-Large, không cần tự huấn luyện;
- **Model-level evaluation riêng** — U-Net/Cityscapes 500 ảnh, MiDaS/KITTI 1000 ảnh;
- **Full same-image pipeline** — cùng một ảnh RGB vào cả hai model, chạy end-to-end;
- **Rule-based fusion** — minh bạch, deterministic, output JSON per-class/per-region;
- **Scene analysis** — semantic distribution, depth regions, traffic context, nearest dynamic object, drivable coverage;
- **Difficulty analysis** — score + level theo heuristic cấu hình được;
- **Visualization** — segmentation / depth / fusion / overview (PNG trong `outputs/`);
- **CLI & reproducibility** — config-driven, kết quả tái lập;
- **Automated tests** — 457 test (456 passed, 1 skipped) chạy offline với synthetic fixtures;
- **AI-assisted development documentation** — tài liệu hoá quy trình dùng AI hỗ trợ (docs 07).

Không phóng đại: các thành quả này là **"đã triển khai và chạy được"**, không phải "đạt chất lượng cao" ở mọi khía cạnh.

### 1.3 Model-Level Results

Nguồn: `outputs/analysis/unet_cityscapes_evaluation.json`, `outputs/analysis/midas_kitti_evaluation.json`. Giá trị làm tròn 4 chữ số thập phân như ở Step 06.

**U-Net — Cityscapes val (500 ảnh):**

| Metric | Giá trị | Ý nghĩa |
|---|---|---|
| Pixel Accuracy | **0.9042** | ~90% pixel hợp lệ được phân loại đúng |
| mIoU | **0.4451** | Trung bình cộng IoU của 19 class (mỗi class như nhau) |
| Mean Dice | **0.5498** | Trung bình Dice/F1 của 19 class |

**MiDaS DPT-Large — KITTI val (1000 ảnh, median scaling, depth cap 80 m):**

| Metric | Giá trị |
|---|---|
| RMSE | **4.2561 m** |
| MAE | **3.0182 m** |
| AbsRel | **0.8489** |
| δ1 / δ2 / δ3 | **0.1671 / 0.3275 / 0.4781** |

Giải thích ngắn — kèm bối cảnh bắt buộc:

- Pixel Accuracy cao hơn mIoU vì nó thiên về các class chiếm nhiều pixel (road, vegetation); mIoU trung bình tuyệt đối cả 19 class nên bị kéo xuống bởi các class nhỏ/hiếm (rider 0.0115, motorcycle 0.0422).
- **Không được diễn giải MiDaS như chứng minh độ chính xác depth bằng mét:** output raw của MiDaS là **relative inverse depth**, không có đơn vị mét. Các con số mét ở bảng trên chỉ có giá trị **sau bước median scaling tại thời điểm evaluation** (căn chỉnh thang đo để so sánh với KITTI ground truth) — không phải khả năng metric-depth tuyệt đối của model, và median scaling không phải là huấn luyện.

### 1.4 Full Pipeline Results

Nguồn: `outputs/analysis/pipeline_evaluation.json`.

**Đây là "2-image smoke test"** (2 ảnh tuỳ chỉnh `pipeline_demo_001.png`, `pipeline_demo_002.png`) — **KHÔNG phải benchmark có ý nghĩa thống kê**. Không được dùng nó để khẳng định chất lượng pipeline trên toàn bộ dataset.

| Quan sát | Giá trị |
|---|---|
| Số ảnh demo | **2** |
| Difficulty score | demo_001: **0.541012**; demo_002: **0.508277** |
| Difficulty level | cả hai đều **medium** (0 easy, 2 medium, 0 hard) |
| Average difficulty (aggregate) | 0.524644 |
| Số class hiện diện | 19 (demo_001), 18 (demo_002) |
| Nearest dynamic object | **car** — proximity **near** (cả hai ảnh) |

Quan sát quan trọng từ smoke test:

- Pipeline chạy thành công end-to-end trên cùng ảnh RGB: load model → infer → align output → fusion → scene analysis → difficulty → visualization;
- Cả 2 cảnh demo đều rơi vào mức **medium** theo heuristic — phân phối easy/hard chưa được khám phá vì mẫu quá nhỏ;
- Nearest dynamic object trong cả hai ảnh là **car ở vùng near** — hợp lý cho cảnh giao thông, nhưng chỉ là nhận xét trên 2 mẫu.

### 1.5 Overall Assessment

**Điều hệ thống chứng minh thành công:**

- Tính khả thi của việc kết hợp semantic segmentation (WHAT) với monocular relative depth (WHERE) theo lối **rule-based fusion** trên cùng một ảnh;
- Pipeline tái lập, có CLI, có kiểm thử tự động, tạo ra các đầu ra JSON/PNG cho từng bước;
- Quy trình học – dự đoán – đánh giá được vận hành có kiểm soát và có văn bản.

**Điều kết quả gợi ý:**

- U-Net gán đúng phần lớn pixel ở class phổ biến (Pixel Accuracy 0.9042) nhưng chưa tốt với class nhỏ/hiếm (mIoU 0.4451);
- MiDaS cung cấp thông tin near/far hữu ích cho phân vùng cảnh, nhưng không sinh sẵn độ sâu mét.

**Điều KHÔNG được kết luận từ thí nghiệm hiện tại:**

- Không kết luận được MiDaS cho metric depth chính xác bằng mét;
- Không khẳng định được chất lượng pipeline trên tập lớn — mới có 2 ảnh smoke test;
- Không kết luận được difficulty score là ground-truth độ khó của cảnh;
- Không khẳng định hệ thống kiểm soát phương tiện hay là hệ thống lái tự động (ngoài phạm vi dự án).

Tránh các suy luận nhân quả chưa được hỗ trợ bởi dữ liệu.

---

## 2. PROJECT CONTRIBUTION

### 2.1 Technical Contribution

Dự án chứng minh về mặt kỹ thuật:

- **Kết hợp thông tin ngữ nghĩa và độ sâu tương đối** cho cùng một cảnh — hai tác vụ bổ trợ nhau thay vì chỉ một tác vụ;
- **Same-image inference** — U-Net và MiDaS chạy trên chính một ảnh RGB, cho phép đối sánh trực tiếp pixel-class với pixel-depth;
- **Rule-based fusion** — ghép phân đoạn + depth thành thống kê per-class/per-region mà không cần dataset ghép cặp hay mạng fusion học được;
- **Scene-level interpretation** — biến hai bản đồ nhiều chiều thành mô tả cảnh có cấu trúc (thành phần, vùng gần/xa, object động gần nhất, drivable coverage).

### 2.2 Coursework Contribution

Dự án phủ tốt các mảng của một khóa học Computer Vision:

- **Segmentation** — kiến trúc U-Net, training, inference, đánh giá;
- **Monocular depth estimation** — MiDaS pretrained, bản chất relative inverse depth;
- **Datasets & preprocessing** — Cityscapes (semantic), KITTI (depth), tách biệt rõ hai dataset, đúng chuẩn xử lý dữ liệu;
- **Model evaluation** — metric phân đoạn (Pixel Accuracy, IoU, Dice) và depth (RMSE, MAE, AbsRel, δ1/δ2/δ3), chạy theo kịch bản tái lập;
- **Pipeline integration** — nối hai model + fusion + scene analysis + difficulty;
- **Testing & reproducibility** — 457 test tự động, CLI config-driven;
- **AI-assisted development** — tài liệu hoá quy trình dùng AI hỗ trợ (docs 07).

### 2.3 Practical Meaning

Khả năng liên quan thực tế (nêu thận trọng):

- **Scene analysis** — mô tả ngữ nghĩa + không gian của một ảnh để hiểu cảnh giao thông;
- **Driver assistance research** — thông tin "vật gì ở gần hay xa" có thể hữu ích cho các nghiên cứu hỗ trợ người lái;
- **Understanding spatial context** — kết hợp WHAT + WHERE giúp lý giải ngữ cảnh không gian tốt hơn từng tác vụ đơn lẻ.

Phải nói rõ: **project này không cài đặt điều khiển phương tiện và không phải hệ thống lái tự động hoàn chỉnh.** Đây là module hiểu cảnh ở mức nhận thức (perception/interpretation), minh hoạ nền tảng cho các mở rộng trong tương lai — không phải là sản phẩm đã triển khai.

---

## 3. LIMITATIONS AND FUTURE WORK

### 3.1 Current Limitations

Tổng hợp từ docs 08 (chỉ những giới hạn được hỗ trợ bởi repository):

- **U-Net yếu trên class nhỏ/hiếm** — rider 0.0115, motorcycle 0.0422, truck 0.0760;
- **Training resolution thấp** — `image_size: [256, 512]` trong `configs/unet.yaml`, thấp hơn nhiều so với native 1024×2048 của Cityscapes;
- **Training setup hạn chế** — 20 epochs, batch_size 1, GPU ~4 GB (ghi chú trong config);
- **MiDaS dự đoán relative inverse depth** — không sinh sẵn metric depth bằng mét;
- **Không có metric-depth output từ raw MiDaS** — mọi giá trị mét chỉ sau bước median scaling evaluation-time;
- **Pipeline smoke test chỉ dùng 2 ảnh** — không đủ để đánh giá thống kê;
- **Không có ground-truth cho ảnh tuỳ chỉnh của pipeline** — không thể định lượng chính xác kết quả pipeline;
- **Difficulty score là heuristic** — không phải ground-truth độ khó;
- **Giới hạn tính toán/VRAM** — nêu theo ghi chú trong `configs/unet.yaml` (batch 4 @[512,1024] gây OOM, MiDaS cùng chạy buộc giới hạn kích thước), không bịa thêm thông số phần cứng.

### 3.2 Possible Future Improvements

Những hướng **chỉ là đề xuất tương lai (future work)** — chưa được triển khai:

- Cải thiện **training U-Net** (data augmentation, thêm epochs, learning rate adjustments) nếu phần cứng cho phép;
- Tăng **training resolution** — thử 512×1024 với batch nhỏ trên GPU mạnh hơn;
- Cải thiện phân đoạn **class nhỏ/hiếm** (rider, motorcycle) — loss trọng số, tăng cường mẫu, hai giai đoạn refine;
- Đánh giá thêm **model depth khác** hoặc fine-tune MiDaS trên dữ liệu có metric depth;
- Dùng **tập ảnh tuỳ chỉnh lớn hơn có ground-truth** để đánh giá pipeline định lượng;
- **Calibrate / validate difficulty score** với nhãn/thứ hạng độ khó thực;
- Cải thiện **fusion rules** — điều chỉnh quantile, gộp thêm siêu dữ liệu;
- Bổ sung **scene-level evaluation mạnh hơn** — metric cho nearest object, drivable coverage, phân bố depth.

**KHÔNG khẳng định** các cải tiến này đã được thực hiện — đây là hướng phát triển, không phải tính năng hiện có.

---

## 4. FINAL PRESENTATION SUMMARY

Phần này để sinh viên nói lại toàn bộ project cho giảng viên — dạng hỏi-đáp ngắn gọn:

**1. Bài toán project giải quyết là gì?**
Từ một ảnh RGB duy nhất, xác định "vật gì ở đâu, gần hay xa" trong cảnh giao thông — kết hợp semantic segmentation (WHAT) với monocular relative depth (WHERE).

**2. Vì sao dùng U-Net?**
U-Net là kiến trúc encoder-decoder phổ biến cho semantic segmentation, phù hợp để phân loại từng pixel thành 19 class của Cityscapes, và ta huấn luyện được trên phần cứng hạn chế.

**3. Vì sao dùng MiDaS?**
MiDaS là model monocular depth pretrained (DPT-Large), dự đoán độ sâu tương đối từ một ảnh — không cần stereo/LiDAR và không cần tự huấn luyện.

**4. Vì sao Cityscapes và KITTI tách biệt?**
Đây là hai dataset độc lập, **không paired** (không có cặp segmentation–depth trên cùng một ảnh). Dự án dùng mỗi dataset để đánh giá đúng một task riêng: Cityscapes → U-Net, KITTI → MiDaS. KHÔNG ghép cặp hai dataset thành một.

**5. MiDaS trả ra cái gì?**
MiDaS trả về **relative inverse depth** — không phải mét. Giá trị lớn hơn nghĩa là gần camera hơn. Để so với KITTI ground truth (mét), ta dùng **median scaling** — căn chỉnh thang đo **tại thời điểm evaluation** (không phải huấn luyện).

**6. Hai output được ghép như thế nào?**
Cùng một ảnh RGB được đưa vào cả hai model (same-image). Output segmentation được resize (nearest) và depth được resize (bilinear) về resolution ảnh, rồi pixel-class được đối sánh trực tiếp với pixel-depth để tính thống kê per-class/per-region.

**7. Fusion là gì?**
Fusion là **luật quy tắc (rule-based)**, không phải mạng học: kết hợp class map + depth map thành các đại lượng minh bạch (median depth của từng class, vùng near/middle/far, nearest dynamic object, drivable coverage).

**8. Hệ thống được đánh giá thế nào?**
Hai mức: **model-level** — U-Net trên Cityscapes val (500 ảnh), MiDaS trên KITTI val (1000 ảnh) với các metric chuẩn; và **pipeline-level** — smoke test 2 ảnh tuỳ chỉnh kiểm tra pipeline chạy trọn vẹn end-to-end.

**9. Kết quả chính là gì?**
U-Net: Pixel Accuracy 0.9042, mIoU 0.4451, Mean Dice 0.5498. MiDaS (sau median scaling): RMSE 4.2561 m, MAE 3.0182 m, AbsRel 0.8489, δ1/δ2/δ3 0.1671/0.3275/0.4781. Pipeline smoke test 2 ảnh: cả hai ở mức medium (0.541012 và 0.508277), nearest dynamic object là car gần.

**10. Giới hạn chính là gì?**
U-Net yếu trên class nhỏ/hiếm; MiDaS chỉ cho relative depth (không phải mét native); pipeline mới 2 ảnh smoke test; không có ground-truth cho ảnh pipeline; difficulty là heuristic; và đây không phải hệ thống điều khiển/lái tự động.

---

## 5. REFERENCES

- `docs/coursework/00_project_architecture.md` — kiến trúc tổng thể.
- `docs/coursework/01_architecture_specification.md` — đặc tả kiến trúc, nhiệm vụ các module.
- `docs/coursework/02_config_utils.md` — hệ thống config/utils.
- `docs/coursework/03_data_and_preprocessing.md` — dataset Cityscapes/KITTI và tiền xử lý.
- `docs/coursework/04_model_implementation.md` — U-Net, MiDaS, Fusion.
- `docs/coursework/05_system_build_flow.md` — luồng xây dựng hệ thống.
- `docs/coursework/06_evaluation.md` — đánh giá model-level (U-Net/Cityscapes, MiDaS/KITTI) và pipeline-level.
- `docs/coursework/07_ai_assisted_development.md` — AI-assisted development workflow.
- `docs/coursework/08_discussion.md` — thảo luận kết quả, giới hạn, đánh giá tổng thể.
- `README.md` — tổng quan dự án, cách chạy CLI, kiểm thử.
- `outputs/analysis/unet_cityscapes_evaluation.json` — kết quả U-Net (source of truth).
- `outputs/analysis/midas_kitti_evaluation.json` — kết quả MiDaS (source of truth).
- `outputs/analysis/pipeline_evaluation.json` — kết quả pipeline smoke test (source of truth).
- `configs/unet.yaml`, `configs/midas.yaml`, `configs/pipeline.yaml` — cấu hình training/eval/fusion/difficulty, ghi chú VRAM.
- `evaluation/depth_metrics.py`, `evaluation/segmentation_metrics.py`, `evaluation/difficulty_analysis.py` — công thức metric và difficulty.
- `scene_understanding/fusion.py`, `scene_understanding/analyzer.py` — logic fusion và scene analysis.
- `tests/` — 457 test (456 passed, 1 skipped).
