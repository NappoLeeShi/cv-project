# STEP 11 — DEMO / FULL PIPELINE

## 1. Mục tiêu

Nối toàn bộ các module đã implement thành một pipeline hoàn chỉnh:

Street Image
    ↓
U-Net
    ↓
Semantic Segmentation
    +
MiDaS
    ↓
Relative Inverse Depth
    ↓
Fusion
    ↓
Scene Understanding
    ↓
Visualization
    ↓
Saved Outputs

Step này chỉ tích hợp các module hiện có.

KHÔNG viết lại model.
KHÔNG thay đổi model architecture.
KHÔNG thay đổi metrics.
KHÔNG thay đổi fusion/analyzer logic.
KHÔNG tải dataset.
KHÔNG tự động download pretrained weights.
KHÔNG cần Cityscapes/KITTI để chạy tests.

---

# 2. Đọc code hiện tại

Trước khi implement, đọc:

- models/unet/model.py
- models/unet/inference.py
- models/midas/model.py
- models/midas/inference.py
- scene_understanding/fusion.py
- scene_understanding/analyzer.py
- visualization/segmentation.py
- visualization/depth.py
- visualization/fusion.py
- visualization/scene.py
- visualization/io.py
- configs/unet.yaml
- configs/midas.yaml
- configs/pipeline.yaml
- utils/config.py
- utils/device.py nếu có

Không tạo duplicate implementation.

---

# 3. Pipeline module

Tạo hoặc sử dụng package:

scene_understanding/

Nếu phù hợp với architecture hiện tại, tạo:

scene_understanding/pipeline.py

Pipeline phải là orchestration layer.

Nó không chứa neural-network logic.

---

# 4. Pipeline class

Implement một class tương tự:

SceneUnderstandingPipeline

Constructor nhận:

- segmentation_predictor
- depth_predictor
- optional config

Không hard-code U-Net hoặc MiDaS bên trong.

Pipeline phải dependency-injection friendly.

Ví dụ:

pipeline = SceneUnderstandingPipeline(
    segmentation_predictor=unet_predictor,
    depth_predictor=midas_predictor,
)

---

# 5. Same-image contract

Đây là requirement quan trọng.

Một input image phải được truyền vào CẢ HAI model:

image
 ├──→ U-Net
 └──→ MiDaS

Không được:

- lấy ảnh Cityscapes cho U-Net
- lấy ảnh KITTI cho MiDaS
- ghép hai dataset thành một scene
- pair Cityscapes image với KITTI image

Demo luôn dùng cùng một image object / same image content.

---

# 6. Pipeline execution

Implement method:

run(image)

Pipeline:

1. validate image
2. U-Net prediction
3. MiDaS prediction
4. validate prediction shapes
5. fusion
6. scene analyzer
7. visualization

Kết quả nên là một structured result, ví dụ:

PipelineResult

chứa tối thiểu:

- segmentation
- depth
- fusion
- scene_report

Có thể thêm:

- visualization paths
- metadata

nếu phù hợp.

---

# 7. Shape contract

Segmentation và depth phải có cùng:

[H, W]

trước khi fusion.

Nếu khác resolution:

- không silently resize trong fusion
- pipeline có thể resize theo một policy rõ ràng nếu cần
- nhưng phải đảm bảo prediction cuối cùng cùng resolution

Ưu tiên sử dụng resolution của original input.

Raise ValueError rõ ràng nếu không thể align.

---

# 8. Scene Analyzer integration

Sử dụng:

scene_understanding.analyzer

Không duplicate:

- semantic distribution
- depth distribution
- traffic context
- interpretation
- region analysis

Pipeline chỉ gọi analyzer.

---

# 9. Visualization integration

Pipeline phải có option:

visualize=True / False

Nếu visualize=True:

tạo:

- segmentation visualization
- depth visualization
- fusion visualization
- full visualization
- scene report JSON

Nếu visualize=False:

không tạo file visualization.

---

# 10. Output directory

Mặc định đọc từ:

configs/pipeline.yaml

visualization.output_dir

Ví dụ:

outputs/visualization/

Có thể tạo một subdirectory cho mỗi run nếu phù hợp.

Không overwrite output không cần thiết nếu project convention đã có.

---

# 11. CLI / Demo entry point

Kiểm tra main.py hiện tại trước khi sửa.

Nếu main.py chưa có CLI hoàn chỉnh thì implement CLI.

Mục tiêu:

python main.py --image path/to/image.jpg

Có thể hỗ trợ:

--image
--output-dir
--unet-checkpoint
--midas-weights
--no-visualization
--device

Không bắt buộc phải dùng tất cả nếu config hiện tại đã có cách tương đương.

---

# 12. IMPORTANT: không tự download model

CLI/demo không được tự động download:

- Cityscapes
- KITTI
- U-Net weights
- MiDaS weights

Nếu checkpoint/weights không tồn tại:

raise clear error:

"U-Net checkpoint is required for real inference"

hoặc tương đương.

Không silently fallback sang random weights cho real demo.

---

# 13. Offline testing

Tests không được phụ thuộc vào:

- internet
- dataset
- pretrained weights
- CUDA
- external files

Dùng dummy predictors.

Ví dụ:

DummySegmentationPredictor

predict(image)
→ deterministic synthetic segmentation map

DummyDepthPredictor

predict(image)
→ deterministic synthetic relative inverse depth

Cả hai nhận cùng image.

---

# 14. Test same-image contract

Tạo test chứng minh:

segmentation_predictor.predict()
và
depth_predictor.predict()

nhận cùng image content.

Có thể dùng object identity nếu pipeline contract cho phép.

Requirement:

Không được vô tình preprocess thành hai source image khác nhau ở orchestration layer.

---

# 15. Test pipeline

Tạo:

tests/test_pipeline.py

Test tối thiểu:

1. pipeline construction
2. run with dummy predictors
3. same image passed to both predictors
4. segmentation output shape
5. depth output shape
6. fusion execution
7. scene analyzer execution
8. PipelineResult structure
9. visualization=False không tạo visualization
10. visualization=True tạo expected outputs
11. output directory auto-created
12. invalid image handling
13. prediction shape mismatch
14. predictor failure propagation
15. deterministic result

---

# 16. Test CLI

Không chạy real models.

Test CLI với dummy/mock hoặc test argument parsing riêng.

Kiểm tra:

- --help
- missing image
- invalid image
- output-dir

Không để pytest trigger model download.

---

# 17. Synthetic demo test

Tạo synthetic image:

H = 128
W = 256

Ví dụ:

- sky region
- road region
- car region
- person region

Dummy segmentation phải tạo ít nhất:

sky = 10
road = 0
person = 11
car = 13

Dummy depth:

- sky → relatively small inverse-depth
- road → middle values
- car/person → larger values

Sau pipeline:

scene report phải có traffic context.

Không cần hard-code exact interpretation nếu analyzer hiện tại tự tính.

---

# 18. Real demo support

Real demo phải sử dụng:

UNetInference
+
MidDepthPredictor
+
SceneUnderstandingPipeline

Không tạo model implementation thứ hai.

Nếu U-Net checkpoint chưa có:

clear error.

Nếu MiDaS weights chưa có:

clear error.

Không download tự động.

---

# 19. Dataset independence

Pipeline demo phải hoạt động với:

một arbitrary street image.

Không import:

CityscapesDataset

hoặc

KITTIDataset

vào pipeline demo.

Dataset loaders chỉ phục vụ evaluation.

Architecture:

DATASET EVALUATION

Cityscapes → U-Net → segmentation metrics

KITTI → MiDaS → depth metrics


DEMO

one street image
       ↓
 ┌─────┴─────┐
 U-Net      MiDaS
   ↓          ↓
 Segmentation Depth
       ↓
     Fusion
       ↓
Scene Understanding

Hai phần phải độc lập.

---

# 20. Result serialization

PipelineResult hoặc report phải hỗ trợ:

JSON serialization

Nhưng:

- numpy arrays không cần nhét nguyên map vào JSON
- torch tensors không được serialize trực tiếp
- scene_report phải dùng save_scene_report hiện có

Nếu cần metadata:

```text
{
  "image_size": [H, W],
  "segmentation_shape": [H, W],
  "depth_shape": [H, W],
  "depth_convention": "inverse_relative_larger_closer"
}
```
Không ghi depth units là meters.

# 21. Error handling

Các lỗi phải rõ ràng:

invalid image
predictor missing
predictor output invalid
shape mismatch
invalid output directory
missing checkpoint
missing MiDaS weights

Không dùng bare:

except:

để nuốt lỗi.

# 22. No unnecessary architecture changes

Không tạo:

training/
agent/
neural fusion network
detector/
tracker/
lane_detection/
3D reconstruction/
cloud deployment

Step 11 chỉ là integration.

# 23. Tests regression

Sau khi implement chạy:

pytest

Tất cả tests từ Step 02 → Step 10 phải tiếp tục pass.

Expected hiện tại:

305 passed, 1 skipped

Sau Step 11 số passed sẽ tăng.

CUDA skip vẫn bình thường.

# 24. Manual offline smoke test

Ngoài pytest, chạy một smoke test bằng dummy predictors.

Ví dụ:

synthetic image
→ pipeline
→ segmentation
→ depth
→ fusion
→ scene report
→ visualization

Không cần real weights.

Nếu smoke test tạo được:

outputs/...

thì report lại các file được tạo.

# 25. Deliverables

Có thể tạo:

scene_understanding/pipeline.py

tests/test_pipeline.py

và sửa:

main.py

configs/pipeline.yaml

hoặc các file khác nếu thực sự cần.

Không sửa các module model/metric/fusion nếu không cần thiết.

# 26. Final report

Sau khi hoàn thành report:

Files created
Files modified
Pipeline architecture
Same-image contract
CLI usage
Offline smoke test result
Full pytest result
Number passed
Number skipped
Any remaining issues

STOP sau Step 11.