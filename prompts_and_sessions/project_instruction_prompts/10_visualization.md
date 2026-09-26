# STEP 10 — VISUALIZATION

## 1. Mục tiêu

Implement visualization layer cho project:

Segmentation + Depth → Fusion → Scene Understanding

Step này CHỈ chịu trách nhiệm:

- visualize ảnh đầu vào
- visualize semantic segmentation
- visualize relative inverse depth
- visualize fusion giữa segmentation và depth
- visualize scene-understanding result
- lưu visualization ra file
- tạo các helper cần thiết để demo sau này

KHÔNG thay đổi logic của:

- U-Net
- MiDaS
- segmentation metrics
- depth metrics
- fusion logic
- scene analyzer

KHÔNG tải dataset.
KHÔNG tải pretrained weights.
KHÔNG gọi internet.
KHÔNG yêu cầu Cityscapes/KITTI để chạy test.

---

# 2. Kiểm tra code hiện tại trước khi sửa

Đọc các file hiện có:

- models/unet/inference.py
- models/midas/inference.py
- evaluation/segmentation_metrics.py
- evaluation/depth_metrics.py
- scene_understanding/fusion.py
- scene_understanding/analyzer.py
- configs/pipeline.yaml
- configs/unet.yaml
- configs/midas.yaml

Đặc biệt phải giữ đúng các contract hiện tại.

U-Net segmentation output:

[B, H, W]

hoặc single-image:

[H, W]

Semantic prediction là Cityscapes train IDs.

MiDaS depth output:

[H, W]

là relative inverse depth.

Convention hiện tại:

larger value = closer to camera

Không được gọi output này là metric depth hoặc meters.

Fusion/analyzer hiện tại dùng:

larger inverse depth = nearer

và region quantiles:

[0.3333, 0.6666...]

---

# 3. Tạo visualization package

Nếu chưa tồn tại, tạo:

visualization/
├── __init__.py
├── segmentation.py
├── depth.py
├── fusion.py
└── scene.py

Không tạo lại logic model.

---

# 4. Semantic Segmentation Visualization

Implement trong:

visualization/segmentation.py

## Required functions

### colorize_segmentation

Input:

- segmentation map [H,W]
- optional palette
- ignore_index=255

Output:

RGB image dạng numpy array:

[H,W,3], uint8

Mỗi train ID phải được ánh xạ sang một màu cố định.

Sử dụng 19 Cityscapes train classes:

0 road
1 sidewalk
2 building
3 wall
4 fence
5 pole
6 traffic light
7 traffic sign
8 vegetation
9 terrain
10 sky
11 person
12 rider
13 car
14 truck
15 bus
16 train
17 motorcycle
18 bicycle

255 = ignore

Palette phải deterministic.

Không random màu mỗi lần chạy.

---

### create_segmentation_overlay

Input:

- original RGB image
- segmentation map
- alpha

Output:

RGB numpy array.

Overlay segmentation lên ảnh gốc.

Kiểm tra:

- shape khớp
- output uint8
- alpha nằm trong [0,1]

Không làm thay đổi segmentation map gốc.

---

### create_segmentation_legend

Tạo legend chứa:

class name
class train ID
class color

Có thể sử dụng matplotlib.

---

# 5. Depth Visualization

Implement:

visualization/depth.py

## Important

Depth prediction của MiDaS là:

relative inverse depth

larger = closer

Không được sửa raw prediction.

Visualization có thể normalize riêng một bản copy để hiển thị.

Ví dụ:

normalized = (depth - min) / (max - min)

Chỉ dùng normalized version cho visualization.

KHÔNG ghi normalized value trở lại prediction.

---

## Required functions

### normalize_depth_for_visualization

Input:

depth [H,W]

Output:

float32 [H,W] trong khoảng [0,1]

Handle:

- NaN
- inf
- constant depth
- empty valid values

Không mutate input.

---

### colorize_depth

Input:

depth [H,W]

Output:

RGB uint8 image.

Dùng matplotlib colormap.

Phải giữ đúng semantic:

larger inverse depth = closer.

Nếu visualization sử dụng colorbar, label phải thể hiện:

"Relative inverse depth (larger = closer)"

Không ghi:

"Depth in meters"

---

### create_depth_figure

Tạo figure gồm:

- depth visualization
- colorbar
- title

Không cần hiển thị trực tiếp nếu caller muốn save.

---

# 6. Fusion Visualization

Implement:

visualization/fusion.py

Mục tiêu:

hiển thị kết quả kết hợp giữa:

semantic segmentation
+
relative depth

Không tạo neural fusion model.

---

## Required functions

### create_fusion_overlay

Input:

- original RGB image
- segmentation map
- depth map
- alpha_seg
- alpha_depth

Output:

RGB numpy array.

Visualization nên giúp nhìn được:

- semantic regions
- near/far structure

Depth vẫn là relative inverse depth.

Không được biểu diễn depth như mét.

---

### create_region_visualization

Input:

- segmentation map
- depth map
- region map

Region map hiện tại có thể chứa:

far
middle
near

theo inverse depth quantiles.

Tạo visualization dễ đọc.

---

# 7. Scene Understanding Visualization

Implement:

visualization/scene.py

Visualization dựa trên output của:

scene_understanding.analyzer

Không viết lại analyzer.

---

## Required

### create_scene_summary

Nhận scene report dictionary.

Tạo một matplotlib figure/text summary thể hiện các thông tin chính nếu có:

- semantic distribution
- depth distribution
- region summary
- traffic context
- interpretation

Không assume key ngoài contract hiện tại.

Nếu một key không tồn tại thì bỏ qua thay vì crash.

---

### save_scene_report

Input:

scene report dictionary
output path

Save JSON.

Phải đảm bảo:

- JSON serializable
- UTF-8
- indent=2

Không modify report.

---

# 8. Combined Visualization

Tạo helper:

visualization/scene.py

### create_full_visualization

Input:

- original image
- segmentation
- depth
- fusion result / region map nếu cần

Tạo một figure gồm tối thiểu:

1. Original image
2. Segmentation
3. Depth
4. Fusion

Mục đích là có một hình tổng quan cho demo/report.

Title rõ ràng.

Depth title phải nói:

Relative inverse depth

Segmentation title:

Semantic Segmentation

Fusion title:

Segmentation + Relative Depth

---

# 9. Save utilities

Visualization functions không được tự động save nếu caller không yêu cầu.

Tạo helper:

save_visualization(image, path)

Requirements:

- tạo parent directory nếu chưa tồn tại
- hỗ trợ PNG/JPG
- RGB output
- không mutate source

Nếu dùng matplotlib figure thì có helper riêng:

save_figure(fig, path)

---

# 10. Config

Kiểm tra:

configs/pipeline.yaml

Nếu chưa có visualization section thì thêm:

visualization:
  enabled: true
  output_dir: outputs/visualization
  alpha_segmentation: 0.5
  alpha_depth: 0.35

Không xóa các config hiện có.

Nếu project đã có cấu trúc config khác thì tích hợp theo cấu trúc hiện tại thay vì tạo duplicate config.

---

# 11. Tests

Tạo:

tests/test_visualization.py

Tests phải chạy hoàn toàn offline.

Không load U-Net thật.

Không load MiDaS thật.

Không cần Cityscapes.

Không cần KITTI.

Không download gì.

---

## Test segmentation

Test:

1. colorize segmentation shape
2. RGB dtype uint8
3. deterministic palette
4. all 19 classes
5. ignore index
6. overlay shape
7. alpha validation
8. input không bị mutate

---

## Test depth

Test:

1. normalization range [0,1]
2. larger depth remains larger after normalization
3. NaN handling
4. inf handling
5. constant map
6. input không bị mutate
7. colorized output shape/dtype

---

## Test fusion

Test:

1. output shape
2. segmentation information preserved
3. depth visualization works
4. region visualization works
5. input maps không bị mutate

---

## Test scene report

Test:

1. minimal valid report
2. missing optional keys không crash
3. JSON serialization
4. UTF-8
5. report không bị mutate

---

## Test full visualization

Dùng synthetic image:

H=128
W=256

Tạo segmentation map với nhiều class.

Tạo synthetic relative inverse depth:

- background values nhỏ
- foreground values lớn

Kiểm tra full visualization tạo figure thành công.

Không yêu cầu display GUI.

Matplotlib phải chạy được trong headless environment.

Có thể dùng backend phù hợp cho testing.

---

# 12. Important validation

Không được:

- đổi train IDs
- đổi Cityscapes mapping
- đổi MiDaS convention
- đổi fusion logic
- thêm neural network
- thêm training
- thêm dataset download
- thêm cloud service
- thêm dependency không cần thiết

Visualization chỉ đọc output từ các module khác.

---

# 13. API design

Các function phải có type hints.

Ví dụ:

def colorize_segmentation(
    segmentation: np.ndarray,
    ignore_index: int = 255,
) -> np.ndarray:
    ...

Tên function có thể điều chỉnh nếu project hiện tại có convention khác, nhưng phải giữ API rõ ràng và nhất quán.

Public functions nên có docstring ngắn.

---

# 14. Error handling

Các input invalid phải raise ValueError rõ ràng.

Ví dụ:

- segmentation không phải 2D
- depth không phải 2D
- image sai shape
- shape mismatch
- alpha ngoài [0,1]

Không dùng silent failure.

---

# 15. Performance

Không cần tối ưu quá mức.

Nhưng:

- không loop từng pixel nếu vectorization dễ dàng
- không copy dữ liệu không cần thiết
- không mutate model outputs
- visualization normalization chỉ thực hiện trên copy / derived array

---

# 16. Deliverables

Sau khi hoàn thành phải có:

visualization/
├── __init__.py
├── segmentation.py
├── depth.py
├── fusion.py
└── scene.py

tests/
└── test_visualization.py

Nếu cần sửa:

configs/pipeline.yaml

---

# 17. Test command

Sau khi implement:

pytest

Không được bỏ qua các test cũ.

Phải đảm bảo toàn bộ test suite trước Step 10 vẫn pass.

Expected:

- existing tests pass
- new visualization tests pass

CUDA test nếu skip do không có CUDA là bình thường.

---

# 18. Final report

Sau khi code xong, report:

1. Files created
2. Files modified
3. Visualization API
4. Tests added
5. Full pytest result
6. Number passed
7. Number skipped
8. Any remaining issue

KHÔNG tự động chuyển sang Step 11.

STOP sau Step 10.