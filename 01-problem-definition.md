# 1. Định Nghĩa Bài Toán (Problem Definition)

## Tên đề tài

**Phân tích ngữ cảnh giao thông dựa trên Semantic Segmentation và Depth Estimation**
*(Traffic Scene Understanding via Semantic Segmentation & Monocular Depth Estimation)*

---

## 1.1 Bối cảnh và Động lực (Context & Motivation)

Trong bối cảnh đô thị hóa và giao thông ngày càng phức tạp, các hệ thống thị giác máy tính đóng vai trò cốt lõi trong việc hỗ trợ xe tự lái, robot di chuyển và các hệ thống giám sát giao thông thông minh. Để một máy tính có thể "hiểu" cảnh vật đường phố, nó cần giải quyết đồng thời hai câu hỏi cơ bản:

> **"Trong ảnh có những gì?"** → *Semantic Segmentation*
>
> **"Chúng cách camera bao xa?"** → *Depth Estimation*

Việc chỉ nhận diện đối tượng (object detection) là chưa đủ — hệ thống cần biết **lớp ngữ nghĩa** của từng điểm ảnh **và** khoảng cách tương ứng để đưa ra quyết định an toàn trong không gian ba chiều.

Đề tài này xây dựng một pipeline kết hợp hai nhiệm vụ trên từ một **ảnh RGB đường phố đơn lẻ (monocular)**, không yêu cầu thiết bị cảm biến chuyên dụng như LiDAR hay camera stereo — giúp hạ thấp chi phí triển khai và mở rộng phạm vi ứng dụng thực tế.

---

## 1.2 Phát biểu bài toán (Problem Statement)

### Đầu vào (Input)

| Thuộc tính | Mô tả |
|---|---|
| Loại dữ liệu | Ảnh màu RGB đường phố (street scene) |
| Kích thước | H × W × 3 (ví dụ: 1024 × 2048 × 3 với Cityscapes) |
| Nguồn thu | Camera đơn gắn trên xe hoặc cơ sở hạ tầng đường bộ |
| Điều kiện | Ban ngày, thời tiết bình thường (phạm vi đề tài) |

### Đầu ra (Output)

| Đầu ra | Ký hiệu | Kích thước | Ý nghĩa |
|---|---|---|---|
| **Semantic Segmentation Mask** | Ŝ | H × W | Nhãn lớp ngữ nghĩa cho từng pixel (road, car, sky, person, ...) |
| **Depth Map** | D̂ | H × W | Giá trị độ sâu tương đối / tuyệt đối (mét) cho từng pixel |

### Luồng xử lý tổng quan

```text
┌──────────────────────────────────┐
│   ẢNH ĐƯỜNG PHỐ RGB (H × W × 3) │
└─────────────────┬────────────────┘
                  │
       ┌──────────┴──────────┐
       │                     │
       ▼                     ▼
 ┌──────────────┐     ┌──────────────┐
 │    U-Net     │     │    MiDaS     │
 │ (Phân đoạn  │     │ (Ước lượng  │
 │  ngữ nghĩa) │     │   độ sâu)   │
 └──────┬───────┘     └──────┬───────┘
        │                    │
        ▼                    ▼
 ┌──────────────┐     ┌──────────────┐
 │ Segmentation │     │  Depth Map   │
 │    Mask Ŝ   │     │      D̂      │
 │ (H × W,     │     │ (H × W,     │
 │  19 nhãn)   │     │  1 kênh)    │
 └──────┬───────┘     └──────┬───────┘
        │                    │
        └──────────┬──────────┘
                   ▼
      ┌────────────────────────┐
      │    SCENE UNDERSTANDING │
      │  (Hiểu ngữ cảnh giao  │
      │   thông toàn diện)    │
      └────────────────────────┘
```

---

## 1.3 Định nghĩa các nhiệm vụ con (Sub-task Definitions)

### 1.3.1 Semantic Segmentation

**Định nghĩa toán học:**

Cho ảnh đầu vào `I` kích thước (H × W × 3), mô hình học ánh xạ:

```
f_seg : I  →  Ŝ ∈ {1, 2, ..., C}^(H × W)
```

trong đó `C` là số lớp ngữ nghĩa (C = 19 với Cityscapes), và `Ŝ(i,j)` là nhãn lớp được dự đoán cho pixel tại vị trí `(i, j)`.

**Đặc điểm của bài toán:**
- Là bài toán **phân loại dày đặc** (dense classification) — mỗi pixel đều cần được gán nhãn.
- Yêu cầu mô hình hiểu đồng thời ngữ cảnh cục bộ (local texture) và toàn cảnh (global layout).
- Khó khăn đặc trưng: mất cân bằng lớp (class imbalance) — diện tích *road* và *sky* lớn hơn nhiều so với *pole* hay *bicycle*.

**Các lớp mục tiêu (Cityscapes – 19 lớp):**

| Nhóm | Các lớp |
|---|---|
| Bề mặt phẳng | road, sidewalk |
| Công trình | building, wall, fence |
| Vật thể cố định | pole, traffic light, traffic sign |
| Thiên nhiên | vegetation, terrain |
| Bầu trời | sky |
| Con người | person, rider |
| Phương tiện | car, truck, bus, train, motorcycle, bicycle |

---

### 1.3.2 Monocular Depth Estimation

**Định nghĩa toán học:**

Cho ảnh đầu vào `I` kích thước (H × W × 3), mô hình học ánh xạ:

```
f_depth : I  →  D̂ ∈ R+^(H × W)
```

trong đó `D̂(i,j)` là ước lượng khoảng cách (mét) từ camera đến điểm bề mặt tương ứng với pixel `(i, j)`.

**Đặc điểm của bài toán:**
- Là bài toán **ill-posed**: từ ảnh 2D duy nhất, có vô số cấu hình 3D có thể sinh ra cùng một ảnh.
- Mô hình phải học các **visual cue** (tín hiệu thị giác ngầm định):
  - *Phối cảnh tuyến tính*: đường song song hội tụ ở xa.
  - *Kích thước tương đối*: đối tượng cùng loại nhỏ hơn khi ở xa.
  - *Che khuất*: vật ở gần che khuất vật ở xa.
  - *Gradient kết cấu*: texture mịn hơn ở khoảng cách xa.
  - *Bóng và ánh sáng*: gợi ý hình dạng bề mặt.
- Kết quả có thể là **độ sâu tuyệt đối** (absolute depth, mét) hoặc **độ sâu tương đối** (relative/affine-invariant depth).

---

## 1.4 Dataset

### 1.4.1 Cityscapes — Dành cho Semantic Segmentation

| Thuộc tính | Thông tin |
|---|---|
| Nguồn | cityscapes-dataset.com |
| Cảnh thu thập | 50 thành phố châu Âu |
| Số ảnh (fine annotation) | 5,000 ảnh (2,975 train / 500 val / 1,525 test) |
| Số ảnh (coarse annotation) | 20,000 ảnh (bổ sung) |
| Độ phân giải | 2048 × 1024 pixels |
| Số lớp | 30 lớp tổng, **19 lớp** dùng cho training và evaluation |
| Định dạng nhãn | Pixel-wise PNG mask (labelId / trainId) |
| Đặc điểm nổi bật | Annotation thủ công chất lượng cao, đa dạng thời tiết và thời điểm |

**Phân bố một số lớp điển hình:**

```
road        ████████████████████ ~40% diện tích
sky         ██████████          ~20% diện tích
building    ████████            ~16% diện tích
vegetation  ██████              ~12% diện tích
car         ████                ~8%  diện tích
person      ██                  ~2%  diện tích
bicycle     █                   ~0.5% diện tích
```

---

### 1.4.2 KITTI — Dành cho Depth Estimation

| Thuộc tính | Thông tin |
|---|---|
| Nguồn | cvlibs.net/datasets/kitti |
| Cảnh thu thập | Karlsruhe, Đức (đường phố, cao tốc, khu dân cư) |
| Thiết bị thu thập | Camera stereo + LiDAR Velodyne HDL-64E |
| Số ảnh (depth split chuẩn Eigen) | ~26,000 train / ~697 test |
| Độ phân giải ảnh | 375 × 1242 pixels (xấp xỉ) |
| Ground truth depth | Sparse LiDAR points (~5% pixels có GT) |
| Phạm vi độ sâu | 0 – 80 mét |
| Đặc điểm nổi bật | Ground truth từ LiDAR thực, chuẩn benchmark phổ biến nhất |

**Lưu ý kỹ thuật:** Ground truth depth của KITTI là **sparse** (thưa thớt) — chỉ khoảng 5% pixel có giá trị độ sâu thực. Phần còn lại được suy ra qua kỹ thuật *depth completion* hoặc được bỏ qua trong hàm mất mát.

---

## 1.5 Mô hình đề xuất (Proposed Models)

### 1.5.1 U-Net — Semantic Segmentation

**Kiến trúc:**

```text
Input (H×W×3)
      │
  ┌───┴───┐
  │Encoder│  ← Conv blocks + MaxPool (giảm H, W; tăng channels)
  │ (×4)  │
  └───┬───┘
      │
  Bottleneck
      │
  ┌───┴───┐
  │Decoder│  ← Transposed Conv + Skip Connections từ Encoder
  │ (×4)  │
  └───┬───┘
      │
  1×1 Conv → Softmax
      │
  Seg Mask (H×W×C)
```

**Ưu điểm:**
- **Skip connections** bảo toàn thông tin không gian chi tiết (biên, cạnh đối tượng nhỏ).
- Kiến trúc Encoder–Decoder đối xứng — đơn giản, dễ huấn luyện.
- Phù hợp với bài toán dense prediction có yêu cầu giữ độ phân giải cao.
- Backbone có thể thay thế bằng ResNet, EfficientNet để tăng hiệu năng.

---

### 1.5.2 MiDaS — Monocular Depth Estimation

**Đặc điểm nổi bật:**
- Huấn luyện trên **10+ dataset** đa dạng (NYU, KITTI, MegaDepth, ReDWeb, ...) → khả năng **zero-shot generalization** mạnh.
- Dự đoán **relative (affine-invariant) depth**: kết quả có tỉ lệ và độ lệch tùy ý, cần căn chỉnh khi so sánh với metric tuyệt đối.
- Phiên bản mới nhất (MiDaS v3.1 với backbone DPT-Large) kết hợp Vision Transformer và Dense Prediction Transformer (DPT).

**Kiến trúc DPT:**

```text
Input (H×W×3)
      │
  ViT Encoder
  (patch embeddings → transformer blocks)
      │
  Feature Reassembly
  (lấy features từ nhiều layer của ViT)
      │
  Fusion Decoder (kết hợp đa tỉ lệ)
      │
  Depth Head
      │
  Depth Map (H×W×1)
```

---

## 1.6 Tiêu chí đánh giá (Evaluation Metrics)

### 1.6.1 Semantic Segmentation

| Metric | Ý nghĩa | Công thức |
|---|---|---|
| **mIoU** | Chỉ số chính, trung bình IoU qua tất cả lớp | TP / (TP + FP + FN), trung bình trên C lớp |
| **Pixel Accuracy** | Tỉ lệ pixel được gán đúng nhãn | Σ TP / tổng số pixel |
| **Class Accuracy** | Xử lý mất cân bằng lớp | Trung bình Recall từng lớp |

**Ngưỡng tham khảo trên Cityscapes val set:**
- State-of-the-art (Mask2Former): mIoU ≈ 84%
- DeepLabv3+: mIoU ≈ 78–80%
- **Mục tiêu đề tài: mIoU ≥ 70%**

---

### 1.6.2 Depth Estimation

| Metric | Ý nghĩa | Chiều tốt |
|---|---|---|
| **AbsRel** | Lỗi tương đối tuyệt đối — chỉ số chính | Càng nhỏ càng tốt |
| **SqRel** | Lỗi bình phương tương đối | Càng nhỏ càng tốt |
| **RMSE** | Lỗi trung bình bình phương | Càng nhỏ càng tốt |
| **δ < 1.25** | % pixel trong ngưỡng sai số chấp nhận được | Càng lớn càng tốt |

**Ngưỡng tham khảo trên KITTI Eigen test split:**
- MiDaS v3.1: AbsRel ≈ 0.059, δ<1.25 ≈ 0.964
- **Mục tiêu đề tài: AbsRel ≤ 0.10, δ<1.25 ≥ 0.88**

---

## 1.7 Thách thức kỹ thuật (Technical Challenges)

| Thách thức | Mô tả | Hướng giải quyết |
|---|---|---|
| **Mất cân bằng lớp** | *Road* và *sky* chiếm phần lớn diện tích, *bicycle* và *pole* rất ít | Weighted Cross-Entropy, Dice Loss |
| **Mất thông tin không gian** | Encoder làm giảm độ phân giải không gian | Skip connections trong U-Net |
| **Depth ambiguity** | Ảnh 2D không xác định duy nhất không gian 3D | Học visual cue, huấn luyện trên dữ liệu đa dạng |
| **Ground truth thưa (KITTI)** | Chỉ ~5% pixel có GT depth | Scale-Invariant Loss, mask invalid pixels |
| **Domain gap** | Cityscapes (châu Âu) ≠ ảnh đường phố tùy ý | Data augmentation, fine-tuning |
| **Căn chỉnh tỉ lệ depth** | MiDaS dự đoán relative depth, không có đơn vị mét | Least-squares alignment với GT |
| **Tốc độ suy diễn** | Mô hình nặng → khó triển khai real-time | Model quantization, ONNX export |

---

## 1.8 Ứng dụng thực tế (Applications)

### Xe tự lái (Autonomous Driving)
- Xác định **vùng có thể di chuyển** (drivable area) từ nhãn *road*.
- Phát hiện và định vị các đối tượng nguy hiểm (người đi bộ, xe ngược chiều) kết hợp với khoảng cách.
- Hỗ trợ hệ thống **ADAS** (Advanced Driver Assistance Systems): cảnh báo va chạm, giữ làn.

### Robot di chuyển (Mobile Robotics)
- Xây dựng bản đồ ngữ nghĩa (semantic map) của môi trường.
- Lập kế hoạch đường đi tránh vật cản dựa trên depth map.
- Nhận biết không gian trống để di chuyển an toàn.

### Giám sát giao thông thông minh (Smart Traffic Monitoring)
- Ước lượng mật độ phương tiện theo không gian.
- Phát hiện vi phạm giao thông (đi vào vùng cấm, leo vỉa hè).
- Thống kê phương tiện theo loại (xe máy, ô tô, xe buýt, ...).

### Thực tế tăng cường (Augmented Reality)
- Chèn nội dung ảo phù hợp với hình học cảnh thực.
- Hiển thị thông tin chỉ đường và cảnh báo theo ngữ cảnh không gian.

---

## 1.9 Phạm vi nghiên cứu (Scope)

### Trong phạm vi (In Scope)

| # | Nội dung |
|---|---|
| 1 | Semantic segmentation từ ảnh RGB đơn lẻ |
| 2 | Monocular depth estimation từ ảnh RGB đơn lẻ |
| 3 | Kết hợp hai đầu ra để hiểu ngữ cảnh giao thông |
| 4 | Huấn luyện / fine-tuning trên Cityscapes và KITTI |
| 5 | Đánh giá định lượng theo các metric tiêu chuẩn |
| 6 | Trực quan hóa kết quả (overlay, colormap) |

### Ngoài phạm vi (Out of Scope)

| # | Nội dung | Lý do |
|---|---|---|
| 1 | Instance segmentation | Độ phức tạp cao, nằm ngoài yêu cầu đề tài |
| 2 | Xử lý video / temporal modeling | Chỉ xử lý ảnh tĩnh |
| 3 | Stereo depth hoặc LiDAR fusion | Không có thiết bị đa cảm biến |
| 4 | Tracking đối tượng theo thời gian | Không xử lý chuỗi frame |
| 5 | Ước lượng tư thế 3D (pose estimation) | Nằm ngoài phạm vi |
| 6 | Tái tạo cảnh 3D (3D reconstruction) | Nằm ngoài phạm vi |
| 7 | Hoạt động trong điều kiện đêm / thời tiết cực đoan | Giới hạn dataset huấn luyện |
| 8 | Triển khai nhúng (embedded deployment) | Tập trung vào nghiên cứu |

---

## 1.10 Tóm tắt (Summary)

| Hạng mục | Chi tiết |
|---|---|
| **Input** | Ảnh RGB đường phố đơn lẻ (H × W × 3) |
| **Output 1** | Semantic Segmentation Mask — nhãn lớp cho từng pixel (19 lớp Cityscapes) |
| **Output 2** | Depth Map — khoảng cách ước lượng cho từng pixel (mét) |
| **Dataset** | Cityscapes (segmentation) + KITTI (depth) |
| **Mô hình** | U-Net (segmentation) + MiDaS v3.1 (depth) |
| **Metric chính** | mIoU (segmentation) + AbsRel / δ<1.25 (depth) |
| **Ứng dụng** | Xe tự lái, robot, giám sát giao thông thông minh |
| **Phương pháp** | Deep learning — dense prediction, monocular vision |