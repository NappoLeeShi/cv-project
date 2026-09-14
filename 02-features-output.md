# 2. Ánh xạ Đặc trưng → Đầu ra (Features → Output Mapping and Out of scope)

Phần này mô tả cách các đặc trưng hình ảnh ở mức thấp và mức trung gian được sử dụng để tạo ra hai đầu ra chính của hệ thống: Semantic Segmentation và Depth Map.

---

## 2.1 Các đặc trưng hình ảnh được khai thác

| Loại đặc trưng | Thông tin thể hiện | Đầu ra liên quan |
|----------|----------|----------|
| **Màu sắc / Cường độ sáng** | Đường thường có màu xám đậm, bầu trời màu xanh, cây cối màu xanh lá | Semantic Segmentation |
| **Texture (Kết cấu bề mặt)** | Nhựa đường, cỏ và kim loại có kết cấu khác nhau | Semantic Segmentation |
| **Biên / Gradient** | Ranh giới đối tượng, đường chân trời | Segmentation và Depth |
| **Phối cảnh / Điểm hội tụ** | Các đường song song hội tụ tạo tín hiệu về khoảng cách | Depth Estimation |
| **Kích thước tương đối** | Các vật thể cùng loại sẽ nhỏ hơn khi ở xa | Depth Estimation |
| **Quan hệ che khuất** | Vật gần thường che khuất vật xa | Depth Estimation |
| **Bóng đổ và độ sáng tối** | Hướng ánh sáng giúp suy ra hình dạng bề mặt | Depth Estimation |
| **Kiến thức ngữ nghĩa (Semantic Priors)** | Bầu trời thường ở phía trên, mặt đường thường ở phía dưới | Cả hai nhiệm vụ |

---

## 2.2 Đầu ra 1 – Semantic Segmentation

### Từ đặc trưng ảnh đến nhãn lớp của từng pixel

```text
Đặc trưng ảnh
(màu sắc, texture, ngữ cảnh cục bộ)
        │
        ▼
     Encoder
        │
        ▼
 Bản đồ đặc trưng mức cao
 (Semantic Feature Map)
        │
        ▼
     Decoder
        │
        ▼
  Per-pixel Logits
 (N kênh tương ứng N lớp)
        │
        ▼
      Argmax
        │
        ▼
 Nhãn lớp của từng pixel
 (Road, Car, Sky, ...)
```

### Các lựa chọn thiết kế quan trọng

- **Multi-scale Context (Ngữ cảnh đa tỷ lệ)**  
  Một pixel không thể được phân loại chính xác nếu chỉ nhìn riêng lẻ. Mô hình cần quan sát cả vùng lân cận và bố cục toàn cảnh.

- **Skip Connections**  
  Truyền thông tin chi tiết từ Encoder sang Decoder nhằm bảo toàn biên và các chi tiết nhỏ bị mất trong quá trình giảm kích thước ảnh.

- **CRF / Refinement (Tùy chọn)**  
  Một số hệ thống sử dụng bước hậu xử lý để làm sắc nét ranh giới giữa các lớp.

### Các lớp mục tiêu (Cityscapes – 19 lớp)

text
road
sidewalk
building
wall
fence
pole
traffic light
traffic sign
vegetation
terrain
sky
person
rider
car
truck
bus
train
motorcycle
bicycle
```

---

## 2.3 Đầu ra 2 – Depth Map

### Từ đặc trưng ảnh đến giá trị độ sâu

```text
Đặc trưng ảnh
(màu sắc, texture, phối cảnh)
        │
        ▼
     Encoder
        │
        ▼
 Biểu diễn đặc trưng
        │
        ├──► Depth Head
        │        │
        │        ▼
        │   Depth Map
        │ (H × W × 1)
        │
        └──► Confidence Map
              (Tùy chọn)
```

### Các lựa chọn thiết kế quan trọng

- **Scale-Invariant Loss**  
  Mô hình tập trung học quan hệ gần – xa giữa các đối tượng thay vì phụ thuộc hoàn toàn vào khoảng cách tuyệt đối.

- **Multi-scale Supervision**  
  Hàm mất mát được tính ở nhiều mức độ phân giải khác nhau giúp quá trình huấn luyện ổn định hơn.

- **Edge-aware Refinement**  
  Giữ cho các biên độ sâu sắc nét tại ranh giới đối tượng.

---

## 2.4 Mối quan hệ giữa Semantic Segmentation và Depth Estimation

Mặc dù Semantic Segmentation và Depth Estimation được thực hiện bằng hai mô hình riêng biệt (U-Net và MiDaS), cả hai đều khai thác các đặc trưng thị giác tương tự từ cùng một ảnh đầu vào.

```text
Input Image
     │
 ┌───┴───┐
 │       │
 ▼       ▼
U-Net   MiDaS
 │       │
 ▼       ▼
Seg.    Depth
Mask    Map
```

### Thông tin bổ sung cho nhau

- **Semantic Segmentation** trả lời câu hỏi:

```text
Trong ảnh có những đối tượng nào?
```

- **Depth Estimation** trả lời câu hỏi:

```text
Các đối tượng đó cách camera bao xa?
```

Khi kết hợp hai kết quả, hệ thống có thể hiểu ngữ cảnh giao thông đầy đủ hơn.

Ví dụ:

| Đối tượng | Kết quả Segmentation | Kết quả Depth |
|----------|----------|----------|
| Xe hơi | Car | 12 m |
| Người đi bộ | Person | 6 m |
| Đường | Road | Khu vực có thể di chuyển |
| Tòa nhà | Building | 40 m |

### Lợi ích khi kết hợp hai nhiệm vụ

- Nâng cao khả năng hiểu ngữ cảnh giao thông.
- Cung cấp đồng thời thông tin ngữ nghĩa và hình học.
- Hỗ trợ các ứng dụng như xe tự lái, robot di chuyển và giao thông thông minh.

Mặc dù U-Net và MiDaS được huấn luyện độc lập, kết quả của chúng có thể được kết hợp để tạo ra thông tin phong phú hơn so với từng nhiệm vụ riêng lẻ.

---

## 2.5 Tóm tắt mối quan hệ giữa đặc trưng và đầu ra

| Đặc trưng mức thấp | Đặc trưng mức trung gian | Đặc trưng mức cao | Segmentation | Depth |
|----------|----------|----------|----------|----------|
| Màu sắc RGB | Biên, góc | Bộ phận đối tượng | ✓ | ✓ |
| Gradient kết cấu | Hướng bề mặt | Bố cục cảnh | ✓ | ✓ |
| — | Điểm hội tụ | Hình học toàn cảnh | — | ✓ |
| — | Ranh giới đối tượng | Phạm vi vật thể | ✓ | ✓ |

---

## 2.6 Phạm vi không thực hiện (Out of Scope)

Phần phân tích đặc trưng trong đề tài chỉ tập trung vào các tín hiệu thị giác được sử dụng cho Semantic Segmentation và Monocular Depth Estimation.

Các nội dung sau không nằm trong phạm vi nghiên cứu:

- Khai thác thông tin theo thời gian từ video như chuyển động đối tượng, Optical Flow hoặc Tracking.
- Hình học đa góc nhìn (Multi-view Geometry) từ nhiều camera hoặc Stereo Vision.
- Dữ liệu từ LiDAR, Radar hoặc các phương pháp Sensor Fusion.
- Phân biệt từng cá thể riêng biệt của cùng một lớp đối tượng (Instance Segmentation).
- Ước lượng tư thế 3D của vật thể hoặc tái tạo toàn bộ cảnh 3D.
- Đánh giá khả năng hoạt động trong các điều kiện thời tiết cực đoan như mưa lớn, sương mù, tuyết hoặc ban đêm.
- Phân tích khả năng giải thích (Explainability) của các đặc trưng được học bởi mạng nơ-ron sâu.

Do đó, đề tài chỉ sử dụng các đặc trưng được trích xuất từ một ảnh RGB đường phố duy nhất để thực hiện Semantic Segmentation và Monocular Depth Estimation.