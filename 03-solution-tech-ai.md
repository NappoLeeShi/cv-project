## 3.1. Kiến trúc tổng thể (Overall Architecture)

Hệ thống nhận đầu vào là một ảnh RGB đường phố và thực hiện hai nhiệm vụ chính:

- Semantic Segmentation: phân chia ảnh thành các vùng đối tượng như đường, xe, người, bầu trời,...
- Monocular Depth Estimation: ước lượng khoảng cách tương đối từ camera đến các vật thể trong ảnh.

### Pipeline

```text
Ảnh đường phố RGB
        │
        ├───────────────┐
        │               │
        ▼               ▼
      U-Net           MiDaS
        │               │
        ▼               ▼
Segmentation Map    Depth Map
        │               │
        └───────┬───────┘
                ▼
        Scene Understanding
```

Trong đó:

- U-Net được sử dụng cho bài toán Semantic Segmentation.
- MiDaS được sử dụng cho bài toán Monocular Depth Estimation.
- Hai kết quả được kết hợp để giúp hệ thống hiểu được ngữ cảnh và khoảng cách tương đối của các đối tượng trong cảnh giao thông.

## 3.2. Các mô hình AI được sử dụng (AI Models Used)

### 3.2.1. Phân đoạn ngữ nghĩa – U-Net (Semantic Segmentation – U-Net)

U-Net là kiến trúc mạng CNN phổ biến cho bài toán Semantic Segmentation. Mô hình gồm hai phần chính:

- Encoder: trích xuất các đặc trưng từ ảnh đầu vào.
- Decoder: khôi phục kích thước ảnh và tạo ra bản đồ phân vùng.

U-Net sử dụng Skip Connection để truyền các đặc trưng từ Encoder sang Decoder, giúp giữ lại thông tin chi tiết về vị trí và hình dạng của đối tượng.

Trong dự án, U-Net được sử dụng để phân vùng các lớp trong ảnh giao thông, ví dụ:

- Road
- Car
- Person
- Sky
- Building
- Vegetation

Kết quả đầu ra là một Segmentation Map, trong đó mỗi pixel được gán vào một lớp tương ứng.

### 3.2.2. Ước lượng độ sâu đơn ảnh – MiDaS (Monocular Depth Estimation – MiDaS)

MiDaS là mô hình Deep Learning dùng để ước lượng độ sâu từ một ảnh đơn.

Mô hình nhận một ảnh RGB và tạo ra Depth Map thể hiện độ sâu tương đối của các vùng trong ảnh.

- Vùng gần camera thường có giá trị depth khác với vùng ở xa.
- Depth Map giúp hệ thống nhận biết tương quan khoảng cách giữa các vật thể.
- MiDaS có thể sử dụng pretrained weights nên phù hợp với bài toán thử nghiệm trên ảnh đường phố.

## 3.3. Công nghệ sử dụng (Technology Stack)

Hệ thống sử dụng các công nghệ chính:

- Python: ngôn ngữ lập trình chính.
- PyTorch: xây dựng và thực hiện mô hình Deep Learning.
- Torchvision: hỗ trợ xử lý ảnh và các thành phần liên quan đến Computer Vision.
- NumPy: xử lý dữ liệu dạng mảng.
- OpenCV/PIL: đọc và tiền xử lý ảnh.
- Matplotlib: trực quan hóa kết quả.
- Git/GitHub: quản lý và chia sẻ mã nguồn của nhóm.

## 3.4. Chiến lược huấn luyện (Training Strategy)

### Phân đoạn ngữ nghĩa (Semantic Segmentation)

Mô hình U-Net được huấn luyện trên dữ liệu Semantic Segmentation. Mỗi ảnh đầu vào đi kèm với Ground Truth Mask.

Quá trình huấn luyện gồm:

```text
Ảnh đầu vào
     ↓
Tiền xử lý
     ↓
U-Net
     ↓
Segmentation Prediction
     ↓
So sánh với Ground Truth
     ↓
Tính Loss
     ↓
Cập nhật trọng số
```

Loss có thể sử dụng Cross Entropy Loss để đo sai lệch giữa kết quả dự đoán và nhãn thực tế.

### Ước lượng độ sâu (Depth Estimation)

Đối với Depth Estimation, mô hình MiDaS có thể sử dụng pretrained weights để tạo Depth Map từ ảnh RGB.

Kết quả Depth Map được đưa qua bước xử lý và trực quan hóa để dễ dàng quan sát sự khác biệt về khoảng cách giữa các vùng trong ảnh.

## 3.5. Quy trình suy luận (Inference Process)

Khi đưa một ảnh đường phố mới vào hệ thống:

1. Đọc ảnh RGB.
2. Tiền xử lý và resize ảnh.
3. Đưa ảnh vào mô hình U-Net.
4. Tạo Segmentation Map.
5. Đưa ảnh vào mô hình MiDaS.
6. Tạo Depth Map.
7. Trực quan hóa các kết quả.
8. Kết hợp thông tin Segmentation và Depth để hỗ trợ Scene Understanding.

```text
Input Image
     │
     ├──► U-Net ──► Segmentation Map
     │
     └──► MiDaS ──► Depth Map
                       │
                       ▼
              Scene Understanding
```

## 3.6. Đánh giá mô hình (Model Evaluation)

### Phân đoạn ngữ nghĩa (Semantic Segmentation)

Các chỉ số có thể sử dụng:

- Pixel Accuracy: tỷ lệ pixel được dự đoán chính xác.
- IoU (Intersection over Union): mức độ chồng lấp giữa vùng dự đoán và Ground Truth.
- mIoU (mean IoU): IoU trung bình trên các lớp.

### Ước lượng độ sâu (Depth Estimation)

Các chỉ số thường được sử dụng:

- AbsRel (Absolute Relative Error)
- SqRel (Squared Relative Error)
- RMSE (Root Mean Square Error)
- δ < 1.25

Các chỉ số này giúp đánh giá mức độ chính xác của Depth Map so với Ground Truth.

## 3.7. Cấu trúc các thành phần trong dự án (Project Structure)

Các thành phần chính của dự án gồm:

```text
cv-project/
│
├── models/
│   └── unet/
│       ├── model.py
│       └── inference.py
│
├── preprocessing/
│
├── evaluation/
│   ├── depth_metrics.py
│   ├── difficulty_analysis.py
│   ├── pipeline_metrics.py
│   └── segmentation_metrics.py
│
├── visualization/
│
├── scene_understanding/
│
├── config.py
├── main.py
├── requirements.txt
│
├── 01-problem-definition.md
├── 02-features-output.md
└── 03-solution-tech-ai.md
```

Cấu trúc này giúp tách riêng các thành phần của hệ thống như mô hình AI, tiền xử lý dữ liệu, đánh giá kết quả và trực quan hóa.