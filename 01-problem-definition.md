# 1. Định nghĩa bài toán

## Nhiệm vụ

**Phân đoạn ngữ nghĩa (Semantic Segmentation) + Ước lượng độ sâu (Depth Estimation) → Hiểu ngữ cảnh cảnh vật (Scene Understanding)**

Cho một ảnh đường phố duy nhất, hệ thống cần tạo ra hai đầu ra dự đoán dày đặc (dense predictions):

| Đầu ra | Mô tả |
|---------|---------|
| Semantic Segmentation Mask | Gán nhãn lớp cho từng pixel (đường, xe hơi, bầu trời, tòa nhà, người đi bộ, ...) |
| Depth Map | Ước lượng khoảng cách từ camera đến từng pixel |

Kết hợp hai đầu ra này giúp hệ thống hiểu được ngữ cảnh của cảnh vật, bao gồm:

- **Trong ảnh có những đối tượng nào?**
- **Các đối tượng đó cách camera bao xa?**

---

## Ý nghĩa và ứng dụng

### Xe tự lái (Autonomous Driving)

Xe cần xác định:

- Khu vực có thể di chuyển (drivable area).
- Các đối tượng xung quanh như xe hơi, người đi bộ, biển báo,...
- Khoảng cách đến các đối tượng để hỗ trợ điều hướng an toàn.

### Robot và hệ thống dẫn đường

Robot cần:

- Nhận biết không gian trống để di chuyển.
- Xác định vị trí vật cản.
- Ước lượng khoảng cách đến vật cản để tránh va chạm.

### Thực tế tăng cường (AR / Mixed Reality)

Việc chèn các đối tượng ảo vào môi trường thực đòi hỏi:

- Hiểu bố cục cảnh vật.
- Biết độ sâu của từng vùng trong ảnh để hiển thị tự nhiên và chính xác.

---

## Luồng xử lý đầu vào – đầu ra

```text
┌─────────────────┐
│  Ảnh đường phố  │
│   (H × W × 3)   │
└────────┬────────┘
         │
 ┌───────┴────────┐
 │                │
 ▼                ▼

U-Net           MiDaS
(Phân đoạn)    (Ước lượng độ sâu)

 ▼                ▼

Segmentation    Depth
Mask            Map

      │
      ▼

Scene Understanding
(Hiểu ngữ cảnh cảnh vật)