# Walkthrough: Model Registry Implementation

Tôi đã hoàn tất việc xây dựng **Model Registry** để quản lý các Model AI bằng Database, thay vì dùng dữ liệu giả. Dưới đây là tóm tắt các thay đổi:

## Changes Made

### 1. Database Schema
Thêm bảng `AIModel` vào file `models.py` của Edge Node. Bảng này đóng vai trò là một **Model Registry**, lưu trữ toàn bộ các thông tin siêu dữ liệu (metadata) của các AI models (YOLO, Anomalib, ResNet...):
- `id`: Mã định danh model
- `name`: Tên file model (VD: `bulong_8ly_s.pt`)
- `type`: Loại mô hình (Detection, Anomaly, Keypoints)
- `version`, `format`, `map_acc`, `speed_ms`: Các thông số kỹ thuật để đánh giá.

### 2. Auto-Seeding (Khởi tạo dữ liệu tự động)
Trong file `web_backend.py`, tôi đã viết hàm `seed_models()`.
Khi bạn khởi động Backend (chạy `run_edge_node.cmd`), hệ thống sẽ quét file cấu hình `config.yaml` và tự động trích xuất các model đang được sử dụng ở xưởng để lưu vào Database nếu DB đang trống.

### 3. API Integration
Cập nhật lại API `/model-registry` (trước đó bị hardcode JSON). Giờ đây API này sẽ truy xuất trực tiếp danh sách mô hình từ bảng `AIModel` trong SQLite và gửi lên giao diện người dùng.

## Validation Result

- Không làm ảnh hưởng đến các tính năng cũ (do tôi sử dụng `Base.metadata.create_all` an toàn để thêm bảng).
- Màn hình **Model Operation** trên Frontend giờ đây sẽ hiển thị tên model y hệt như những file `.pt` hay `.onnx` bạn đang cấu hình trong source code.

> [!TIP]
> Bước tiếp theo của hệ thống nếu nâng cấp lên Production là: Frontend có thể gọi API upload 1 file model `.pt` mới. Backend sẽ nhận file này, cập nhật vào bảng `AIModel`, sau đó tự động reload lại `InferenceEngine` bằng model mới mà không cần phải khởi động lại toàn bộ chương trình (Hot-swap Model).
