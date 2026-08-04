# Hoàn thành Nâng cấp Kiến trúc Frontend Hybrid (Edge + Cloud)

Tuyệt vời! Tôi đã tích hợp thành công mô hình Decentralized Hybrid vào Frontend. Giao diện người dùng (UI) hiện tại đã thông minh hơn, có khả năng linh hoạt chuyển đổi nguồn dữ liệu giữa **Edge Node** và **Cloud Server** tùy theo tác vụ.

## Các thay đổi chính

### 1. Phân tách Môi trường (Environment Variables)
Thay vì hardcode cứng một địa chỉ IP duy nhất, tôi đã thiết lập file [0_frontend/.env](file:///d:/ivs/my_folder/freelance/visual_inspection/sprint_20250908_20250912_direction/demo_bulong/mti_visual_inspection_ai/0_frontend/.env) định nghĩa 2 nguồn cấp dữ liệu song song:
```env
VITE_EDGE_API_URL=http://localhost:59851
VITE_CLOUD_API_URL=http://localhost:8031
```
*(Bạn có thể dễ dàng sửa IP này thành địa chỉ thật của Edge và Cloud trong lúc deploy).*

### 2. Trang "Vision Inspection" -> Lấy từ EDGE
Trang [VisionInspection.tsx](file:///d:/ivs/my_folder/freelance/visual_inspection/sprint_20250908_20250912_direction/demo_bulong/mti_visual_inspection_ai/0_frontend/src/pages/VisionInspection.tsx) đã được cấu hình lại để luôn gọi về biến `VITE_EDGE_API_URL`.
Điều này đảm bảo khi Kỹ sư đứng ở màn hình máy tính dưới xưởng, họ vẫn nhận được kết quả nhận diện cực nhanh (Zero-latency) và máy tính xưởng vẫn chạy ầm ầm ngay cả khi bị đứt cáp mạng internet ra bên ngoài.

### 3. Trang Mới: "Defect Search" -> Lấy từ CLOUD
Tôi đã xây dựng mới hoàn toàn trang [DefectSearch.tsx](file:///d:/ivs/my_folder/freelance/visual_inspection/sprint_20250908_20250912_direction/demo_bulong/mti_visual_inspection_ai/0_frontend/src/pages/DefectSearch.tsx).
Trang này gọi trực tiếp API `POST /search/by-image` của **Cloud Server** (`VITE_CLOUD_API_URL`).
> [!TIP]
> **Tính năng nổi bật:** Khi bạn tải lên một ảnh lỗi (ví dụ: vết xước lạ trên bu-lông), API Cloud sẽ dùng công nghệ Vector Embedding (pgvector) để lục tung cơ sở dữ liệu của **Tất cả các nhà máy / Edge nodes** và trả về danh sách các lỗi giống hệt hoặc tương tự.

### 4. Cập nhật Router và Sidebar
- Gắn thêm menu **Defect Search** (Biểu tượng Kính lúp) vào `Sidebar.tsx`.
- Gắn thẻ quyền `defect-search` cho các vai trò `ENGINEER` và `ADMIN`. (User thường ở xưởng sẽ không thấy nút này để tránh bấm nhầm).

## Hướng dẫn Kiểm tra (Verify)

1. Khởi động Edge API (`web_backend.py` ở port 59851).
2. Khởi động Cloud API (`main.py` của defect_embedding ở port 8031).
3. Mở Terminal mới, trỏ vào thư mục `0_frontend` và chạy:
   ```bash
   npm run dev
   ```
4. Đăng nhập bằng quyền **ENGINEER** hoặc **ADMIN**.
5. Bạn sẽ thấy tab mới **Defect Search** ở cột trái. Nhấp vào, thử upload 1 tấm ảnh lỗi và xem sức mạnh của Cloud Vector Database nhé!
