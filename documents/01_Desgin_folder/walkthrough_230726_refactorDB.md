# Database Unification Walkthrough

> [!TIP]
> **Thành quả**: Toàn bộ hệ thống giờ đây đã chia sẻ chung một bộ khung dữ liệu cực kỳ thống nhất và gọn gàng! "Edge có bảng nào, Server có bảng đó".

## 1. Phía Server
Tôi đã quy hoạch lại file `models.py` ở Server:
- **Xóa bỏ hoàn toàn bảng `CloudInspectionRecord` và `EdgeSyncRecord`**. 
- Khởi tạo 3 bảng "chính chủ" y hệt như Edge Node:
  - `InspectionRecord`
  - `BoltObject`
  - `AnomalyDetail`
- Đặc biệt, bảng `InspectionRecord` ở Server đã được gắn thêm trường `edge_node_id` để Server biết chính xác dữ liệu từ máy Edge nào gửi lên (phục vụ cho việc thống kê đa chi nhánh sau này).

## 2. Phía Edge Node
- Tôi đã tạo thêm model `QCProductPhotoLibrary` ở Edge. Bảng này sẽ đóng vai trò giữ siêu dữ liệu (Metadata) tương ứng với kho ảnh mẫu của Server (bỏ đi phần embedding vector vì SQLite không hỗ trợ).

## 3. Data Sync Worker (Luồng đồng bộ)
- Worker ở Edge Node (file `data_sync_worker.py`) đã được "dạy" cách vét toàn bộ dữ liệu ở 3 bảng (`InspectionRecord`, `BoltObject`, `AnomalyDetail`) và đóng gói (nest) thành một cấu trúc JSON lồng nhau siêu việt, gán kèm `edge_node_id = "EDGE_001"`.

## 4. API Endpoints
- Hàm `sync_up` ở Server khi nhận JSON sẽ tự động giải mã ra, và đổ dữ liệu rẽ nhánh chính xác vào 3 bảng tương ứng bằng cơ chế **Quan hệ (Relationship)** của SQLAlchemy.
- Hàm `get_analytics` cho màn hình Vision Analytics giờ đây đã đọc dữ liệu chuẩn từ chính bảng `InspectionRecord`!

> [!NOTE]
> Bạn có thể kiểm tra trực tiếp ở Terminal đang chạy Server. FastAPI đã tự reload và tạo 3 bảng mới chuẩn chỉ trong PostgreSQL của bạn.
> 
> Từ giờ trở đi, quản lý dữ liệu sẽ trở nên cực kỳ trong suốt và "nhẹ đầu"!
