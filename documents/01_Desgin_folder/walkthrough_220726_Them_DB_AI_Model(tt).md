# Walkthrough: Cloud-Edge Synchronization

Tôi đã hoàn thành việc xây dựng kiến trúc Data & Model Synchronization hai chiều giữa Edge Node và Cloud Server. Dưới đây là các luồng xử lý đã được lập trình:

## 1. Cloud Server Architecture
Tôi đã nâng cấp Database của Cloud Server (trong thư mục `3_cloud_server/defect_embedding`) để chuẩn bị cho việc quản lý tập trung:
- **`CloudAIModel`**: Bảng Central Registry lưu tất cả các bản Model. Khi Kỹ sư AI train xong 1 mô hình mới, họ sẽ lưu vào bảng này với trạng thái `PRODUCTION`.
- **`CloudInspectionRecord`**: Data Lake tập trung. Lưu toàn bộ siêu dữ liệu báo lỗi (OK/NG) từ TẤT CẢ các nhà máy/dây chuyền gửi về.

Tôi đã mở 2 API Endpoint trên Cloud:
1. `GET /api/models/latest`: Trả về danh sách model đang có hiệu lực.
2. `POST /api/dataset/sync`: Nhận gói tin JSON từ các Edge Node gửi lên và Insert vào Data Lake.

## 2. Edge Node Sync
Tôi đã tích hợp 2 API Sync vào Backend của phần mềm chạy tại xưởng (`1_edge_node/edge/apps/edge-agent/web_backend.py`):

### Tải Mô hình mới (`/api/sync/models-down`)
- **Hoạt động:** Edge Node gửi yêu cầu lên Cloud Server lấy danh sách các mô hình `PRODUCTION`. 
- Nó đối chiếu với bảng `ai_models` dưới local. Nếu có Model mới (chưa có ID) hoặc Version cao hơn, nó sẽ tạo record mới (Status: `STANDBY`) để chuẩn bị tải file weights về.

### Đẩy Dữ liệu báo lỗi (`/api/sync/dataset-up`)
- **Hoạt động:** Tận dụng bảng `SyncState` (đã có sẵn), hệ thống tự động lọc ra các biên bản kiểm tra (InspectionRecord) có ID lớn hơn `last_synced_record_id`.
- Edge đóng gói các bản ghi này lại và gửi `POST` lên Cloud Server.
- Nếu thành công, Edge cập nhật lại trạng thái `last_synced_record_id`, đảm bảo **không bao giờ gửi trùng dữ liệu**, ngay cả khi mất mạng tạm thời.

> [!NOTE]
> Trong môi trường thực tế, 2 API Sync của Edge Node sẽ được gắn vào một **Background Scheduler (Cron job)** để tự động chạy ngầm mỗi 5-10 phút. Đối với tính năng truyền file dung lượng lớn (ảnh lỗi 5MB/tấm hoặc file model YOLOv8 50MB/file), thường người ta sẽ cấu hình upload qua minIO/AWS S3 SDK thay vì bắn trực tiếp Base64 qua REST API để tránh ngẽn RAM.
