# Implementation Plan: Cloud-Edge Sync & Model Registry

Để hệ thống hoàn chỉnh từ Edge tới Cloud, chúng ta cần xây dựng kiến trúc Data/Model Synchronization 2 chiều. Dưới đây là phác thảo chi tiết cấu trúc Database và Logic đồng bộ tôi đề xuất thực hiện.

## 1. Cấu trúc DB Server (`3_cloud_server/.../models.py`)

Thêm các bảng mới vào PostgreSQL/SQLAlchemy của Cloud Server:

### `CloudAIModel` (Central Model Registry)
Lưu trữ toàn bộ các model từng được train trên Cloud.
- `id` (PK), `name`, `type`, `version`, `format`
- `status`: Có thể là `PRODUCTION` (đã duyệt để đẩy xuống Edge), `STAGING` (đang test), `ARCHIVED` (đã cũ).
- `file_path`: Đường dẫn vật lý trên Cloud Storage.
- `map_acc`, `speed_ms`: Các chỉ số đánh giá lúc train xong.

### `CloudInspectionRecord` (Central Data Lake)
Lưu trữ dữ liệu từ TẤT CẢ các Edge Node gửi lên.
- `id` (PK), `edge_node_id`: Để phân biệt dữ liệu đến từ chuyền nào (VD: `edge_bulong_1`).
- `timestamp`, `ng_detected`, `total_objects`
- `original_image_path`: File ảnh được upload từ Edge lên Cloud.

## 2. Cơ chế Sync Server -> Edge (OTA Model Update)

**Logic:**
1. Cloud Server mở API `GET /api/models/latest`: Trả về danh sách các model có `status='PRODUCTION'`.
2. Edge Node (trong `web_backend.py`) thêm API `/api/sync/models-down`:
   - Gửi Request lên Cloud Server để lấy danh sách model mới nhất.
   - So sánh version với bảng `ai_models` cục bộ ở Edge.
   - Nếu có bản mới: Cập nhật thông tin DB Edge (Trong thực tế sẽ đi kèm hàm download file `.pt` từ Cloud về thư mục `model_checkpoint` cục bộ).

## 3. Cơ chế Sync Edge -> Server (Dataset Upload)

**Logic:**
1. Cloud Server mở API `POST /api/dataset/sync`: Nhận gói tin JSON chứa dữ liệu lỗi (NG/OK) và lưu vào `CloudInspectionRecord`.
2. Edge Node thêm API `/api/sync/dataset-up`:
   - Lấy bảng `SyncState` (đã có sẵn trong Edge) để biết `last_synced_id` là bao nhiêu.
   - Query trong bảng `inspection_records` lấy ra các record MỚI CHƯA ĐƯỢC SYNC.
   - Bắn dữ liệu này lên API Cloud.
   - Nếu thành công, cập nhật lại `last_synced_id`.

## User Review Required

> [!WARNING]
> Việc xây dựng luồng đồng bộ này sẽ chạm tới cả thư mục `1_edge_node` và `3_cloud_server`.
> Trong demo này, thay vì code chức năng upload file ảnh gốc rất nặng qua mạng, tôi sẽ ưu tiên **Sync Metadata (JSON)** giữa 2 Database trước để chứng minh luồng đi của dữ liệu. Nếu bạn đồng ý, tôi sẽ tiến hành.

Hãy bấm **Proceed** để tôi bắt tay vào code phần Server Database và các API đồng bộ nhé!
