# Database Unification Plan

Mục tiêu: Hợp nhất (đồng bộ hóa) cấu trúc Database giữa Edge Node và Cloud Server để đảm bảo "Edge có bảng nào, Server có bảng đó" và ngược lại, giúp dễ dàng quản lý và mở rộng. Tại các bảng ở Server sẽ bổ sung thêm trường `edge_node_id` để phân biệt dữ liệu từ các máy Edge khác nhau.

## Open Questions
> [!WARNING]
> **Vấn đề kỹ thuật với bảng `QCProductPhotoLibrary`**: 
> Bảng này ở Server đang dùng kiểu dữ liệu `Vector` của PostgreSQL (pgvector) để chứa vector embedding. Tuy nhiên, Database ở Edge đang dùng **SQLite** (không hỗ trợ pgvector). 
> **Đề xuất**: Ở Edge, chúng ta vẫn sẽ tạo bảng `QCProductPhotoLibrary` để đồng bộ thông tin (Metadata), nhưng không có cột `embedding`. Bạn có đồng ý với cách xử lý này không?

## Proposed Changes

### 1. Phía Cloud Server (`3_cloud_server/defect_embedding/models.py`)

- **[NEW] `InspectionRecord`**: Giống Edge + thêm cột `edge_node_id`.
- **[NEW] `BoltObject`**: Giống Edge + liên kết Khóa ngoại với `InspectionRecord`.
- **[NEW] `AnomalyDetail`**: Giống Edge + liên kết Khóa ngoại với `BoltObject`.
- **[MODIFY] `CloudAIModel`**: Đổi tên thành `AIModel` cho giống hệt Edge (hoặc thêm bảng `AIModel` và giữ nguyên API).
- **[DELETE] `EdgeSyncRecord`**: Bảng hứng data thô này sẽ không cần nữa. Khi Edge đồng bộ lên, Server sẽ parse JSON và lưu thẳng vào 3 bảng chuẩn (`InspectionRecord`, `BoltObject`, `AnomalyDetail`).

### 2. Phía Edge Node (`1_edge_node/edge/services/database/models.py`)

- **[NEW] `QCProductPhotoLibrary`**: Tạo bảng này ở Edge (nhưng bỏ cột `embedding` do giới hạn của SQLite) để có thể lưu trữ metadata của thư viện mẫu.
- Các bảng hiện có (`InspectionRecord`, `BoltObject`, `AnomalyDetail`, `AIModel`) sẽ được giữ nguyên (vì chúng đã chuẩn).

### 3. Phía Sync Worker (`2_sync_services/data_sync_worker.py`)

- **[MODIFY] `sync_job`**: Khi đồng bộ, sẽ đọc trọn vẹn 3 bảng `InspectionRecord`, `BoltObject`, `AnomalyDetail` và đóng gói thành JSON gửi lên Cloud.

### 4. Phía Cloud Server API (`3_cloud_server/defect_embedding/main.py`)

- **[MODIFY] `/api/sync/up`**: Thay vì lưu vào `EdgeSyncRecord`, API sẽ parse JSON và lưu trực tiếp vào các bảng `InspectionRecord`, `BoltObject`, `AnomalyDetail` tương ứng, và gán thêm `edge_node_id`.
- **[MODIFY] `/api/analytics`**: Sửa lại API Dashboard để đọc dữ liệu thống kê từ bảng `InspectionRecord` và `BoltObject` (thay vì đọc từ `EdgeSyncRecord` như hiện tại).

## Verification Plan

### Manual Verification
1. Sau khi sửa code, tôi sẽ chạy thử lệnh khởi tạo lại Database trên Server và Edge để kiểm tra cấu trúc bảng.
2. Sẽ sinh thử một số dữ liệu ảo nghiệm thu trên Server để đảm bảo Dashboard (Analytics) vẫn hoạt động mượt mà với 3 bảng mới.
3. Chờ phản hồi của bạn về `QCProductPhotoLibrary` trước khi chốt toàn bộ schema.
