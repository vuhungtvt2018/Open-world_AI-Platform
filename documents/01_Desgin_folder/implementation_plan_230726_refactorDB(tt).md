# Kế hoạch Đồng bộ Thư viện mẫu (QCProductPhotoLibrary)

## Mục tiêu
Đồng bộ một chiều từ Cloud Server (PostgreSQL) xuống Edge Node (SQLite) cho bảng `QCProductPhotoLibrary`. Cơ chế sử dụng là **Pull-based Delta Sync** (Edge chủ động gọi API lấy phần dữ liệu mới/bị sửa đổi).

## Trả lời câu hỏi về `config.yaml`
> Bạn hỏi: *"Có nên cập nhật `config.yaml` cho các DB syn giữa edge và server không?"*
> **Trả lời:** CÓ. Rất nên cập nhật. Chúng ta sẽ thêm một key mới là `API_SYNC_LIBRARY_DOWN_URL` vào file `config.yaml` để khai báo tường minh endpoint này. Việc này giúp tách biệt rõ ràng luồng đẩy viễn trắc (`API_SYNC_UP_URL`) và luồng tải thư viện (`API_SYNC_LIBRARY_DOWN_URL`), tránh code chết cứng URL (hardcode).

## Proposed Changes

### 1. Phía Sync Services Config (`2_sync_services/config.yaml`)
- **[MODIFY] `config.yaml`**: Thêm biến `API_SYNC_LIBRARY_DOWN_URL: "http://127.0.0.1:8031/api/sync/library/down"`

### 2. Phía Cloud Server API (`3_cloud_server/defect_embedding/main.py`)
- **[MODIFY] `main.py`**: Thêm endpoint `GET /api/sync/library/down`.
  - Nhận tham số `last_update` (mặc định lấy tất cả nếu không có).
  - Trả về danh sách dữ liệu (ID, SM_ID, ItemCode, ImageName, ImageType, ErrorDetail, Insert_PIC, Insert_Date, Update_PIC, Update_Date) có `Update_Date > last_update`.
  - **KHÔNG** trả về trường `embedding` để tối ưu băng thông.

### 3. Phía Edge Node Worker (`2_sync_services/data_sync_worker.py`)
- **[MODIFY] `data_sync_worker.py`**:
  - Tạo bảng `qc_product_photo_library` trong SQLite (nếu chưa có).
  - Đọc mốc thời gian `last_library_sync_time` từ bảng `sync_states`.
  - Gọi API `GET /api/sync/library/down?last_update=...`.
  - Thực hiện lệnh **Upsert** (Cập nhật nếu đã tồn tại, Thêm mới nếu chưa có) vào SQLite.
  - Lưu lại `last_library_sync_time` dựa trên `Update_Date` lớn nhất nhận được.
  - Tần suất gọi: Có thể lồng vào vòng lặp hiện tại nhưng kiểm tra mỗi 10 chu kỳ (khoảng 20 giây) để giảm tải, hoặc kiểm tra mỗi chu kỳ. (Đề xuất: Mỗi chu kỳ 2s gọi 1 lần vẫn rất nhẹ vì Server dùng DB index trả kết quả trong 1ms).

## Open Questions
- Không có vấn đề gì bất thường. Việc đồng bộ một chiều từ Server xuống Edge cực kì an toàn và không gây xung đột (Conflict) vì chúng ta coi Server là Nguồn chân lý duy nhất (Single Source of Truth) cho bảng dữ liệu này. Edge chỉ lấy về để xem/sử dụng.

## Verification Plan
1. Viết code cho endpoint `/api/sync/library/down` ở Cloud.
2. Cập nhật `data_sync_worker.py`.
3. Quan sát log Terminal để đảm bảo Worker tải thành công dữ liệu từ Cloud xuống ghi vào SQLite.
