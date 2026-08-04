# Database Unification & Sync Walkthrough

> [!TIP]
> **Thành quả**: Bảng `QCProductPhotoLibrary` (Thư viện mẫu) giờ đây đã được đồng bộ hóa hoàn toàn tự động một chiều từ Cloud Server xuống Edge Node! Hệ thống đã đảm bảo kiến trúc Single Source of Truth tuyệt đối.

## Những thay đổi chính

### 1. Phân bổ Cấu hình Rõ ràng
Trong `2_sync_services/config.yaml`, tôi đã thêm khóa:
```yaml
API_SYNC_LIBRARY_DOWN_URL: "http://127.0.0.1:8031/api/sync/library/down"
```
Việc này tách biệt API đẩy data lên (`API_SYNC_UP_URL`) và API kéo library xuống, giúp cho việc đổi IP / Domain sau này cực kỳ thuận tiện và không bị chồng chéo luồng.

### 2. Thiết kế Cloud Server API siêu nhẹ
Trên `main.py`, tôi đã khởi tạo endpoint `GET /api/sync/library/down`.
- API này có bộ lọc `last_update` để đảm bảo cơ chế **Delta Sync** (chỉ tải phần chênh lệch / phần thay đổi mới nhất).
- **Điểm nhấn**: Code cố tình loại bỏ cột `embedding` (vốn chứa vector dữ liệu rất nặng) ra khỏi payload trả về. Giúp network truyền tin gần như tức thời.

### 3. Edge Worker Tự động Upsert
Trong `data_sync_worker.py`:
- Tôi đã thêm mã lệnh tự động khởi tạo bảng `qc_product_photo_library` trong SQLite (chuẩn khớp Schema 100% với PostgreSQL) đề phòng trường hợp bạn cài đặt máy mới (Plug & Play).
- Worker sẽ ngầm định đọc biến `last_library_sync_time` từ SQLite `sync_states`. Gọi lên Server, nhận JSON, thực thi lệnh `INSERT ... ON CONFLICT(ID) DO UPDATE` để quét dữ liệu. 
- Ngay khi nạp xong, nó ghi đè lại timestamp vào `sync_states`. Quá trình này diễn ra hoàn toàn vô hình!

> [!NOTE]
> 🚀 Bạn có thể test ngay bây giờ! Terminal đang tự động load lại. Nếu bạn dùng Postman bắn thử một tấm ảnh (hoặc tạo một Record) vào API `/photos` ở Server, ngay lập tức 2 giây sau, Data Sync Worker ở phía Edge sẽ "nhánh" tay kéo tấm ảnh (Metadata) đó lưu vào SQLite. Cực kỳ nuột nà!
