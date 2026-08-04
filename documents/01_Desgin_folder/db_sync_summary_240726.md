# Sơ đồ Cấu trúc & Đồng bộ Database (Edge ↔ Cloud Server)

Dưới đây là danh sách toàn bộ các bảng trong hệ thống tại 2 đầu (Edge và Server) cùng với chiều đồng bộ (Sync Direction) giữa chúng sau quá trình chuẩn hóa.

## 1. Dữ liệu Sản xuất (Production Data)

**Chiều đồng bộ:** `Edge` ➡️ `Cloud Server` (Đẩy lên / Push)

**Vai trò:** Edge là nơi phát sinh dữ liệu (soi lỗi ảnh thời gian thực). Sau khi có kết quả, Edge sẽ đóng gói và đẩy lên Server để lưu trữ dài hạn và phục vụ huấn luyện lại (Retraining).

**Khi nào push**: Khi chạy services sync data `2_sync_services/data_sync_worker.py`

| Bảng tại Edge (SQLite) | Bảng tại Server (PostgreSQL) | Mô tả | Tình trạng API hiện tại |
| :--- | :--- | :--- | :--- |
| `inspection_records` | `InspectionRecord` | Lưu thông tin tổng quan của một lượt chụp ảnh (thời gian, kết quả OK/NG, latency...). | Worker ở Edge gom data đẩy lên `/api/sync/up` mỗi 2 giây thong qua `2_sync_services`  API `/api/sync/up` được tạo ở server |
| `bolt_objects` | `BoltObject` | Lưu danh sách các vật thể (bu-lông) được detect trong bức ảnh của `inspection_records`. Bảng này liên kết 1-N với bảng `inspection_records`. | Worker ở Edge gom data đẩy lên `/api/sync/up` mỗi 2 giây. Hiện tại đang dùng API này để đẩy data xuống |
| `anomaly_details` | `AnomalyDetail` | Lưu chi tiết từng vết xước/lỗi (bounding box, loại lỗi) trên từng con bu-lông. Liên kết 1-N với `bolt_objects`. | Worker ở Edge gom data đẩy lên `/api/sync/up` mỗi 2 giây. Hiện tại đang dùng API này để đẩy data xuống |

---

## 2. Dữ liệu Cấu hình & Trí tuệ (Config & Intelligence)

**Chiều đồng bộ:** `Cloud Server` ➡️ `Edge` (Kéo xuống / Pull)
**Vai trò:** Cloud Server là "Trung tâm chỉ huy" (Single Source of Truth). Những cấu hình, model hay thư viện ảnh chuẩn sẽ được tạo tại Server, sau đó Edge định kỳ tự động tải về để chạy Offline với tốc độ siêu tốc.

| Bảng tại Edge (SQLite) | Bảng tại Server (PostgreSQL) | Mô tả | Tình trạng API hiện tại |
| :--- | :--- | :--- | :--- |
| `qc_product_photo_library` | `QCProductPhotoLibrary` | Thư viện lưu trữ metadata của các mẫu ảnh tham chiếu. **Lưu ý:** Server lưu kèm trường Vector `embedding` (bằng `pgvector`), nhưng khi kéo xuống Edge, trường này bị loại bỏ nhằm tiết kiệm tài nguyên cho thiết bị phần cứng nhỏ. | Hiện tại được sử dụng qua API `/api/sync/down` sau mỗi 2 giây |

**Chiều đồng bộ:** `Cloud Server` ➡️⬅️  `Edge` (Kéo xuống, đẩy lên/Pull, Push)

**Vai trò:**
Bạn nhận xét rất tinh tế! Đúng là khi bạn bấm nút **"Sync with Model Hub"** trên giao diện, thực chất hệ thống đang thực hiện đồng bộ **HAI CHIỀU (Bi-directional Sync)** đối với bảng `ai_models`.

Cụ thể, nếu xem mã nguồn trong `ModelOperation.tsx` (tại hàm `handleSync`), quy trình diễn ra theo 2 bước liên tiếp:

1. **Chiều Lên (Edge ➡️ Server):** Đầu tiên, Edge sẽ gọi API `/api/sync/models-up`. Quá trình này đẩy danh sách các Model hiện đang có mặt tại Edge lên Cloud Server để Cloud Server nắm được máy Edge này đang cài đặt các model nào, trạng thái ra sao.
2. **Chiều Xuống (Server ➡️ Edge):** Ngay sau đó, Edge gọi tiếp API `/api/sync/models-down` để kéo thông tin từ Cloud Server về. Nếu trên Server có bản cập nhật Model mới (phiên bản cao hơn hoặc độ chính xác mAP thay đổi), Edge sẽ tự động cập nhật lại bảng `ai_models` dưới local.

**Tại sao lại cần 2 chiều thông qua nút bấm?**
Bởi vì Model AI thường có dung lượng lớn và việc chuyển đổi model đang chạy (ví dụ từ YOLOv8 sang YOLOv10) có thể ảnh hưởng đến dây chuyền sản xuất. Do đó, khác với Thư viện ảnh (`qc_product_photo_library`) tự động kéo ngầm mỗi 2 giây, việc đồng bộ `ai_models` được thiết kế **thủ công qua nút bấm** để người vận hành (Kỹ sư) có thể chủ động kiểm soát thời điểm nâng cấp hệ thống AI, tránh gián đoạn hệ thống.

Bạn có muốn tôi cập nhật lại file mô tả `db_sync_summary_240726.md` bên cạnh để làm rõ sự khác biệt thú vị này không?

| Bảng tại Edge (SQLite) | Bảng tại Server (PostgreSQL) | Mô tả | Tình trạng API hiện tại |
| :--- | :--- | :--- | :--- |
| `ai_models` | `AIModel` | Danh sách các model AI (YOLO, Segmentation, Anomaly) được Server huấn luyện. Edge sẽ pull thông tin bảng này xuống để biết có model mới không và tải về chạy | Hiện tại pull/push 2 chiều bằng cách bấm nút `Sync with Model Hub` trên UI `Model Operation` |

---

## 3. Dữ liệu Quản trị Đồng bộ (Internal State)

**Chiều đồng bộ:** Không đồng bộ (Chạy độc lập nội bộ)
**Vai trò:** Lưu trữ vết của quá trình đồng bộ (con trỏ thời gian) để hệ thống biết được mình đã lấy dữ liệu đến đâu, phục vụ cơ chế **Delta Sync** (chỉ tải phần chênh lệch thay vì tải toàn bộ).

| Bảng tại Edge (SQLite) | Bảng tại Server (PostgreSQL) | Mô tả |
| :--- | :--- | :--- |
| `sync_states` | `SyncState` | Bảng này chỉ lưu 2 cột `key` và `value`. (Ví dụ: `last_library_sync_time=2026-07-23 09:00:00`, hoặc `last_synced_record_id=1024`). Bảng này không được gửi qua lại giữa 2 đầu mạng. |

> [!TIP]
> **Đánh giá Kiến trúc Hiện tại**
> Với cấu trúc này, mọi tính năng báo cáo/truy vấn nhanh tại nhà máy (như Analytics, Dashboard) đều được chọc thẳng vào database ở Edge để tốc độ đạt tối đa. Ngược lại, những tính năng nặng như **Defect Search bằng công nghệ Vector** hay lưu trữ dữ liệu nhiều tháng vẫn đang lấy qua Cloud Server!
