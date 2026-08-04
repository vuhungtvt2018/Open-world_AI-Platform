Dưới đây là danh sách tổng hợp toàn bộ các bảng CSDL (Database Schema) và **Chiều đồng bộ (Sync Direction)** giữa Edge Node và Cloud Server trong dự án của bạn ở thời điểm hiện tại:

### 1. Nhóm Dữ liệu Viễn trắc & Kết quả AI (Telemetry & AI Results)

Đây là nhóm dữ liệu có dung lượng lớn nhất và được tạo ra liên tục trong quá trình Edge Node (Camera) hoạt động.

| Tên Bảng (Table) | Giải thích chức năng | Chiều đồng bộ | Tình trạng API hiện tại |
| :--- | :--- | :---: | :--- |
| **`inspection_records`** | Lưu lịch sử mỗi lần Camera chụp (thời gian, tổng số lỗi, latency...) | **Edge ➡️ Server** | Worker ở Edge gom data đẩy lên `/api/sync/up` mỗi 2 giây. |
| **`bolt_objects`** | Chi tiết từng con bu-lông được cắt ra (crop) trong 1 lần chụp. | **Edge ➡️ Server** | Đóng gói lồng (nested) bên trong json của `inspection_records`. |
| **`anomaly_details`** | Chi tiết từng vết lỗi (vết xước, mẻ, gỉ sét) trên từng con bu-lông. | **Edge ➡️ Server** | Đóng gói lồng (nested) bên trong json của `bolt_objects`. |

### 2. Nhóm Quản lý Model AI (Model Registry)

Giúp Server (Trung tâm) biết được các trạm Edge dưới nhà máy đang chạy những phiên bản Model AI nào, độ chính xác (mAP) và tốc độ (ms) ra sao.

| Tên Bảng (Table) | Giải thích chức năng | Chiều đồng bộ | Tình trạng API hiện tại |
| :--- | :--- | :---: | :--- |
| **`ai_models`** (Edge) <br/> **`cloud_ai_models`** (Server) | Danh mục các Model AI (Detection, Anomaly, Keypoint) đang được kích hoạt. | **Edge ➡️ Server** | Edge push thông tin các model đang dùng lên API `/api/models/sync-up` của Server. |

### 3. Nhóm Dữ liệu Tri thức (Vector Search & Metadata)

Phục vụ cho tính năng "Defect Search" (Tìm lỗi tương tự trong CSDL mẫu).

| Tên Bảng (Table) | Giải thích chức năng | Chiều đồng bộ | Tình trạng API hiện tại |
| :--- | :--- | :---: | :--- |
| **`qc_product_photo_library`** | Thư viện mẫu các sản phẩm Lỗi (NG). Ở Server sẽ có thêm cột `embedding` (Vector) để search AI, ở Edge chỉ có Text Metadata. | **Server ➡️ Edge** <br/> *(Chưa code tự động)* | Hiện tại mới chỉ đồng bộ Cấu trúc Bảng (Schema). Dữ liệu thật mới chỉ tập trung xử lý ở Server (qua API `/photos`). Có thể viết script tải metadata về Edge sau nếu cần thiết. |

### 4. Nhóm Trạng thái Hệ thống (System & Config)

Dùng để giữ trạng thái đồng bộ hoặc điều khiển cấu hình (Thresholds) từ xa.

| Tên Bảng (Table) | Giải thích chức năng | Chiều đồng bộ | Tình trạng API hiện tại |
| :--- | :--- | :---: | :--- |
| **`sync_states`** | Chỉ tồn tại ở **Edge**. Dùng để đánh dấu mốc thời gian (timestamp) của record cuối cùng đã gửi thành công lên Server (tránh gửi trùng). | Không đồng bộ | (Chỉ dùng Local ở SQLite) |
| Cấu hình **`config.yaml`** | Cập nhật các thông số ngưỡng (Threshold), ngày dọn rác ổ cứng... từ xa. | **Server ➡️ Edge** | Edge chủ động gọi API `/api/sync/down` của Server để lấy File Config và tự động ghi đè xuống Local. |

---
**💡 Tổng kết:**

- Luồng chính yếu của hệ thống hiện tại là luồng **Bottom-Up (Edge đẩy lên Server)** để cung cấp Data cho màn hình Vision Analytics và Defect Search.
- Luồng **Top-Down (Server rót xuống Edge)** hiện tại mới chỉ áp dụng cho việc *Cập nhật file Config từ xa*.
