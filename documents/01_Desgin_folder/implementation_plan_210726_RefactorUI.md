# Kế hoạch Thiết kế UI cho Kiến trúc Phân tán (Decentralized Architecture)

Qua quá trình điều tra toàn bộ 4 khối nguồn (`1_edge_node`, `2_sync_services`, `3_cloud_server`, `0_frontend`), tôi đã nắm rõ luồng dữ liệu của hệ thống:
- **Edge Node:** Chạy AI thời gian thực, lưu SQLite nội bộ.
- **Sync Services:** Quét SQLite định kỳ, đẩy data (JSON + Image) lên Server qua API `/api/sync/up`.
- **Cloud Server:** Nhận data, sinh Vector Embedding, lưu vào PostgreSQL (pgvector) để phục vụ tra cứu toàn cục.

## Trả lời câu hỏi: "UI nên lấy dữ liệu từ Edge hay Server?"

Trong một nền tảng AI phân tán chuẩn mực (như Azure IoT Edge hoặc AWS Greengrass), **Frontend (UI) phải lấy dữ liệu từ CẢ HAI nơi**, nhưng sẽ chia theo chức năng (Pages) cụ thể:

### 1. Dữ liệu lấy từ EDGE NODE (Thời gian thực / Vận hành cục bộ)
Các trang phục vụ Kỹ sư vận hành trực tiếp tại dây chuyền nhà máy (Local Operator):
- **Vision Inspection (Live):** Xem camera trực tiếp, upload ảnh chạy inference, xem kết quả NG/OK tức thời.
- **Edge Dashboard:** Các thống kê tốc độ (latency), tỷ lệ lỗi trong ca làm việc hiện tại.
> **Lý do:** Yêu cầu độ trễ cực thấp (Zero-latency) và phải hoạt động ngay cả khi rớt mạng nội bộ (mất kết nối tới Server).

### 2. Dữ liệu lấy từ CLOUD SERVER (Tổng hợp / Phân tích toàn cục)
Các trang phục vụ Quản đốc, QC Manager, hoặc AI Engineer:
- **Global Monitoring:** Xem biểu đồ tổng hợp từ *nhiều máy Edge* (Máy 1, Máy 2...).
- **Defect Search (Vector DB):** Tiết mục đặc sắc nhất của Cloud. Khi có 1 lỗi lạ, AI Engineer tải ảnh lên UI -> UI gọi API Cloud để tìm tất cả các lỗi tương tự trong quá khứ nhờ `pgvector`.
- **Dataset / Model Operation:** Quản lý tập dữ liệu tổng và phân bổ (deploy) model mới xuống Edge.
> **Lý do:** Server có tài nguyên lưu trữ vô hạn, có CSDL mạnh (PostgreSQL) để join và search, không làm nặng tải cho thiết bị Edge đang bận chạy AI.

---

## User Review Required

> [!IMPORTANT]
> **Đề xuất Kế hoạch Nâng cấp UI:**
> Thay vì hardcode `API_BASE_URL` trỏ về 1 nơi, tôi sẽ thiết lập để UI hoạt động theo mô hình **Hybrid (Lai)**:
> 
> 1. Bổ sung `VITE_EDGE_API_URL` và `VITE_CLOUD_API_URL` vào `.env` của frontend.
> 2. Giữ nguyên trang **Vision Inspection** gọi về Edge.
> 3. Nâng cấp trang **Global Monitoring** (hoặc tạo trang mới **Defect Analytics**) để gọi API tới Cloud Server (hiện đang chạy port `8031`), show bảng lịch sử tổng hợp và tính năng "Tìm kiếm lỗi tương tự bằng hình ảnh".
>
> Bạn có đồng ý với thiết kế kiến trúc UI Hybrid này không? Bấm **Proceed** để tôi bắt đầu code phần Frontend.
