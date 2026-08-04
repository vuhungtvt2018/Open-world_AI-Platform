# Kế hoạch Chuyển đổi UI sử dụng API tại Edge Node

## Mục tiêu
Tối ưu hóa tốc độ tải dữ liệu cho toàn bộ UI bằng cách chuyển việc truy xuất dữ liệu từ Cloud Server (port `8031`) về thẳng Edge Node (port `8000`). Điều này đảm bảo nhà máy có thể kiểm tra kết quả tức thời mà không bị trễ (latency) do đường truyền mạng.

## Open Questions
> [!WARNING]
> **Vấn đề kỹ thuật với tính năng "Defect Search" (Tìm kiếm lỗi bằng ảnh)**
> 
> Hiện tại màn hình `DefectSearch.tsx` đang gọi API `/search/by-image`. API này sử dụng công nghệ tìm kiếm Vector (pgvector) của PostgreSQL trên Cloud Server. Tuy nhiên, theo như thiết kế trước đó của chúng ta: **Edge Node sử dụng SQLite và KHÔNG lưu trữ vector embedding** (để tối ưu Edge).
> 
> Nếu chuyển hoàn toàn `DefectSearch` về Edge, Edge sẽ không thể tìm kiếm bằng Vector được. Bạn muốn tôi xử lý tính năng này thế nào?
> - **Lựa chọn 1 (Khuyên dùng)**: Chuyển toàn bộ các màn hình khác (Analytics, Dashboard, Dataset...) về Edge, nhưng RIÊNG `DefectSearch` vẫn giữ nguyên gọi lên Cloud (vì đây là tính năng nặng cần sức mạnh Cloud).
> - **Lựa chọn 2**: Bỏ qua màn hình `DefectSearch` hoặc chỉ làm giả (mock) cho màn hình này ở Edge.
> 
> Xin hãy cho tôi biết ý kiến của bạn!

## Proposed Changes

### Frontend (`0_frontend`)
- **[MODIFY] `src/pages/Analytics.tsx`**: 
  - Đổi `CLOUD_API_URL` thành `EDGE_API_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000'`.
- **[MODIFY] `src/pages/DefectSearch.tsx`**:
  - *(Tùy thuộc vào lựa chọn của bạn ở trên, tôi sẽ giữ nguyên CLOUD hoặc chuyển sang EDGE)*.

### Edge Backend (`1_edge_node/edge/apps/edge-agent/web_backend.py`)
- **[MODIFY] `web_backend.py`**:
  - Bổ sung API `GET /api/analytics` vào Edge (Port từ Cloud sang).
  - Logic của API này sẽ query trực tiếp trên bảng `inspection_records` (thống kê OK/NG, sản lượng) và `qc_product_photo_library` (thống kê mã lỗi Pareto) từ CSDL SQLite tại Edge. Vì dữ liệu ở Edge là dữ liệu tức thời và mới nhất (real-time), biểu đồ sẽ nhảy rất nhanh và chính xác!

## Verification Plan
1. Viết code chuyển đổi API `/api/analytics` vào `web_backend.py` ở Edge.
2. Sửa file Frontend.
3. Reload Frontend và kiểm tra xem đồ thị Vision Analytics có hiển thị mượt mà với dữ liệu từ Edge hay không.
