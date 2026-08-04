# Kế hoạch Loại bỏ Mockup & Tích hợp Dữ liệu thật (Decentralized UI)

Như bạn đã chỉ ra, ngoài trang `Vision Inspection` và `Defect Search` đã được đấu nối API thật, các trang quản trị khác đang hiển thị số liệu "fix cứng" (mockup). 

Tôi đã rà soát lại các endpoints hiện có trên **Edge Server (port 8000)** và **Cloud Server (port 8031)** và đề xuất phương án xử lý như sau:

## User Review Required

> [!WARNING]
> Hiện tại **Cloud Server** mới chỉ có bảng Vector Database (`qc_productphotolibrary`) dành cho bài toán Search, và **Edge Server** chỉ có bảng `InspectionRecord` lưu lịch sử chạy. 
> 
> Với các trang phức tạp như **Dataset Management**, **Model Operation**, chúng ta chưa hề có Database hay Backend API phục vụ chúng. Xin hãy cho ý kiến về định hướng: bạn muốn tôi code thêm Database/API thật cho chúng, hay tạm thời dựng API trả data giả (Stub API) ở Backend để UI đọc xuống cho chuẩn kiến trúc?

## Phân tích & Đề xuất Chi tiết

### 1. `GlobalMonitoring.tsx` & `ProductionOperation.tsx` (Lấy từ Edge API)
- **Tình trạng:** `ProductionOperation` đang dùng `setInterval` tự tăng bộ đếm. `GlobalMonitoring` có gọi `/history` nhưng chưa móc nối vào các thẻ UI.
- **Giải pháp:** 
  - Khai thác tối đa API `GET /history` hiện có của Edge Node.
  - Tính tổng số lượng hàng OK/NG, tỷ lệ Defect Rate, và độ trễ trung bình (Avg Latency) từ mảng `history.results`.
  - Hiển thị trực tiếp lên UI (Real-time).

### 2. `Analytics.tsx` (Lấy từ Cloud API)
- **Tình trạng:** Hoàn toàn là số liệu hardcode. Theo kiến trúc, trang này phải lấy data tổng hợp từ Cloud.
- **Giải pháp đề xuất:** 
  - Sẽ tạo một Endpoint mới `GET /api/analytics` trên **Cloud Server (main.py)**.
  - Endpoint này sẽ đếm số lượng lỗi (`NG`) theo từng loại (`ErrorDetail`) từ bảng `qc_productphotolibrary` để vẽ biểu đồ **Defect Pareto Analysis**.

### 3. `DatasetManagement.tsx` & `ModelOperation.tsx` (Lấy từ Edge/Cloud API)
- **Tình trạng:** Hoàn toàn là giao diện UI, chưa có backend tương ứng.
- **Giải pháp đề xuất:**
  - Vì phạm vi của 2 module này rất lớn (yêu cầu quản lý file dataset, version model), để phục vụ MVP, tôi đề xuất tạo 2 API tĩnh `GET /api/dataset-stats` và `GET /api/model-registry` trên Edge Server. UI sẽ call API này thay vì fix cứng trong file `.tsx`. Điều này đảm bảo UI tuân thủ đúng kiến trúc gọi API.

## Kế hoạch Triển khai (Execution Plan)

Nếu bạn đồng ý, tôi sẽ thực hiện theo thứ tự:
1. **Bước 1:** Cập nhật logic Frontend cho `GlobalMonitoring` và `ProductionOperation` tính toán số liệu thật từ `GET /history`.
2. **Bước 2:** Bổ sung Endpoint `/api/analytics` vào Cloud Server và cập nhật `Analytics.tsx` lấy data từ Cloud.
3. **Bước 3:** (Tùy chọn) Viết các API stub cho Dataset/Model để Frontend lấy data động hoàn toàn.

Bạn có đồng ý với kế hoạch này không, hay bạn muốn thu hẹp/mở rộng phần nào?
