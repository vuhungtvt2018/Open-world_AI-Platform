# Kế hoạch Triển khai UI Hybrid (Edge + Cloud)

- [x] Bước 1: Cấu hình Environment
  - [x] Tạo file `0_frontend/.env`
  - [x] Thêm `VITE_EDGE_API_URL` (trỏ tới `localhost:59851`)
  - [x] Thêm `VITE_CLOUD_API_URL` (trỏ tới `localhost:8031`)
- [x] Bước 2: Tái cấu trúc API call
  - [x] Sửa `VisionInspection.tsx` để lấy base URL từ `import.meta.env.VITE_EDGE_API_URL`
- [x] Bước 3: Nâng cấp `GlobalMonitoring` (hoặc tạo `DefectSearch`)
  - [x] Tạo `DefectSearch.tsx` để kết nối API Cloud
  - [x] Thêm Defect Search vào `App.tsx` và `Sidebar.tsx`
- [ ] Bước 4: Chạy thử và Verify
  - [ ] Build & chạy thử Frontend
  - [ ] Test lấy data từ cả Edge và Cloud
