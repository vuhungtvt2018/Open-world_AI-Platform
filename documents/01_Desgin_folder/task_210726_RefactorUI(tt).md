# Task Checklist: Decentralized UI Integration

- [x] **Phase 1: Edge API Integration (GlobalMonitoring & ProductionOperation)**
  - [x] Sửa `GlobalMonitoring.tsx` tính toán Throughput, Defect Rate, Latency từ `/history` (Đã có sẵn logic)
  - [x] Sửa `ProductionOperation.tsx` tính tổng số lượng OK, NG từ `/history`
- [x] **Phase 2: Cloud API Integration (Analytics)**
  - [x] Thêm endpoint `GET /api/analytics` vào `main.py` của Cloud Server để đếm lỗi
  - [x] Sửa `Analytics.tsx` kết nối tới `VITE_CLOUD_API_URL/api/analytics`
- [x] **Phase 3: Stub APIs cho Dataset & Model**
  - [x] Thêm `GET /dataset-stats` vào `web_backend.py` (Edge)
  - [x] Thêm `GET /model-registry` vào `web_backend.py` (Edge)
  - [x] Sửa `DatasetManagement.tsx` dùng biến API
  - [x] Sửa `ModelOperation.tsx` dùng biến API
- [x] **Phase 4: Verify**
  - [x] Update walkthrough
