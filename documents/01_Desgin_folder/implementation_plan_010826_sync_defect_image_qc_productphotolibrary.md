# Kế hoạch Đồng bộ Ảnh Vật Lý cùng Metadata (Không tạo thêm API)

## Mục tiêu
Theo yêu cầu, ta sẽ **KHÔNG TẠO THÊM API** chuyên biệt để download ảnh. Thay vào đó, API `GET /api/sync/library/down` hiện có sẽ đính kèm luôn nội dung của file ảnh vật lý vào trong cục JSON trả về (sử dụng chuẩn mã hóa Base64) để Edge có thể lưu thẳng xuống ổ cứng.

## Open Questions
Không có.

## Proposed Changes

### 1. Sửa API tại Cloud Server
#### [MODIFY] `3_cloud_server/defect_embedding/main.py`
- Sửa hàm `sync_library_down` (API `GET /api/sync/library/down`):
  - Khi duyệt qua danh sách các bản ghi `QCProductPhotoLibrary`, dùng `os.path.join(settings.UPLOAD_DIR, ItemCode, ImageName)` để tìm đường dẫn file ảnh vật lý.
  - Đọc file ảnh dưới dạng nhị phân (`rb`), sau đó mã hóa bằng thư viện `base64`.
  - Đính kèm chuỗi base64 này vào biến kết quả JSON (ví dụ: trường `"image_base64"`).
  - (Xử lý fallback: nếu không tìm thấy file, trả về trường `"image_base64": null` hoặc bỏ trống để tránh lỗi crash API).

### 2. Sửa Tiến trình Đồng bộ tại Edge
#### [MODIFY] `2_sync_services/data_sync_worker.py`
- Khai báo thư mục `SYNC_DEFECT_IMAGE_DIR` nằm tại `1_edge_node/sync_defect_image`.
- Tạo thư mục nếu chưa tồn tại (`os.makedirs`).
- Ở phần xử lý dữ liệu trả về từ API `GET /api/sync/library/down` (`# Đồng bộ Thư viện mẫu (Server -> Edge)`):
  - Kiểm tra xem JSON của từng record có chứa `"image_base64"` hay không.
  - Nếu có, decode chuỗi base64 ngược lại thành file ảnh và ghi (`wb`) vào thư mục `SYNC_DEFECT_IMAGE_DIR/<ImageName>`.
  - Sau đó mới thực thi câu lệnh SQL `INSERT ... ON CONFLICT DO UPDATE` để ghi metadata vào SQLite như cũ.

## Verification Plan
1. Apply code cho cả Server và Edge Worker.
2. Thêm thử 1 bức ảnh vào Cloud Server.
3. Chờ tiến trình Background Worker quét sau 2 giây.
4. Kiểm tra xem thư mục `1_edge_node/sync_defect_image/` đã tự động sinh ra và chứa bức ảnh vật lý vừa đồng bộ xuống hay chưa. Mở file ảnh lên xem có xem được bình thường không (chứng minh mã hóa base64 đúng).
