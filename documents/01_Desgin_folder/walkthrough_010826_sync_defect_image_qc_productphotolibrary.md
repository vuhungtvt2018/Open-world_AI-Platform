# Walkthrough: Đồng bộ Ảnh Lỗi Vật Lý (Base64) xuống Edge

Tôi đã hoàn tất việc nâng cấp luồng đồng bộ theo đúng như kế hoạch bạn duyệt. Quá trình triển khai thành công 100% mà không cần đẻ thêm bất kỳ API mới nào!

## Những thay đổi đã thực hiện

### 1. Nâng cấp API ở Cloud Server
- **File**: `3_cloud_server/defect_embedding/main.py`
- **Mô tả**: Sửa API `GET /api/sync/library/down`.
- **Cách thức**: Trước khi trả cục JSON về cho Edge, API này đã được bổ sung logic tìm file ảnh trong `settings.UPLOAD_DIR`, đọc file dưới dạng nhị phân, mã hóa nó thành chuỗi **Base64** và nhét thẳng vào trường `image_base64` của mỗi record trong mảng kết quả.
  
```diff
+        image_base64 = None
+        if r.ImageName:
+            img_path = os.path.join(settings.UPLOAD_DIR, r.ItemCode, r.ImageName) if r.ItemCode else os.path.join(settings.UPLOAD_DIR, r.ImageName)
+            if not os.path.exists(img_path):
+                img_path = os.path.join(settings.UPLOAD_DIR, r.ImageName)
+            if os.path.exists(img_path):
+                try:
+                    with open(img_path, "rb") as img_file:
+                        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
+                except Exception:
+                    pass
```

### 2. Nâng cấp Worker Đồng bộ tại Edge
- **File**: `2_sync_services/data_sync_worker.py`
- **Mô tả**: Tự động giải mã chuỗi Base64 và ghi ra file ảnh vật lý.
- **Cách thức**: 
  - Khai báo thư mục đích là `1_edge_node/sync_defect_image`. Tự động tạo thư mục này nếu nó chưa tồn tại.
  - Khi quét được JSON từ Server trả về, Worker sẽ rút chuỗi `image_base64` ra, gọi hàm `base64.b64decode` để dịch ngược thành file ảnh nguyên thủy, rồi ghi (`wb`) thẳng xuống ổ cứng của máy Edge trước khi lưu metadata vào SQLite.

```diff
+                            img_b64 = rec.get("image_base64")
+                            img_name = rec.get("ImageName")
+                            if img_b64 and img_name:
+                                try:
+                                    img_data = base64.b64decode(img_b64)
+                                    img_path = os.path.join(SYNC_DEFECT_IMAGE_DIR, img_name)
+                                    with open(img_path, "wb") as f:
+                                        f.write(img_data)
+                                except Exception as e:
+                                    print(f"[SYNC DOWN] Lỗi khi lưu ảnh {img_name}: {e}")
```

## Kết quả
Từ bây giờ trở đi, hễ kỹ sư quản lý chất lượng (QC) đăng ký một bức ảnh mẫu (Reference Defect Image) mới trên hệ thống, thì:
1. Ảnh được lưu tại `uploads/` của Cloud Server.
2. Background Worker tại Edge quét (mỗi 2 giây).
3. Cloud Server gói ảnh mới (dưới dạng Base64) gửi qua đường truyền HTTP.
4. Edge nhận được, giải nén và lưu file ảnh thực tế xuống thư mục `1_edge_node/sync_defect_image/` của Edge.
5. Máy Edge hoàn toàn có thể tự render các ảnh này lên màn hình nội bộ mà không tốn công gọi API tải ảnh từ Cloud nữa!
