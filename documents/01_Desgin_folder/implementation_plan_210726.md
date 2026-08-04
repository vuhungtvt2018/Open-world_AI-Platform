# Hoàn thiện MVP bằng Kiến trúc Component-centric

Để biến mã nguồn hiện tại thành một **Product MVP Hoàn chỉnh**, tôi sẽ tiến hành "lắp ráp" các phần đang chạy rời rạc (`web_backend.py`, `pipeline.py`, `YOLODetector`) vào đúng khuôn mẫu (Architecture Skeletons) mà chúng ta đã thiết kế (`BaseVisionTask`, `EventBus`, `AutoBackend`).

## User Review Required

> [!IMPORTANT]
> Dưới đây là kế hoạch thay đổi (Refactoring) vào sâu bên trong ruột của AI Pipeline. Nó sẽ thay đổi cách dữ liệu luân chuyển, giúp hệ thống không còn là một cục code nguyên khối (Monolith) mà trở thành một hệ thống hướng sự kiện (Event-driven).
> Vui lòng bấm **Proceed** nếu bạn đồng ý.

## Proposed Changes

Tôi sẽ chỉnh sửa 3 module cốt lõi sau:

### 1. Refactor AI Tasks (Chuẩn hóa Model)
Tất cả các thuật toán AI giờ sẽ tuân thủ 1 interface chung là `BaseVisionTask` và trả về `InferenceResult` thay vì trả về YOLO dict lộn xộn.
#### [MODIFY] [detection/__init__.py](file:///d:/ivs/my_folder/freelance/visual_inspection/sprint_20250908_20250912_direction/demo_bulong/mti_visual_inspection_ai/1_edge_node/edge/packages/ai/tasks/detection/__init__.py)
- Cho class `YOLODetector` kế thừa từ `BaseVisionTask`.
- Implement đủ 4 hàm: `load_model`, `preprocess`, `predict`, `postprocess`.
- Trả về đối tượng `InferenceResult` (có chứa boxes, masks, metadata).

### 2. Tích hợp Event-driven vào Pipeline
Pipeline không cần phải tự mình lưu DB hay quan tâm frontend làm gì. Nó chỉ làm AI và "hét" lên khi xong việc.
#### [MODIFY] [pipeline.py](file:///d:/ivs/my_folder/freelance/visual_inspection/sprint_20250908_20250912_direction/demo_bulong/mti_visual_inspection_ai/1_edge_node/edge/packages/workflow/pipeline.py)
- Import `default_event_bus` từ `packages.workflow.events`.
- Sau khi có kết quả inference, thay vì `return` một dict lằng nhằng, nó sẽ gọi `default_event_bus.publish(EventBus.EVENT_INFERENCE_DONE, result=...)`.

### 3. Tích hợp Callbacks vào Web Backend
Tách bạch luồng AI và luồng lưu trữ/UI.
#### [MODIFY] [web_backend.py](file:///d:/ivs/my_folder/freelance/visual_inspection/sprint_20250908_20250912_direction/demo_bulong/mti_visual_inspection_ai/1_edge_node/edge/apps/edge-agent/web_backend.py)
- Định nghĩa hàm callback `on_inference_done(result)` để nhận dữ liệu từ `EventBus` và ghi xuống SQLite.
- Xóa bỏ logic ghi Database đồng bộ đang nằm ở API `/inspect`, giúp API phản hồi tức thời hơn.

## Verification Plan

### Manual Verification
1. Sau khi tái cấu trúc, tôi sẽ yêu cầu bạn mở Web UI lên và thử upload một ảnh bu-lông.
2. Kiểm tra xem luồng inference có chạy trơn tru không.
3. Kiểm tra xem kết quả inference có tự động ghi nhận vào thẻ History (nhờ EventBus) hay không.
