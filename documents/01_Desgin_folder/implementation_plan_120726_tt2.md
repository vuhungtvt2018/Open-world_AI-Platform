# Kế hoạch Tái cấu trúc Phase 3: Bám sát Bản đồ Kiến trúc Gemini (The Complete Platform Skeleton)

Sau khi đọc lại kỹ càng toàn bộ nội dung trong `Untitled-1.ini` và đối chiếu với những gì đã làm, tôi nhận thấy Giai đoạn 1 & 2 mới chỉ dừng ở việc "tách" mã nguồn demo hiện tại (Edge Agent) thành các layer, chứ **chưa thực sự xây dựng bộ khung (Skeleton) hoàn chỉnh** của toàn bộ nền tảng (Vision AI Platform) mà Gemini đã thiết kế.

Để thực sự trở thành một nền tảng có thể "lắp Lego" cho nhiều dự án sau này, chúng ta cần tạo sẵn tất cả các "khuôn" (thư mục rỗng/packages) cho các mảng chưa có code. Dưới đây là kế hoạch bổ sung toàn bộ cấu trúc còn thiếu:

## 1. Mở rộng Packages theo đúng sơ đồ
*Về việc đánh số `01-core`, `02-utils`: Trong Python, tên thư viện (module) không được bắt đầu bằng số. Để đơn giản và không làm phức tạp các lệnh import hiện tại, tôi đề xuất vẫn giữ tên thư mục là `core`, `utils`... nhưng sẽ bổ sung README hoặc Docstring để quy định rõ Layer.*

### Xây dựng bộ khung khổng lồ cho `packages/ai/`
Theo Gemini, khối AI cần chứa toàn bộ bài toán CV và vòng đời MLOps. Tôi sẽ tạo các thư mục (và file `__init__.py` trống) cho:
- **`packages/ai/tasks/`**: Thêm `classification/`, `detection/` (đã có yolo_detector), `segmentation/`, `pose/`, `tracking/`, `recognition/`.
- **`packages/ai/engines/`**: Thêm `onnxruntime/`, `tensorrt/`, `openvino/`, `pytorch/`.
- **`packages/ai/engineering/`**: Thêm `dataset/`, `data_ops/`, `annotation/`, `model/`, `training/`, `validation/`, `experiment/`, `metrics/`.
- **`packages/ai/pipeline/`**: Thêm module micro-pipeline nội bộ của AI.

## 2. Hoàn thiện Tầng `services/` (Microservices)
Hiện tại ta mới có `services/database/` và `services/session.py`. Theo bản vẽ của Gemini, ta cần cấu trúc Microservices thực thụ. Tôi sẽ tạo các khuôn rỗng cho:
- `services/camera-service/`
- `services/inference-service/`
- `services/workflow-service/`
- `services/deployment-service/`
- `services/notification-service/`
*(Tôi sẽ đưa logic database/session cũ vào một folder dùng chung `services/shared/` hoặc giữ nguyên tùy bạn).*

## 3. Hoàn thiện Tầng `apps/` (End-user Apps)
Ta mới có `apps/edge-agent/`. Tôi sẽ bổ sung:
- `apps/dashboard/`
- `apps/annotation/`
- `apps/training/`
- `apps/cli/`

## 4. Bổ sung Tầng `solutions/` (Các giải pháp Domain)
Đây là "Đỉnh của kim tự tháp" - nơi đóng gói nghiệp vụ. Tôi sẽ tạo thư mục và các file `pyproject.toml` rỗng cho:
- `solutions/smart-factory/`
- `solutions/warehouse/`
- `solutions/retail/`
- `solutions/traffic/`
- `solutions/robotics/`

## 5. Bổ sung `infrastructure/`
- Tạo thư mục `infrastructure/` để chứa các file Docker, Terraform, K8s sau này.

---

> [!IMPORTANT]
> **User Review Required:**
> 1. Kế hoạch này sẽ sinh ra hàng loạt thư mục rỗng và file `__init__.py` để định hình **toàn bộ bộ xương (skeleton)** của Platform giống y hệt lời khuyên của Gemini. Bạn có đồng ý với việc tạo bộ khung khổng lồ này ngay bây giờ không?
> 2. Về việc đánh số `01-core`, `02-utils`: Tôi đề xuất giữ nguyên tên folder là `core`, `utils` để `import` trong code Python không bị lỗi cú pháp, bạn có đồng ý không? (Nếu bạn nhất quyết muốn có số, ta sẽ phải dùng cấu trúc `packages/01-core/core/...` (nested path) - sẽ hơi cồng kềnh một chút).

Nếu bạn đồng ý, hãy nhấn **Proceed** hoặc phản hồi để tôi bắt đầu thiết lập bộ khung này ngay lập tức!
