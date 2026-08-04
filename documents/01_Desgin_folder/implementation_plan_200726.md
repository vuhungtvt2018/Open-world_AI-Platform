# Kế hoạch Bổ sung Chuẩn hóa Nâng cao (Học từ Ultralytics)

Dựa trên nguyên tắc tối thượng: **Tuyệt đối CHỈ THÊM MỚI và HỢP THỨC HÓA thiết kế, KHÔNG xóa bỏ hay thay đổi logic/kiến trúc hiện tại của team.**

Dưới đây là kế hoạch chi tiết chia thành 5 hạng mục bổ sung (Add-ons). Các hạng mục này hoạt động như những interface (giao diện) hoặc module độc lập để làm giàu thêm tính chuyên nghiệp cho Platform mà không làm vỡ các block (packages) hiện tại.

## User Review Required

> [!IMPORTANT]
> Vui lòng xem xét từng bước bổ sung dưới đây. Nếu bạn đồng ý, hãy chọn **Proceed**.
> Nếu bạn chỉ muốn thực hiện một số bước cụ thể (ví dụ: chỉ làm Base Classes và AutoBackend trước), hãy phản hồi lại cho tôi biết để tôi điều chỉnh Task list.

---

## Proposed Changes (Các bổ sung chi tiết)

### Bước 1: Base Classes (Tính đa hình cho AI)
Chúng ta sẽ bổ sung các Abstract Base Class (ABC) để định hình khuôn mẫu cho mọi Model AI được tích hợp vào nền tảng.
#### [NEW] `packages/ai/tasks/base_task.py`
- Tạo class `BaseVisionTask` (kế thừa `ABC`).
- Định nghĩa các abstract method: `load_model()`, `preprocess()`, `predict()`, `postprocess()`.
- Tạo Pydantic model `InferenceResult` để chuẩn hóa định dạng kết quả (gồm bboxes, labels, scores, masks...).
*(Các model hiện tại của bạn trong `yolo_detector.py` có thể từ từ refactor để kế thừa `BaseVisionTask` sau này, tạm thời chúng ta cứ đặt khuôn mẫu vào trước).*

### Bước 2: AutoBackend (Quản lý Framework linh hoạt)
Bổ sung một lớp Factory trừu tượng hóa các framework học máy (PyTorch, ONNX, TensorRT).
#### [NEW] `packages/ai/engines/autobackend.py`
- Tạo class `AutoBackend`.
- Chứa logic để kiểm tra đuôi file weights (`.pt`, `.onnx`, `.engine`) và tự động gọi engine phù hợp để load model.
- Điều này giống y hệt cách Ultralytics cô lập sự phức tạp của hardware inference.

### Bước 3: Centralized Configuration (Schema Cấu hình)
Bổ sung Schema định hình cấu hình bằng Pydantic, giúp cấu hình an toàn, có validation (kiểm tra kiểu dữ liệu) và tự động sinh code nhắc lệnh (autocomplete).
#### [NEW] `packages/core/config_schema.py`
- Định nghĩa các class kế thừa `pydantic.BaseModel` như: `PlatformConfig`, `CameraConfig`, `AIConfig`, `DatabaseConfig`.
*(File `config.py` cũ của bạn vẫn giữ nguyên để parse YAML, chúng ta chỉ thêm file schema này để team có thể dùng làm Data Type khi code).*

### Bước 4: Event-driven System (Hệ thống Callbacks)
Bổ sung cơ chế phát sự kiện (Pub/Sub) nội bộ.
#### [NEW] `packages/workflow/events.py`
- Tạo class `EventBus` đơn giản (để Đăng ký - `subscribe` và Phát - `publish` sự kiện).
- Định nghĩa các hằng số sự kiện: `FRAME_READY`, `INFERENCE_DONE`, `ANOMALY_DETECTED`.
#### [NEW] `packages/workflow/callbacks.py`
- Tạo class `BaseCallback`.
- Thêm sẵn một số callback mẫu như `LoggerCallback` (chuyên ghi log) hay `AlertCallback` (chuyên gửi cảnh báo).

### Bước 5: Unified CLI (Giao diện dòng lệnh hợp nhất)
Bổ Bản ứng dụng CLI tập trung sử dụng thư viện `Typer` (hoặc `argparse`), giúp tương tác với nền tảng qua 1 câu lệnh duy nhất (như `yolo ...`).
#### [NEW] `apps/cli/main.py`
- Tạo ứng dụng Typer với các sub-commands (tạm thời để rỗng ruột hoặc gọi in ra log):
  - `vision-cli run edge-agent`
  - `vision-cli train`
  - `vision-cli export`
#### [NEW] `apps/cli/pyproject.toml`
- Bổ sung cấu hình workspace cho app cli và khai báo entrypoint để khi gõ `vision-cli` trên terminal nó sẽ chạy `apps/cli/main.py`.

---

## Verification Plan

### Manual Verification
1. Sau khi code xong 5 file trên, tôi sẽ cung cấp cho bạn một đoạn script ngắn (ví dụ `test_platform.py`).
2. Script này sẽ dùng thử `BaseVisionTask`, tạo một sự kiện bằng `EventBus` và gọi thử `AutoBackend`.
3. Bạn chạy script đó để thấy các "viên gạch mới" chạy trơn tru mà không hề đụng chạm hay làm chết logic của hệ thống cũ.
