# Kế hoạch Tái cấu trúc `1_edge_node` theo Kiến trúc Gemini (Monorepo DAG)

Theo yêu cầu của bạn, tôi sẽ tiến hành tái cấu trúc mã nguồn trong thư mục `1_edge_node` tuân theo triết lý Monorepo và DAG (Directed Acyclic Graph) gồm 7 tầng mà Gemini đã đề xuất.

**Nguyên tắc cốt lõi:**
- **Không phụ thuộc ngược (One-way dependency):** Tầng trên (ví dụ: `apps`, `services`, `workflow`) mới được gọi tầng dưới (`ai`, `camera`, `utils`, `core`). Tuyệt đối không có chiều ngược lại.
- **Tách bạch rõ ràng:** Chia nhỏ `app/` hiện tại thành các package riêng biệt đóng vai trò như các khối "Lego".

Vì bạn yêu cầu tôi không được suy diễn và phải hỏi ý kiến trước khi tiến hành bất kỳ thay đổi nào, dưới đây là **bản nháp cấu trúc thư mục mới** và **danh sách mapping (chuyển đổi) các file hiện tại sang cấu trúc mới**.

## Open Questions & Yêu cầu xác nhận (User Review Required)

> [!IMPORTANT]
> **Xin bạn vui lòng xem kỹ và trả lời các câu hỏi sau trước khi tôi bắt đầu gõ code/di chuyển file:**
> 1. Cấu trúc thư mục mới (bên dưới) đã đúng ý bạn chưa? Bạn có muốn đổi tên `1_edge_node` thành `vision-ai-platform` (hoặc tên khác) luôn không, vì giờ nó sẽ mang hình hài của một monorepo?
> 2. Các file `run_main.bat`, `run_setup_app.bat`, `run_web_backend.bat` tôi dự định đưa vào thư mục `scripts/` ở thư mục gốc để cho gọn, bạn có đồng ý không?
> 3. File `pipeline_backup.py` có vẻ là file backup cũ, tôi có thể xóa file này không? Hay bạn muốn giữ lại ở thư mục `archive/`?
> 4. Trong tương lai bạn dùng công cụ quản lý package nào cho Monorepo? (Poetry, uv workspaces, hay chỉ dùng `requirements.txt` thường)? Tạm thời tôi sẽ tạo file `pyproject.toml` trống làm gốc.

---

## Proposed Changes (Chi tiết Tái cấu trúc)

Dưới đây là sơ đồ ánh xạ (mapping) từ file cũ sang cấu trúc mới:

### [Tầng 1] `packages/core` (Base Types, Config, Exceptions)
Nơi chứa các cấu hình gốc và định nghĩa cơ bản.
#### [NEW] `packages/core/__init__.py`
#### [MODIFY] `packages/core/config.py` (Di chuyển từ `app/config.py`)

### [Tầng 2] `packages/utils` (Stateless Helpers)
Các hàm tiện ích dùng chung.
#### [NEW] `packages/utils/__init__.py`
#### [MODIFY] `packages/utils/utils.py` (Di chuyển từ `app/utils.py`)
#### [MODIFY] `packages/utils/visualize.py` (Di chuyển từ `app/visualize.py`)
#### [MODIFY] `packages/utils/cleaner.py` (Di chuyển từ `app/cleaner.py`)

### [Tầng 3] `packages/camera` (Hardware & Streaming)
Giao tiếp phần cứng, đọc luồng video.
#### [NEW] `packages/camera/__init__.py`
#### [MODIFY] `packages/camera/camera.py` (Di chuyển từ `app/camera.py`)

### [Tầng 4] `packages/ai` (AI Engines & Tasks)
Chứa các bài toán AI, không phụ thuộc vào business logic.
#### [NEW] `packages/ai/__init__.py`
#### [MODIFY] `packages/ai/tasks/anomaly/anomaly.py` (Di chuyển từ `app/detectors/anomaly.py`)
#### [MODIFY] `packages/ai/tasks/yolo_detector/yolo_detector.py` (Di chuyển từ `app/detectors/yolo_detector.py`)
#### [MODIFY] `packages/ai/engineering/data_collector/data_collector.py` (Di chuyển từ `app/tools/data_collector.py`)

### [Tầng 5] `packages/workflow` (DAG Execution Engine)
Ghép nối Camera và AI lại với nhau thành pipeline.
#### [NEW] `packages/workflow/__init__.py`
#### [MODIFY] `packages/workflow/pipeline.py` (Di chuyển từ `app/pipeline.py`)

### [Tầng 6] `services` (Microservices / Backend Daemons)
Nơi xử lý giao tiếp database, session, internal API.
#### [NEW] `services/__init__.py`
#### [MODIFY] `services/database/crud.py` (Di chuyển từ `app/database/crud.py`)
#### [MODIFY] `services/database/models.py` (Di chuyển từ `app/database/models.py`)
#### [MODIFY] `services/database/session.py` (Di chuyển từ `app/database/session.py`)
#### [MODIFY] `services/session.py` (Di chuyển từ `app/session.py`)

### [Tầng 7] `apps` (End-user Applications)
Ứng dụng thực thi cuối cùng (trường hợp này là Edge Agent).
#### [NEW] `apps/edge-agent/__init__.py`
#### [MODIFY] `apps/edge-agent/web_backend.py` (Di chuyển từ `web_backend.py`)
#### [MODIFY] `apps/edge-agent/config.yaml` (Di chuyển từ `config.yaml`)

### Dọn dẹp (Cleanup)
#### [DELETE] Thư mục `app/` (Sau khi đã di dời toàn bộ file)

---

## Verification Plan
1. **Kiểm tra cú pháp:** Sau khi di chuyển file, tôi sẽ cập nhật lại toàn bộ các câu lệnh `import` trong tất cả các file tương ứng (Ví dụ đổi từ `from app.config import ...` thành `from packages.core.config import ...`).
2. **Kiểm tra DAG:** Đảm bảo `packages/utils` không gọi tới `packages/ai`, `packages/camera` không gọi `packages/workflow` v.v.

Vui lòng phản hồi các câu hỏi ở phần **Open Questions** để tôi có thể bắt đầu bước viết code và di chuyển file!
