# Kế hoạch Thiết lập Môi trường và UV Workspace (Giai đoạn 2)

Cảm ơn bạn đã xác nhận và cung cấp các định hướng rõ ràng. Dựa vào yêu cầu của bạn, tôi xin đệ trình kế hoạch triển khai (Implementation Plan) cho các hạng mục còn lại:

## 1. Đổi tên thư mục và dọn dẹp
- **Đổi tên root directory:** Đổi tên thư mục `1_edge_node` thành `vision-ai-platform`. 
  - *Lưu ý: Vì hệ điều hành Windows có thể khóa thư mục nếu đang mở bằng IDE, tôi sẽ tạo một script Python nhỏ để đổi tên thư mục gốc một cách an toàn. Nếu thất bại (do bị khóa), tôi sẽ nhờ bạn đổi tên thủ công.*
- **Xóa file:** Xóa `pipeline_backup.py` nếu nó tồn tại.

## 2. Di chuyển các file Script
- Tạo thư mục `scripts/` (nếu chưa có).
- Di chuyển `run_main.bat`, `run_setup_app.bat`, và `run_web_backend.bat` vào `scripts/`.
- Cập nhật lại đường dẫn bên trong các file `.bat` này để chúng vẫn hoạt động đúng từ thư mục `scripts/` (bằng cách `cd ..` về root hoặc cấu hình lại đường dẫn tuyệt đối).

## 3. Cấu hình quản lý Package bằng `uv workspaces`
Thiết lập toàn bộ dự án dưới dạng một Monorepo workspace quản lý bởi `uv`. Mỗi component (package) sẽ là một module độc lập. 

### Sơ đồ Dependencies giữa các Component
- `core`: Không phụ thuộc component khác (chứa `pyyaml`...).
- `utils`: Gọi `core`.
- `camera`: Gọi `core`.
- `ai`: Gọi `core`, `utils`, `camera` (chứa `ultralytics`, `anomalib`, `opencv-python`).
- `services`: Gọi `core` (chứa `sqlalchemy`).
- `workflow`: Gọi `core`, `utils`, `camera`, `ai`, `services`.
- `edge-agent`: Gọi `core`, `workflow`, `utils`, `camera`, `services` (chứa `fastapi`, `uvicorn`, `psutil`).

### Danh sách các file `pyproject.toml` sẽ được tạo:

#### [NEW] `/vision-ai-platform/pyproject.toml` (Root)
Khai báo toàn bộ workspace:
```toml
[project]
name = "vision-ai-platform"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "core", "utils", "camera", "ai", "workflow", "services", "edge-agent"
]

[tool.uv.workspace]
members = [
    "packages/*",
    "services",
    "apps/*"
]

# Chỉ định rõ nguồn các dependencies local (từ workspace)
[tool.uv.sources]
core = { workspace = true }
utils = { workspace = true }
camera = { workspace = true }
ai = { workspace = true }
workflow = { workspace = true }
services = { workspace = true }
edge-agent = { workspace = true }
```

#### [NEW] `/vision-ai-platform/packages/core/pyproject.toml`
```toml
[project]
name = "core"
version = "0.1.0"
dependencies = [
    "pyyaml"
]
```

#### [NEW] `/vision-ai-platform/packages/ai/pyproject.toml`
```toml
[project]
name = "ai"
version = "0.1.0"
dependencies = [
    "core",
    "utils",
    "camera",
    "ultralytics",
    "opencv-python",
    "torch",
    "torchvision",
    "anomalib"
]

[tool.uv.sources]
core = { workspace = true }
utils = { workspace = true }
camera = { workspace = true }
```

*(Tương tự cho các file `pyproject.toml` của `utils`, `camera`, `workflow`, `services`, `edge-agent`)*.

---

> [!IMPORTANT]
> **User Review Required:**
> Bạn hãy click **Proceed** (nếu dùng web UI) hoặc xác nhận đồng ý để tôi bắt đầu thực thi việc xóa file, đổi tên thư mục, tạo các thư mục/file cấu hình workspace như trên nhé.
> Nếu bạn muốn thay đổi bất kỳ thành phần dependencies (sự phụ thuộc) nào trong các `pyproject.toml`, hãy phản hồi lại cho tôi biết.
