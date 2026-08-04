# Kế hoạch Tái cấu trúc Thư mục (Project Restructuring)

Góp ý của bạn là một **nguyên tắc thiết kế hệ thống (System Design Principle) cực kỳ chuẩn xác**. Khi dự án mở rộng thành hệ thống phân tán (Distributed System), việc trộn lẫn mã nguồn của Cloud/Server, Edge Node và Sync Worker trong cùng một thư mục sẽ gây ra rất nhiều khó khăn cho việc đóng gói (Docker) và triển khai (CI/CD) sau này.

> [!IMPORTANT]
> **User Review Required**
> Dưới đây là đề xuất cấu trúc cây thư mục mới. Nếu bạn đồng ý, tôi sẽ dùng lệnh tạo thư mục và di chuyển (move) mã nguồn tương ứng, đồng thời sửa lại toàn bộ các dòng `import` trong code để hệ thống không bị lỗi.

## Cấu trúc Đề xuất (Proposed Architecture)

Chúng ta sẽ chia lại thư mục gốc `mti_visual_inspection_ai/` thành 3 vùng độc lập hoàn toàn:

```text
mti_visual_inspection_ai/
├── 1_edge_node/                (Mã nguồn chạy tại nhà máy - Edge IPC/Jetson)
│   ├── app/                    # Core AI (camera.py, pipeline.py, utils.py)
│   ├── database/               # Edge Local DB (SQLite SQLAlchemy)
│   ├── web_backend.py          # Edge HMI (FastAPI)
│   └── config.yaml
│
├── 2_sync_services/            (Các dịch vụ đồng bộ chạy ngầm độc lập)
│   ├── data_sync_worker.py     # Đổi tên từ sync_inspection_logs.py
│   ├── config_sync_worker.py   # Dành cho tính năng OTA Config sau này
│   └── model_sync_worker.py    # Dành cho tính năng OTA Model sau này
│
└── 3_cloud_server/             (Mã nguồn triển khai trên Cloud / Server Trung tâm)
    └── defect_embedding/       # Thư mục embedding_server hiện tại (PostgreSQL + pgvector)
```

## Các bước Thực thi (Execution Steps)

Nếu bạn "Chốt" kế hoạch này, tôi sẽ thực hiện các bước sau:

### [MOVE] Tách Cloud Server
- Chuyển toàn bộ thư mục `embedding_server/` vào `3_cloud_server/defect_embedding/`.

### [MOVE] Tách Sync Services
- Rút thư mục `app/services/` hiện tại ra ngoài và đổi tên thành `2_sync_services/`.
- Sửa lại file `sync_inspection_logs.py` để nó có thể chạy độc lập như một Microservice (có file `requirements.txt` riêng hoặc dùng chung nhưng tách biệt ngữ cảnh). Nó vẫn sẽ đọc cùng file `visual_inspection_edge.db`.

### [MOVE] Quy hoạch Edge Node
- Đẩy toàn bộ các file còn lại (`app/`, `web_backend.py`, `config.yaml`, `run_*.bat`, `requirements*.txt`) vào thư mục `1_edge_node/`.

### [MODIFY] Sửa lỗi Import
- Cập nhật lại đường dẫn import trong `web_backend.py` để trỏ đúng tới Sync Worker (hoặc chuyển Sync Worker thành một tiến trình khởi chạy song song qua file `.bat` thay vì import trực tiếp vào FastAPI để đảm bảo tính cô lập tuyệt đối).

## Open Question
- Đối với **Sync Services**, bạn muốn import nó chạy ngầm bên trong `web_backend.py` (FastAPI) như cũ, hay muốn tôi tách nó ra thành một file chạy riêng biệt hoàn toàn (ví dụ: chạy bằng lệnh `python data_sync_worker.py` ở một terminal khác)? Việc tách ra hoàn toàn file chạy riêng sẽ đúng chuẩn Microservice hơn.
