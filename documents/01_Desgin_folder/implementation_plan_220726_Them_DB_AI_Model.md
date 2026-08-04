# Model Registry Implementation Plan

Bối cảnh: Hiện tại màn hình **Model Operation** đang hiển thị dữ liệu cứng (hardcoded) trong API `/model-registry`, không phản ánh đúng các model thực tế đang được sử dụng ở Edge Node (được định nghĩa trong `config.yaml`).

Để hệ thống lớn hoạt động chuyên nghiệp (MLOps), việc có một Database quản lý Model (Model Registry) là hoàn toàn cần thiết để theo dõi phiên bản, định dạng, độ chính xác và quản lý việc cập nhật model (Hot-swap/OTA updates).

## Proposed Changes

### `services/database/models.py`
Thêm bảng `AIModel` vào DB để lưu trữ thông tin các model AI.

#### [MODIFY] [models.py](file:///d:/ivs/my_folder/freelance/visual_inspection/sprint_20250908_20250912_direction/demo_bulong/mti_visual_inspection_ai/1_edge_node/edge/services/database/models.py)
Thêm cấu trúc bảng:
```python
class AIModel(Base):
    """Bảng quản lý Model (Model Registry)"""
    __tablename__ = "ai_models"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    type: Mapped[str] = mapped_column(String(50))     # Detection, Anomaly, Alignment, Classification
    format: Mapped[str] = mapped_column(String(20))   # PT, ONNX, IR...
    version: Mapped[str] = mapped_column(String(20))
    map_acc: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20))   # PRODUCTION, STAGING, INACTIVE
    file_path: Mapped[str] = mapped_column(String(255))
    speed_ms: Mapped[str] = mapped_column(String(20), nullable=True)
```

### `apps/edge-agent/web_backend.py`
1. Viết một hàm khởi tạo (seed data) chạy lúc khởi động server để quét file `config.yaml` và tự động ghi các model đang được dùng vào bảng `ai_models`.
2. Sửa lại API `/model-registry` để Query từ Database thay vì trả về mảng JSON cứng.

#### [MODIFY] [web_backend.py](file:///d:/ivs/my_folder/freelance/visual_inspection/sprint_20250908_20250912_direction/demo_bulong/mti_visual_inspection_ai/1_edge_node/edge/apps/edge-agent/web_backend.py)
* Viết hàm `seed_model_registry(db, cfg)` để lấy thông tin từ `cfg.MODEL_PATH`, `cfg.ANOMALY_MODEL_PATH`, v.v...
* Cập nhật endpoint:
```python
@app.get("/model-registry")
async def get_model_registry():
    # Gọi db.query(AIModel).all()
    # Serialize trả về đúng format UI đang mong đợi (id, name, type, format, version, mAP, status, speed)
```

## User Review Required

> [!IMPORTANT]
> Khi thêm bảng mới vào Database (SQLite), có thể file `visual_inspection_edge.db` cũ sẽ thiếu bảng này gây ra lỗi (nếu chưa setup Alembic cho migration). 
> Tôi sẽ xử lý bằng cách tạo lệnh `Base.metadata.create_all(bind=engine)` trong Backend để tự động tạo bảng nếu nó chưa tồn tại mà không làm hỏng dữ liệu cũ.

Bạn có đồng ý với kế hoạch trên không? Nhấn **Proceed** để tôi tiến hành sửa code nhé!
