from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class CameraConfig(BaseModel):
    """Cấu hình cho thiết bị Camera"""
    camera_id: str = Field(..., description="ID hoặc đường dẫn RTSP của camera")
    resolution: List[int] = Field(default=[1920, 1080], description="Độ phân giải [width, height]")
    fps: int = Field(default=30, description="Số khung hình trên giây")
    skip_frames: int = Field(default=0, description="Số frame bỏ qua để giảm tải")

class ModelConfig(BaseModel):
    """Cấu hình chung cho một model AI"""
    name: str = Field(..., description="Tên model")
    weights_path: str = Field(..., description="Đường dẫn file weights")
    device: str = Field(default="cpu", description="cpu hoặc cuda")
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)

class AIConfig(BaseModel):
    """Cấu hình tổng thể của phân hệ AI"""
    models: Dict[str, ModelConfig] = Field(default_factory=dict, description="Danh sách các models được đăng ký")
    auto_select_engine: bool = Field(default=True, description="Sử dụng AutoBackend thay vì load tay")

class DatabaseConfig(BaseModel):
    """Cấu hình cơ sở dữ liệu"""
    dsn: str = Field(default="sqlite:///./vision_platform.db")
    pool_size: int = Field(default=5)

class PlatformConfig(BaseModel):
    """
    Cấu hình gốc (Root Config) của toàn bộ Vision Platform.
    Tương tự như cấu trúc default.yaml của Ultralytics, nhưng an toàn kiểu dữ liệu hơn nhờ Pydantic.
    """
    project_name: str = Field(default="Vision AI Platform")
    camera: CameraConfig
    ai: AIConfig
    database: DatabaseConfig
    
    class Config:
        # Nếu dùng Pydantic BaseSettings, ta có thể tự động parse từ file .env hoặc YAML
        pass
