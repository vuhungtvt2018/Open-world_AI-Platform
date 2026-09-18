from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Literal

"""
18092026 - KHAI - Combine PhotoBase, PhotoCreate and PhotoRead into PhotoInfo
"""
class PhotoInfo(BaseModel):
    ID: int
    SM_ID: Optional[str] = None
    ItemCode: Optional[str] = None
    ImageName: Optional[str] = None
    ImageType: Optional[int] = None
    ErrorDetail: Optional[str] = None
    Insert_PIC: Optional[str] = None
    Insert_Date: Optional[datetime] = None
    Update_PIC: Optional[str] = None
    Update_Date: Optional[datetime] = None
    model_config = {"from_attributes": True}

"""
18092026 - KHAI - Change PhotoUpdateErrorDetail to UpdateErrorDetailPayload
"""
class UpdateErrorDetailPayload(BaseModel):
    ErrorDetail: str

"""
18092026 - KHAI - Change SearchRequestByImage to SearchByImageRequest
"""
class SearchByImageRequest(BaseModel):
    top_k: int = Field(default=10, ge=1, le=200)
    metric: str = Field(default="cosine")
    ItemCode: Optional[str] = None

class SearchResultItem(BaseModel):
    ID: int
    distance: float
    SM_ID: Optional[str] = None
    ItemCode: Optional[str] = None
    ImageName: Optional[str] = None
    ErrorDetail: Optional[str] = None

"""
18092026 - KHAI - Move SyncModelRequest, SyncModelsPayload, SyncCameraRequest, 
SyncCamerasPayload to schemas.py
"""
class SyncModelRequest(BaseModel):
    id: str
    name: str
    type: str
    format: str
    version: str
    map_acc: float = None
    status: str
    file_path: str
    speed_ms: str = None

class SyncModelsPayload(BaseModel):
    models: List[SyncModelRequest]

# 23082026-KIET-Chuẩn hóa một cấu hình camera nhận từ Edge Node
class SyncCameraRequest(BaseModel):
    camera_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=150)
    source_type: Literal["rtsp", "basler"]
    source_url: Optional[str] = None
    serial_number: Optional[str] = None
    assigned_task: Optional[Literal["detection", "inspection"]] = None
    enabled: bool = False
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    base_revision: int = 0

# 23082026-KIET-Chuẩn hóa payload đồng bộ camera của một Edge Node
class SyncCamerasPayload(BaseModel):
    edge_node_id: str = Field(min_length=1, max_length=100)
    cameras: List[SyncCameraRequest]