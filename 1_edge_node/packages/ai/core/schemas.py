from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field

class TaskType(str, Enum):
    """
    05082026 - KIET - Khai báo các tác vụ AI được hỗ trợ
    """
    DETECTION = "detection"
    OBB = "obb"

class BoundingBox(BaseModel):
    """
    05082026 - KIET - Chuẩn hóa bounding box không xoay theo định dạng XYXY.
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center(self):
        """
        05082026 - KIET - Tính tọa độ Bounding box
        """
        x_center = (self.xmin + self.xmax)/2.0
        y_center = (self.ymin + self.ymax)/2.0

        return x_center, y_center

class OrientedBoundingBox(BaseModel):
    """
    05082026 - KIET - Chuẩn hóa orient bounding box theo định dạng XYWHR.
    """

    cx: float
    cy: float
    width: float
    height: float
    angle: float
    polygon: list[list[float]] = Field(default_factory = list)

    @property
    def center(self):
        """
        05082026 - KIET - Trả về tọa độ tâm của oriented bounding box.
        """
        return self.cx, self.cy


class Prediction(BaseModel):
    """
    05082026 - KIET - Chuẩn hóa prediction dùng chung cho OBB và Detection
    """

    class_id: int
    class_name: Optional[str] = None
    confidence: float
    bbox: BoundingBox | None = None
    obb: OrientedBoundingBox | None = None 
    track_id: int | None = None


    @property
    def center(self) -> tuple[float, float] | None:
        """
        05082026 - KIET - Lấy tọa độ tâm phục vụ counting và tracking
        """

        if self.obb is not None:
            return self.obb.center

        if self.bbox is not None:
            return self.bbox.center
        return None


class InferenceResult(BaseModel):
    """
    05082026 - KIET - Chuẩn hóa kết quả trả về từ model.
    """
    image_id: Optional[str] = None
    task: TaskType
    image_width: int
    image_height: int
    predictions: list[Prediction] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    raw_output: Any = None
