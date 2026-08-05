from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import numpy as np


class BoundingBox(BaseModel):
    """Chuẩn hóa dữ liệu Bounding Box"""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int
    class_name: Optional[str] = None

class BoudingBoxOBB(BaseModel):
    """
    05082026 - KIET -Chuẩn hóa dữ liệu cho OBB
    """
    cx: float
    cy: float
    weight: float
    height: float
    phi: float
    confidence: float
    class_id: int
    class_name: Optional[str] = None

class InferenceResult(BaseModel):
    """Chuẩn hóa cấu trúc kết quả trả về từ mọi model AI"""
    image_id: Optional[str] = None
    boxes: List[BoundingBox] = Field(default_factory=list)
    masks: Optional[Any] = None  # numpy array hoặc list
    keypoints: Optional[Any] = None
    classification_label: Optional[str] = None
    classification_score: Optional[float] = None
    raw_output: Optional[Any] = None  # Để lưu kết quả gốc từ model nếu cần
    processing_time_ms: Optional[float] = None


class BaseVisionTask(ABC):
    """
    Khuôn mẫu Abstract Base Class cho tất cả các tác vụ Vision AI (Detection, Classification, Anomaly...).
    Lấy cảm hứng từ BasePredictor của Ultralytics.
    """

    def __init__(self, model_path: str, device: str = "cpu", config: Optional[Dict[str, Any]] = None):
        self.model_path = model_path
        self.device = device
        self.config = config or {}
        self.model = None
        self.load_model()

    @abstractmethod
    def load_model(self) -> None:
        """Logic load model (ONNX, PyTorch, OpenVINO...) từ self.model_path"""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """Tiền xử lý ảnh: resize, normalize, chuyển thành tensor..."""
        pass

    @abstractmethod
    def predict(self, preprocessed_data: Any) -> Any:
        """Đưa dữ liệu qua mạng neural để lấy raw output"""
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any, original_image: np.ndarray) -> InferenceResult:
        """Hậu xử lý raw output (NMS, scale bboxes...) và trả về InferenceResult chuẩn"""
        pass

    def run(self, image: np.ndarray) -> InferenceResult:
        """
        Hàm chính Pipeline cho 1 lượt inference hoàn chỉnh.
        Workflow sẽ chỉ gọi hàm này.
        """
        preprocessed = self.preprocess(image)
        raw_output = self.predict(preprocessed)
        result = self.postprocess(raw_output, image)
        return result
