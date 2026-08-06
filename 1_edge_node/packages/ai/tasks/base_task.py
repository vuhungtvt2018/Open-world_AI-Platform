from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

from packages.core.config import BaseConfig


class BoundingBox(BaseModel):
    """Chuẩn hóa dữ liệu Bounding Box"""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float
    class_id: int
    class_name: Optional[str] = None


"""
06082026 - KHAI - Add more attributes and method for anomaly detection
"""
class InferenceResult(BaseModel):
    """Chuẩn hóa cấu trúc kết quả trả về từ mọi model AI"""
    # --- Thông tin chung ---
    image_id: Optional[str] = None
    processing_time_ms: Optional[float] = None

    # --- Kết quả Detection / Segmentation ---
    boxes: List[BoundingBox] = Field(default_factory=list)
    masks: Optional[Any] = None  # numpy array hoặc list

    # --- Kết quả Classification / Anomaly General Status ---
    is_ng: bool = False
    classification_label: Optional[str] = None
    classification_score: Optional[float] = None  # Hoặc score của anomaly

    # --- Trường mở rộng dành riêng cho Anomaly Detection ---
    anomaly_map_raw: Optional[Any] = None  # Raw anomaly map (np.ndarray)
    anomaly_map_masked: Optional[Any] = None  # Anomaly map sau khi nhân mask
    heatmap_display: Optional[Any] = None  # Heatmap RGB/BGR đã superimpose (np.ndarray)
    overlap_ratio: float = 0.0
    visualized_image: Optional[Any] = None  # Crop vis / Image đã vẽ bounding box

    # --- Raw Output từ model ---
    raw_output: Optional[Any] = None

    # Cấu hình Pydantic v2 cho phép chứa kiểu dữ liệu arbitrary (như np.ndarray)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert kết quả về dạng Dict tương thích với pipeline Anomaly cũ."""
        return {
            "score": self.classification_score or 0.0,
            "is_ng": self.is_ng,
            "hm_disp": self.heatmap_display,
            "am_raw": self.anomaly_map_raw,
            "am_masked": self.anomaly_map_masked,
            "overlap_ratio": self.overlap_ratio,
            "anomalies": [
                {
                    "k": i,
                    "bbox_in_object_crop": [
                        int(b.x1),
                        int(b.y1),
                        int(b.x2),
                        int(b.y2),
                    ],
                    "area_px": int((b.x2 - b.x1) * (b.y2 - b.y1)),
                }
                for i, b in enumerate(self.boxes)
            ],
            "crop_vis": self.visualized_image,
        }


"""
06082026 - KHAI - Modify __init__, postprocess and run
"""
class BaseVisionTask(ABC):
    """
    Khuôn mẫu Abstract Base Class cho tất cả các tác vụ Vision AI (Detection, Classification, Anomaly...).
    Lấy cảm hứng từ BasePredictor của Ultralytics.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        config: Optional[Union[Dict[str, Any], BaseConfig]] = None
    ):
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
    def postprocess(self, raw_output: Any, *args, **kwargs) -> InferenceResult:
        """Hậu xử lý raw output (NMS, scale bboxes...) và trả về InferenceResult chuẩn"""
        pass

    def run(self, image: np.ndarray, *args, **kwargs) -> InferenceResult:
        """
        Hàm chính Pipeline cho 1 lượt inference hoàn chỉnh.
        Workflow sẽ chỉ gọi hàm này.
        """
        preprocessed = self.preprocess(image)
        raw_output = self.predict(preprocessed)
        result = self.postprocess(raw_output, *args, **kwargs)
        return result
