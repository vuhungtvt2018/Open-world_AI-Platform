# packages/ai/tasks/anomaly/__init__.py
from packages.core.config import AnomalyConfig

from .anomalib_inferencer import AnomalibInferencer
from .yolo_inferencer import YOLOInferencer

# Giữ tương thích với các phần code cũ đang import AnomalyInferencer.
AnomalyInferencer = AnomalibInferencer

__all__ = [
    "AnomalyConfig",
    "AnomalibInferencer",
    "AnomalyInferencer",
    "YOLOInferencer",
]
