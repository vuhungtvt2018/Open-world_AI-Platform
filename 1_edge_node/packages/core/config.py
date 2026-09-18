import yaml
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

"""
06082026 - KHAI - Create base class for configuration
"""
@dataclass
@abstractmethod
class BaseConfig(ABC):
    """Base class for configuration, allowing for importing from YAML file"""
    @staticmethod
    @abstractmethod
    def from_yaml(path: str | Path) -> "BaseConfig":
        raise NotImplementedError()

"""
06082026 - KHAI - Make AppConfig subclass of BaseConfig
"""
@dataclass
class AppConfig(BaseConfig):
    # ===== Required (NO defaults) - must come first =====
    RTSP_URL: str
    CAPTURE_DIR: str
    PRODUCT_NAME: str
    MODEL_PATH: str
    MODEL_COUNTING_PATH: Optional[str] = None

    # ===== Edge/Robot Config (defaults) =====
    PROJECT_PATH: Optional[str] = None
    EDGE_IP: Optional[str] = None
    PORT: Optional[int] = 8000
    EDGE_CODE: Optional[str] = None

    # ===== Camera (defaults) =====
    FLIP_VERTICAL: bool = False                        # lật dọc frame khi thu
    CAM_WIDTH: Optional[int] = None                   # độ rộng mong muốn (có thể bị backend bỏ qua)
    CAM_HEIGHT: Optional[int] = None                  # độ cao mong muốn (có thể bị backend bỏ qua)

    # ===== Capture / display =====
    DISPLAY_SCALE: float = 0.25
    PAD_RATIO: float = 0.1

    # ===== Anomalib =====
    ANOMALY_BACKEND: str = "anomalib"

    # ===== Anomalib =====
    ANOMALY_MODEL_PATH: str = "exported_models/weights/onnx/model.onnx"
    ANOMALY_DEVICE: str = "CPU"
    ANOMALY_INPUT_SIZE: int = 512
    ANOMALY_SCORE_THRESHOLD: float = 0.8
    ANOMALY_INSIDE_OVERLAP_MIN: float = 0.5

    """
    25082026 - KHANH - Add runtime class-name mapping for YOLO anomaly models
    """
    YOLO_CLASS_NAMES: Dict[int, str] = field(default_factory=dict)

    # Anomaly region extraction & display
    ANOMALY_MIN_AREA_RATIO: float = 0.001
    ANOMALY_BBOX_PAD_RATIO: float = 0.02
    SAVE_ALL_ANOMALIES: bool = False
    SHOW_ALL_ANOMALY_BOXES: bool = True

    # Additional Anomalib config
    ANOMALY_AMAP_THRESHOLD: float = 0.7
    REDO_CENTER_CROP: bool = True
    CENTER_CROP: int = 448

    # ===== Data collector tool =====
    DATA_COLLECTOR_FOLDER: str = "data_collector"      # thư mục con dưới CAPTURE_DIR/PRODUCT_NAME

    # ===== Sync API =====
    API_SYNC_ENABLE: bool = False
    API_SYNC_URL: str = "http://192.168.0.122:8030//WebApi/QualityControl/VisualInspection/SaveScannedImage"
    API_TIMEOUT: float = 8.0
    PRODUCTION_INSTRUCTION: str = ""
    ITEM_CODE: str = ""
    OTHER_INFO_DEFAULT: str = ""

    # === Defect Classification API (bật/tắt phân loại lỗi) ===
    DEFECT_CLS_ENABLE: bool = False
    DEFECT_CLS_URL: str = "http://127.0.0.1:8031/search/by-image"
    DEFECT_CLS_TOPK: int = 1
    DEFECT_CLS_METRIC: str = "cosine"
    DEFECT_CLS_SIM_THRESHOLD: float = 0.8
    
    # === Keypoints Detection ===
    KEYPOINT_DETECTION: bool = False
    KEYPOINTS_MODEL_PATH: str = 'model_checkpoint\bulong_8ly_keypoints.pt'
    KEYPOINTS_SCORE_THRESHOLD: float = 0.5

    # === Automatic Disk Cleanup ===
    DISK_CLEANUP_DAYS: float = 15.0
    DISK_CLEANUP_INTERVAL_HOURS: float = 24.0

    @staticmethod
    def from_yaml(path: str | Path) -> "AppConfig":
        path = str(Path(path))
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)
        return AppConfig(
            # Required
            RTSP_URL=raw["RTSP_URL"],
            CAPTURE_DIR=raw["CAPTURE_DIR"],
            
            # Edge/Robot
            PROJECT_PATH=raw.get("PROJECT_PATH"),
            EDGE_IP=raw.get("EDGE_IP"),
            PORT=int(raw.get("PORT", 8000)) if "PORT" in raw else None,
            EDGE_CODE=raw.get("EDGE_CODE"),
            PRODUCT_NAME=raw["PRODUCT_NAME"],
            MODEL_PATH=raw["MODEL_PATH"],
            MODEL_COUNTING_PATH=raw.get("MODEL_COUNTING_PATH"),

            # Camera
            FLIP_VERTICAL=bool(raw.get("FLIP_VERTICAL", False)),
            CAM_WIDTH=(int(raw["CAM_WIDTH"]) if "CAM_WIDTH" in raw and raw["CAM_WIDTH"] is not None else None),
            CAM_HEIGHT=(int(raw["CAM_HEIGHT"]) if "CAM_HEIGHT" in raw and raw["CAM_HEIGHT"] is not None else None),

            # Capture / display
            DISPLAY_SCALE=float(raw.get("DISPLAY_SCALE", 0.25)),
            PAD_RATIO=float(raw.get("PAD_RATIO", 0.1)),

            # Anomaly backend
            ANOMALY_BACKEND=raw.get("ANOMALY_BACKEND", "anomalib"),

            # Anomalib
            ANOMALY_MODEL_PATH=raw.get("ANOMALY_MODEL_PATH", "exported_models/weights/onnx/model.onnx"),
            ANOMALY_DEVICE=raw.get("ANOMALY_DEVICE", "CPU"),
            ANOMALY_INPUT_SIZE=int(raw.get("ANOMALY_INPUT_SIZE", 512)),
            ANOMALY_SCORE_THRESHOLD=float(raw.get("ANOMALY_SCORE_THRESHOLD", 0.8)),
            ANOMALY_INSIDE_OVERLAP_MIN=float(raw.get("ANOMALY_INSIDE_OVERLAP_MIN", 0.5)),
            YOLO_CLASS_NAMES={
                int(class_id): str(class_name)
                for class_id, class_name in (raw.get("YOLO_CLASS_NAMES") or {}).items()
            },

            # Anomaly region extraction & display
            ANOMALY_MIN_AREA_RATIO=float(raw.get("ANOMALY_MIN_AREA_RATIO", 0.001)),
            ANOMALY_BBOX_PAD_RATIO=float(raw.get("ANOMALY_BBOX_PAD_RATIO", 0.02)),
            SAVE_ALL_ANOMALIES=bool(raw.get("SAVE_ALL_ANOMALIES", False)),
            SHOW_ALL_ANOMALY_BOXES=bool(raw.get("SHOW_ALL_ANOMALY_BOXES", True)),

            # Additional Anomalib config
            ANOMALY_AMAP_THRESHOLD=float(raw.get("ANOMALY_AMAP_THRESHOLD", 0.7)),
            REDO_CENTER_CROP=bool(raw.get("REDO_CENTER_CROP", True)),
            CENTER_CROP=int(raw.get("CENTER_CROP", 448)),

            # Data collector tool
            DATA_COLLECTOR_FOLDER=raw.get("DATA_COLLECTOR_FOLDER", "data_collector"),

            # Sync API
            API_SYNC_ENABLE=bool(raw.get("API_SYNC_ENABLE", False)),
            API_SYNC_URL=raw.get("API_SYNC_URL", "http://192.168.0.122:8030//WebApi/QualityControl/VisualInspection/SaveScannedImage"),
            API_TIMEOUT=float(raw.get("API_TIMEOUT", 8.0)),
            PRODUCTION_INSTRUCTION=raw.get("PRODUCTION_INSTRUCTION", ""),
            ITEM_CODE=raw.get("ITEM_CODE", ""),
            OTHER_INFO_DEFAULT=raw.get("OTHER_INFO_DEFAULT", ""),
            
            # Defect Classification API
            DEFECT_CLS_ENABLE=bool(raw.get("DEFECT_CLS_ENABLE", False)),
            DEFECT_CLS_URL=raw.get("DEFECT_CLS_URL", "http://127.0.0.1:8031/search/by-image"),
            DEFECT_CLS_TOPK=int(raw.get("DEFECT_CLS_TOPK", 1)),
            DEFECT_CLS_METRIC=str(raw.get("DEFECT_CLS_METRIC", "cosine")),
            DEFECT_CLS_SIM_THRESHOLD=float(raw.get("DEFECT_CLS_SIM_THRESHOLD", 0.8)),
            
            # === Keypoints Detection ===
            KEYPOINT_DETECTION = bool(raw.get('KEYPOINT_DETECTION', False)),
            KEYPOINTS_MODEL_PATH = raw.get("KEYPOINTS_MODEL_PATH", 'model_checkpoint\bulong_8ly_keypoints.pt'),
            KEYPOINTS_SCORE_THRESHOLD = float(raw.get("KEYPOINTS_SCORE_THRESHOLD", 0.5)),

            # === Automatic Disk Cleanup ===
            DISK_CLEANUP_DAYS = float(raw.get("DISK_CLEANUP_DAYS", 15.0)),
            DISK_CLEANUP_INTERVAL_HOURS = float(raw.get("DISK_CLEANUP_INTERVAL_HOURS", 24.0))
        )

"""
06082026 - KHAI - Move AnomalyConfig from 1_edge_node\packages\ai\tasks\anomaly\__init__.py to 1_edge_node\packages\core\config.py; make it a subclass of BaseConfig
"""
@dataclass
class AnomalyConfig(BaseConfig):
    input_size: int
    score_thres: float
    inside_overlap_min: float
    min_area_ratio: float
    bbox_pad_ratio: float
    save_all: bool
    show_all_boxes: bool
    amap_threshold: float = 0.7
    redo_center_crop: bool = True
    center_crop: int = 448
    class_names: Dict[int, str] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str) -> "AnomalyConfig":
        path = str(Path(path))
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        return AnomalyConfig(
            input_size=int(raw.get("ANOMALY_INPUT_SIZE", 256)),
            score_thres=float(raw.get("ANOMALY_SCORE_THRESHOLD", 0.5)),
            inside_overlap_min=float(raw.get("ANOMALY_INSIDE_OVERLAP_MIN", 0.0)),
            min_area_ratio=float(raw.get("ANOMALY_MIN_AREA_RATIO", 1e-3)),
            bbox_pad_ratio=float(raw.get("ANOMALY_BBOX_PAD_RATIO", 0.02)),
            save_all=bool(raw.get("SAVE_ALL_ANOMALIES"), False),
            show_all_boxes=bool(raw.get("SHOW_ALL_ANOMALY_BOXES", False)),
            amap_threshold=float(raw.get("ANOMALY_AMAP_THRESHOLD", 0.7)),
            redo_center_crop=bool(raw.get("REDO_CENTER_CROP", False)),
            center_crop=bool(raw.get("CENTER_CROP", 448)),
            class_names={
                int(class_id): str(class_name)
                for class_id, class_name in (raw.get("YOLO_CLASS_NAMES") or {}).items()
            },
        )
