from typing import List, Dict, Optional, Union, Literal, Tuple
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration class with common Pydantic settings."""
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


class ModelConfig(BaseConfig):
    """Configuration for model architecture and weights."""
    model_path: str = Field(
        default="yolov8n.pt", 
        description="Path to model weights file or pretrained model identifier (e.g., 'yolov8n.pt', 'yolov8x.yaml')"
    )
    task: str = Field(
        default="detect", 
        description="Task type: 'detect', 'segment', 'classify', 'pose', 'obb'"
    )
    num_classes: Optional[int] = Field(
        default=None, 
        description="Number of target classes (overrides model default if set)"
    )
    imgsz: Union[int, List[int]] = Field(
        default=640, 
        description="Input image size as integer (640) or list [height, width]"
    )


class HardwareConfig(BaseConfig):
    """Configuration for computing resources."""
    device: str = Field(
        default="0", 
        description="CUDA device(s) e.g. '0', '0,1,2,3', 'cpu', or 'mps'"
    )
    workers: int = Field(
        default=8, 
        ge=0, 
        description="Number of worker threads for dataloading"
    )
    half: bool = Field(
        default=False, 
        description="Use FP16 half-precision inference"
    )


class PredictorConfig(BaseConfig):
    """Configuration for inference / prediction."""
    conf_threshold: float = Field(
        default=0.25, 
        ge=0.0, 
        le=1.0, 
        alias="conf",
        description="Object confidence threshold for detection"
    )
    iou_threshold: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0, 
        alias="iou",
        description="Intersection Over Union (IoU) threshold for NMS"
    )
    max_det: int = Field(
        default=300, 
        ge=1, 
        description="Maximum number of detections per image"
    )
    classes: Optional[List[int]] = Field(
        default=None, 
        description="Filter results by class IDs, e.g. [0, 2, 3]"
    )
    agnostic_nms: bool = Field(
        default=False, 
        description="Class-agnostic NMS"
    )
    save_txt: bool = Field(
        default=False, 
        description="Save results to a text file"
    )
    save_conf: bool = Field(
        default=False, 
        description="Save confidences in exported text results"
    )


class TrainerConfig(BaseConfig):
    """Configuration for training models."""
    epochs: int = Field(default=100, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=16, ge=-1, alias="batch", description="Batch size (-1 for AutoBatch)")
    optimizer: str = Field(
        default="auto", 
        description="Optimizer choice: 'SGD', 'Adam', 'AdamW', 'RMSProp', or 'auto'"
    )
    lr0: float = Field(default=0.01, gt=0.0, description="Initial learning rate")
    lrf: float = Field(default=0.01, gt=0.0, description="Final learning rate fraction (lr0 * lrf)")
    momentum: float = Field(default=0.937, ge=0.0, le=1.0, description="SGD momentum/Adam beta1")
    weight_decay: float = Field(default=0.0005, ge=0.0, description="Optimizer weight decay")
    warmup_epochs: float = Field(default=3.0, ge=0.0, description="Warmup epochs")
    patience: int = Field(default=50, ge=0, description="Early stopping patience (epochs without improvement)")
    
    # Common Data Augmentations (Ultralytics defaults)
    hsv_h: float = Field(default=0.015, ge=0.0, le=1.0, description="HSV-Hue augmentation fraction")
    hsv_s: float = Field(default=0.7, ge=0.0, le=1.0, description="HSV-Saturation augmentation fraction")
    hsv_v: float = Field(default=0.4, ge=0.0, le=1.0, description="HSV-Value augmentation fraction")
    degrees: float = Field(default=0.0, description="Image rotation (+/- deg)")
    translate: float = Field(default=0.1, ge=0.0, le=1.0, description="Image translation (+/- fraction)")
    scale: float = Field(default=0.5, ge=0.0, description="Image scale (+/- gain)")
    fliplr: float = Field(default=0.5, ge=0.0, le=1.0, description="Image flip left-right probability")
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0, description="Image mosaic probability")


class EvaluatorConfig(BaseConfig):
    """Configuration for evaluation / validation metrics."""
    split: str = Field(default="val", description="Dataset split to evaluate on ('val', 'test')")
    save_json: bool = Field(default=False, description="Save results to JSON file for COCO evaluation")
    plots: bool = Field(default=True, description="Save plots and charts during evaluation")
    rect: bool = Field(default=False, description="Use rectangular testing for faster inference")


class TrackerConfig(BaseConfig):
    """Configuration for multi-object tracking (e.g., ByteTrack / BotSORT)."""
    tracker_type: str = Field(
        default="bytetrack", 
        description="Tracking algorithm: 'bytetrack' or 'botsort'"
    )
    track_high_thresh: float = Field(
        default=0.5, 
        description="Threshold for first association step in ByteTrack"
    )
    track_low_thresh: float = Field(
        default=0.1, 
        description="Threshold for second association step"
    )
    new_track_thresh: float = Field(
        default=0.6, 
        description="Threshold to initiate a new track"
    )
    track_buffer: int = Field(
        default=30, 
        description="Frames to keep lost tracks active"
    )
    match_thresh: float = Field(
        default=0.8, 
        description="Matching threshold for data association"
    )


class ExporterConfig(BaseConfig):
    """Configuration class for Ultralytics YOLO model export settings.
    
    Inherits strict extra field checking and type flexibilities from BaseConfig.
    """
    format: Literal[
        "onnx", "torchscript", "engine", "openvino", "coreml", 
        "saved_model", "pb", "tflite", "edgetpu", "tfjs", 
        "paddle", "ncnn", "mnn"
    ] = Field(
        default="onnx", 
        description="Target export format for deployment environment"
    )
    imgsz: Union[int, Tuple[int, int], List[int]] = Field(
        default=640, 
        description="Target image size for model input (e.g., 640 or (640, 480))"
    )
    quantize: Optional[Union[int, str]] = Field(
        default=None, 
        description="Quantization precision: 16 (FP16), 8/'int8' (INT8/PTQ), or None for FP32"
    )
    dynamic: bool = Field(
        default=False, 
        description="Enable dynamic input shape dimensions for formats like ONNX/TensorRT"
    )
    simplify: bool = Field(
        default=True, 
        description="Simplify model graph using tools like onnxslim"
    )
    opset: Optional[int] = Field(
        default=None, 
        description="ONNX opset version (uses latest supported by system if None)"
    )
    batch: int = Field(
        default=1, 
        description="Exported model batch size for inference"
    )
    nms: bool = Field(
        default=False, 
        description="Embed Non-Maximum Suppression (NMS) directly into the exported model graph"
    )
    device: Optional[Union[int, str]] = Field(
        default=None, 
        description="Device for export execution (e.g., 'cpu', 0, 'cuda:0')"
    )
    data: Optional[str] = Field(
        default=None, 
        description="Path to dataset YAML file (required for INT8 quantization calibration)"
    )
    fraction: float = Field(
        default=1.0, 
        description="Fraction of validation dataset to use for INT8 calibration"
    )
    workspace: Optional[float] = Field(
        default=None, 
        description="Maximum workspace memory allocation in GiB for TensorRT optimization"
    )

    def to_ultralytics_dict(self) -> dict:
        """Converts config instance into a clean dictionary for model.export(**kwargs)."""
        return self.model_dump(exclude_none=True)


class AppConfig(BaseConfig):
    """Master application configuration aggregating sub-configs."""
    project_name: str = Field(default="vision_pipeline", description="Project workspace directory name")
    run_name: str = Field(default="exp", description="Experiment run identifier")
    data_path: str = Field(default="coco8.yaml", description="Path to dataset configuration YAML or directory")
    
    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)