from typing import List, Dict, Optional, Union, Literal, Tuple
from pydantic import BaseModel, Field
import yaml


class BaseConfig(BaseModel):
    """Base configuration class with common Pydantic settings."""
    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


class DatasetConfig(BaseConfig):
    """Configuration for dataset"""
    data_path: str = Field(description="Path to dataset")
    dataset_type: str = Field(
        description=(
            "Dataset type: 'yolo', 'depth', 'yolo-multimodal', "
            "'grounding', 'yolo-concat', 'semantic', "
            "'polygon-semantic', 'classification'"
        )
    )


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
    save_dir: str = Field(default="run", description="Directory path to save training results")
    epochs: int = Field(default=100, ge=1, description="Number of training epochs")
    save_period: int = Field(default=10, ge=1, description="Save checkpoint every N epochs")
    amp: bool = Field(default=True, description="Automatic Mixed Precision (AMP) training")
    batch_size: int = Field(default=16, ge=-1, alias="batch", description="Batch size (-1 for AutoBatch)")
    resume: bool = Field(default=False, description="Resume training from last checkpoint in the run dir")
    optimizer: str = Field(
        default="auto", 
        description="Optimizer choice: 'SGD', 'MuSGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', or 'auto'"
    )
    lr0: float = Field(default=0.01, gt=0.0, description="Initial learning rate")
    lrf: float = Field(default=0.01, gt=0.0, description="Final learning rate fraction (lr0 * lrf)")
    alpha: float = Field(default=0.99, gt=0.0, description="RMSProp alpha")
    beta1: float = Field(default=0.9, gt=0.0, description="Adam/AdamW beta1")
    beta2: float = Field(default=0.99, gt=0.0, description="Adam/AdamW beta2")
    momentum: float = Field(default=0.937, ge=0.0, le=1.0, description="SGD momentum")
    weight_decay: float = Field(default=0.0005, ge=0.0, description="Optimizer weight decay")
    eps: float = Field(default=1e-8, ge=0.0, description="Epsilon to avoid division by 0")
    warmup_epochs: float = Field(default=3.0, ge=0.0, description="Warmup epochs")
    warmup_bias_lr: float = Field(default=0.1, ge=0.0, description="Bias learning rate during warmup")
    patience: int = Field(default=50, ge=0, description="Early stopping patience (epochs without improvement)")
    cos_lr: bool = Field(default=False, description="Use cosine learning rate scheduler")
    
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
    save_txt: bool = Field(default=False, description="Save results as .txt files (xywh format)")
    save_conf: bool = Field(default=False, description="Save confidence scores with results")
    plots: bool = Field(default=True, description="Save plots and charts during evaluation")
    visualize: bool = Field(default=True, description="Save images during evaluation")
    rect: bool = Field(default=False, description="Use rectangular testing for faster inference")
    save_dir: str = Field(default="results", description="Directory to save evaluation results")
    show_labels: bool = Field(default=True, description="Whether to display class labels in the visualization")
    show_conf: bool = Field(default=True, description="Whether to display confidence values in the visualization")
    task: str = Field(default="detect", description="Ultralytics task, values: detect, classify, semantic, segment, obb, pose")
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0, alias="conf", description="Object confidence threshold for detection")
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0, alias="iou", description="Intersection Over Union (IoU) threshold for NMS")
    max_det: int = Field(default=300, ge=1, description="Maximum number of detections per image")
    classes: Optional[List[int]] = Field(default=None, description="Filter results by class IDs, e.g. [0, 2, 3]")
    agnostic_nms: bool = Field(default=False, description="Class-agnostic NMS")
    single_cls: bool = Field(default=False, description="If True, single class training is used.")


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


def get_config_from_yaml(config_file: str, config_type: str):
    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    if config_type == "model":
        return ModelConfig.model_validate(config_dict)
    elif config_type == "predictor":
        return PredictorConfig.model_validate(config_dict)
    elif config_type == "trainer":
        return TrainerConfig.model_validate(config_dict)
    elif config_type == "evaluator":
        return EvaluatorConfig.model_validate(config_dict)
    elif config_type == "tracker":
        return TrackerConfig.model_validate(config_dict)
    elif config_type == "exporter":
        return ExporterConfig.model_validate(config_dict)
    elif config_type == "dataset":
        return DatasetConfig.model_validate(config_dict)
    elif config_type == "app":
        return AppConfig.model_validate(config_dict)
    else:
        raise ValueError(
            "Value of config_type must be in the following list: "
            "['model', 'predictor', 'trainer', 'evaluator', 'tracker', 'exporter', 'dataset', 'app']"
        )