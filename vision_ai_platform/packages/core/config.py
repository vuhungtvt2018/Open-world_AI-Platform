from typing import List, Optional, Tuple, Union, Literal, Any
from pydantic import BaseModel, Field, ConfigDict
import cv2
import warnings


class BaseConfig(BaseModel):
    """Base configuration class with common Pydantic settings."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


# ==============================================================================
# YOLO MODEL CONFIG
# ==============================================================================
class YOLOConfig(BaseConfig):
    """Configuration settings for YOLO models."""
    # Common
    task: Literal["detect", "segment", "semantic", "depth", "classify", "pose", "obb"] = Field(
        default="detect", description="YOLO execution task"
    )
    mode: Literal["train", "val", "predict", "export", "track", "benchmark"] = Field(
        default="train", description="YOLO execution mode"
    )
    imgsz: Union[int, List[int], Tuple[int]] = Field(default=640, description="Image size for model")

    # Predictor configuration
    source: Optional[str] = Field(default=None, description="Path/dir/URL/stream for images or videos")
    vid_stride: int = Field(default=1, ge=1, description="Read every Nth frame for video sources")
    stream_buffer: bool = Field(default=False, description="Buffer all frames vs keep most recent frame")
    visualize: bool = Field(default=False, description="Visualize model features")
    augment: bool = Field(default=False, description="Apply test-time augmentation (TTA)")
    agnostic_nms: bool = Field(default=False, description="Class-agnostic NMS")
    classes: Optional[Union[int, List[int]]] = Field(default=None, description="Filter results by class id(s)")
    retina_masks: bool = Field(default=False, description="High-resolution segmentation masks")
    embed: Optional[List[int]] = Field(default=None, description="Return feature embeddings from layer indices")
    conf: Optional[float] = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold for predictions")
    iou: float = Field(default=0.7, ge=0.0, le=1.0, description="IoU threshold for NMS")
    max_det: int = Field(default=300, ge=1, description="Maximum number of detections per image")
    
    # Visualization options
    show: bool = Field(default=False, description="Display images/videos in a window")
    save_frames: bool = Field(default=False, description="Save individual frames from video predictions")
    save_txt: bool = Field(default=False, description="Save results as .txt files")
    save_conf: bool = Field(default=False, description="Save confidence scores with results")
    save_crop: bool = Field(default=False, description="Save cropped prediction regions")
    show_labels: bool = Field(default=True, description="Draw class labels")
    show_conf: bool = Field(default=True, description="Draw confidence values")
    show_boxes: bool = Field(default=True, description="Draw bounding boxes")
    line_width: Optional[int] = Field(default=None, description="Line width of boxes")

    # Training configuration
    model: Optional[str] = Field(default=None, description="Path to model file")
    data: Optional[str] = Field(default=None, description="Path to data config file")
    epochs: int = Field(default=100, ge=1, description="Number of epochs to train for")
    time: Optional[float] = Field(default=None, gt=0.0, description="Max hours to train")
    patience: int = Field(default=100, ge=0, description="Early stopping patience")
    batch: Union[int, float] = Field(default=16, description="Batch size (int) or AutoBatch fraction (float)")
    save: bool = Field(default=True, description="Save checkpoints")
    save_period: int = Field(default=-1, description="Checkpoint save frequency in epochs")
    cache: Union[bool, Literal["ram", "disk"]] = Field(default=False, description="Cache images in RAM or disk")
    device: Optional[Union[int, str, List[Union[int, str]]]] = Field(default=None, description="CUDA/CPU/MPS device specification")
    workers: int = Field(default=8, ge=0, description="Dataloader workers")
    project: Optional[str] = Field(default=None, description="Project name")
    name: Optional[str] = Field(default=None, description="Experiment name")
    exist_ok: bool = Field(default=False, description="Overwrite existing project/name dir")
    pretrained: Union[bool, str] = Field(default=True, description="Use pretrained weights or path to file")
    cls_remap: bool = Field(default=True, description="Remap pretrained classification head rows")
    optimizer: str = Field(default="auto", description="Optimizer choice")
    verbose: bool = Field(default=True, description="Verbose output logging")
    seed: int = Field(default=0, description="Random seed")
    deterministic: bool = Field(default=True, description="Enable deterministic operations")
    single_cls: bool = Field(default=False, description="Treat multi-class dataset as single class")
    rect: bool = Field(default=False, description="Rectangular training/val batches")
    cos_lr: bool = Field(default=False, description="Use Cosine LR scheduler")
    close_mosaic: int = Field(default=10, ge=0, description="Disable mosaic augmentation for final N epochs")
    resume: bool = Field(default=False, description="Resume training from last checkpoint")
    amp: bool = Field(default=True, description="Automatic Mixed Precision (AMP)")
    fraction: float = Field(default=1.0, gt=0.0, le=1.0, description="Dataset fraction to use")
    profile: bool = Field(default=False, description="Profile speeds for loggers")
    freeze: Optional[Union[int, List[int]]] = Field(default=None, description="Freeze first N or specified layer indices")
    multi_scale: float = Field(default=0.0, ge=0.0, description="Multi-scale range fraction")
    compile: Union[bool, str] = Field(default=False, description="Enable torch.compile()")
    channels_last: bool = Field(default=False, description="Use NHWC memory format")

    # Task Specific Gains / Augmentations
    overlap_mask: bool = Field(default=True, description="Overlap masks for instance segmentation")
    mask_ratio: int = Field(default=4, ge=1, description="Mask downsample ratio")
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Classification dropout")
    
    # Loss Gains & Hyperparameters
    lr0: float = Field(default=0.01, gt=0.0, description="Initial learning rate")
    lrf: float = Field(default=0.01, gt=0.0, description="Final learning rate fraction")
    momentum: float = Field(default=0.937, ge=0.0, le=1.0, description="Momentum/beta1")
    weight_decay: float = Field(default=0.0005, ge=0.0, description="Weight decay")
    warmup_epochs: float = Field(default=3.0, ge=0.0, description="Warmup epochs")
    warmup_momentum: float = Field(default=0.8, ge=0.0, le=1.0, description="Warmup initial momentum")
    warmup_bias_lr: float = Field(default=0.1, ge=0.0, description="Warmup bias LR")
    distill_model: Optional[str] = Field(default=None, description="Path to teacher model for distillation")
    dis: float = Field(default=6.0, description="Distillation loss weight")
    box: float = Field(default=7.5, description="Box loss gain")
    cls: float = Field(default=0.5, description="Classification loss gain")
    cls_pw: float = Field(default=0.0, description="Class weights power for class imbalance")
    dfl: float = Field(default=1.5, description="Distribution Focal Loss gain")
    pose: float = Field(default=12.0, description="Pose loss gain")
    kobj: float = Field(default=1.0, description="Keypoint objectness gain")
    rle: float = Field(default=1.0, description="RLE loss gain")
    angle: float = Field(default=1.0, description="Oriented bounding box angle loss gain")
    dlog: float = Field(default=1.0, description="Depth SILog loss gain")
    dgrad: float = Field(default=0.5, description="Depth gradient loss gain")
    dlam: float = Field(default=1.0, description="Depth SILog variance focus")
    nbs: int = Field(default=64, description="Nominal batch size for loss normalization")
    
    # Augmentation Probabilities & Hyperparameters
    hsv_h: float = Field(default=0.015, ge=0.0, le=1.0, description="HSV Hue fraction")
    hsv_s: float = Field(default=0.7, ge=0.0, le=1.0, description="HSV Saturation fraction")
    hsv_v: float = Field(default=0.4, ge=0.0, le=1.0, description="HSV Value fraction")
    degrees: float = Field(default=0.0, description="Rotation degrees")
    translate: float = Field(default=0.1, description="Translation fraction")
    scale: Union[float, Tuple[float, float]] = Field(default=0.5, description="Scale gain (+/-) or explicit min/max tuple")
    shear: float = Field(default=0.0, description="Shear degrees")
    perspective: float = Field(default=0.0, description="Perspective fraction")
    flipud: float = Field(default=0.0, ge=0.0, le=1.0, description="Vertical flip probability")
    fliplr: float = Field(default=0.5, ge=0.0, le=1.0, description="Horizontal flip probability")
    bgr: float = Field(default=0.0, ge=0.0, le=1.0, description="BGR channel swap probability")
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0, description="Mosaic probability")
    mixup: float = Field(default=0.0, ge=0.0, le=1.0, description="MixUp probability")
    cutmix: float = Field(default=0.0, ge=0.0, le=1.0, description="CutMix probability")
    copy_paste: float = Field(default=0.0, ge=0.0, le=1.0, description="Copy-paste probability")
    copy_paste_mode: Literal["flip", "mixup"] = Field(default="flip", description="Copy-paste strategy")
    auto_augment: str = Field(default="randaugment", description="Classification auto-augmentation policy")
    erasing: float = Field(default=0.4, ge=0.0, le=1.0, description="Random erasing probability")

    # Evaluation configuration
    val: bool = Field(default=True, description="Run validation during training")
    split: Literal["val", "test", "train"] = Field(default="val", description="Dataset split to evaluate")
    save_json: bool = Field(default=False, description="Save COCO JSON or PNG masks for external evaluation")
    conf: Optional[float] = Field(default=0.001, ge=0.0, le=1.0, description="Confidence threshold for evaluation")
    quantize: Optional[Union[int, str]] = Field(default=None, description="Precision quantization settings")
    dnn: bool = Field(default=False, description="Use OpenCV DNN for ONNX inference")
    plots: bool = Field(default=True, description="Save plots and images during evaluation")
    end2end: Optional[bool] = Field(default=None, description="Use end2end head (e.g. YOLOv10/YOLO26)")

    # Exporter configuration
    format: Literal[
        "onnx", "torchscript", "engine", "openvino", "coreml", 
        "saved_model", "pb", "tflite", "edgetpu", "tfjs", 
        "paddle", "ncnn", "mnn"
    ] = Field(
        default="onnx", 
        description="Target export format for deployment environment"
    )
    keras: bool = Field(default=False, description="TF SavedModel only (format=saved_model); enable Keras layers during export")
    optimize: bool = Field(default=False, description="DEEPX only; higher compiler optimization (slower compile, faster inference)")
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
    nms: bool = Field(
        default=False, 
        description="Embed Non-Maximum Suppression (NMS) directly into the exported model graph"
    )
    fraction: float = Field(
        default=1.0, 
        description="Fraction of validation dataset to use for INT8 calibration"
    )
    workspace: Optional[float] = Field(
        default=None, 
        description="Maximum workspace memory allocation in GiB for TensorRT optimization"
    )

    # Override configuration
    cfg: Optional[str] = Field(default=None, description="Path to a config.yaml that overrides defaults")

    # Tracker file
    tracker: str = Field(default="tracktrack.yaml", description="Tracker config: botsort.yaml, bytetrack.yaml, ocsort.yaml, deepocsort.yaml, fasttrack.yaml, tracktrack.yaml")


# ==============================================================================
# TRACKER CONFIG
# ==============================================================================
class TrackerConfig(BaseConfig):
    """Configuration settings for object tracking algorithms."""

    # Core Parameters (common across most trackers)
    tracker_type: str = Field(
        default="tracktrack",
        description="Tracker backend: botsort|bytetrack|deepocsort|fasttrack|ocsort|tracktrack"
    )
    track_high_thresh: float = Field(
        default=0.25,
        description="First-stage/high-confidence match threshold"
    )
    track_low_thresh: float = Field(
        default=0.1,
        description="Second-stage threshold for low-score matches"
    )
    new_track_thresh: float = Field(
        default=0.25,
        description="Threshold/minimum score to start a new track"
    )
    match_thresh: float = Field(
        default=0.8,
        description="Association similarity threshold (IoU/cost)"
    )
    track_buffer: int = Field(
        default=30,
        description="Frames to keep lost tracks active"
    )
    fuse_score: bool = Field(
        default=False,
        description="Fuse detection score with motion/IoU for matching"
    )

    # ReID & Global Motion Compensation (GMC)
    gmc_method: str = Field(
        default="sparseOptFlow",
        description="Global motion compensation method: sparseOptFlow|orb|sift|ecc|none"
    )
    with_reid: bool = Field(
        default=False,
        description="Enable ReID model usage for feature matching"
    )
    model: str = Field(
        default="auto",
        description="ReID model path or name ('auto' uses detector features)"
    )
    proximity_thresh: float = Field(
        default=0.5,
        description="Min IoU to consider tracks proximate for ReID"
    )
    appearance_thresh: float = Field(
        default=0.8,
        description="Min appearance similarity threshold for ReID"
    )

    # OC-SORT & Deep OC-SORT Specifics
    delta_t: int = Field(
        default=3,
        description="Temporal window for velocity direction computation in OCM"
    )
    inertia: float = Field(
        default=0.2,
        description="Weight of velocity consistency cost in association"
    )
    use_byte: bool = Field(
        default=False,
        description="Enable ByteTrack-style low-confidence second association pass"
    )
    alpha_fixed_emb: float = Field(
        default=0.95,
        description="Base EMA factor for track embedding updates in Deep OC-SORT"
    )

    # FastTracker Specifics
    reset_velocity_offset_occ: int = Field(
        default=5,
        description="History frames back to restore KF velocity on occlusion onset"
    )
    reset_pos_offset_occ: int = Field(
        default=3,
        description="History frames back to restore KF position on occlusion onset"
    )
    enlarge_bbox_occ: float = Field(
        default=1.1,
        description="One-shot bbox height scale while occluded"
    )
    dampen_motion_occ: float = Field(
        default=0.5,
        description="Velocity dampening factor applied while occluded (0-1)"
    )
    active_occ_to_lost_thresh: int = Field(
        default=10,
        description="Max consecutive occluded frames before marking lost"
    )
    occ_cover_thresh: float = Field(
        default=0.7,
        description="Fraction of track's area covered by another to declare occlusion"
    )
    occ_reappear_window: int = Field(
        default=40,
        description="Frames a recently-occluded lost track stays re-findable"
    )
    init_iou_suppress: float = Field(
        default=0.7,
        description="Suppress new-track init if IoU with any active track >= threshold"
    )

    # TrackTrack Specifics
    lost_match_thr: float = Field(
        default=0.0,
        description="Looser rebind cost gate for still-lost tracks (0 disables)"
    )
    iou_weight: float = Field(
        default=0.5,
        description="Weight for HMIoU distance in cost matrix"
    )
    reid_weight: float = Field(
        default=0.5,
        description="Weight for cosine distance in cost matrix"
    )
    conf_weight: float = Field(
        default=0.1,
        description="Weight for confidence distance in cost matrix"
    )
    angle_weight: float = Field(
        default=0.05,
        description="Weight for corner angle distance in cost matrix"
    )
    penalty_p: float = Field(
        default=0.2,
        description="Cost penalty for low-confidence detections in iterative assignment"
    )
    penalty_q: float = Field(
        default=0.4,
        description="Cost penalty for deleted/recovered detections in iterative assignment"
    )
    reduce_step: float = Field(
        default=0.05,
        description="Threshold reduction step per iteration"
    )
    tai_thr: float = Field(
        default=0.55,
        description="IoU threshold for Track-Aware Initialization (TAI) NMS suppression"
    )
    min_track_len: int = Field(
        default=3,
        description="Minimum history length before track is confirmed"
    )
    device: Optional[Union[int, str, List[Union[int, str]]]] = Field(default=None, description="CUDA/CPU/MPS device specification")


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


class ExporterConfig(BaseConfig):
    """Configuration class for Ultralytics YOLO model export settings.
    
    Inherits strict extra field checking and type flexibilities from BaseConfig.
    """
    name: Optional[str] = Field(default=None, description="Experiment name")
    format: Literal[
        "onnx", "torchscript", "engine", "openvino", "coreml", 
        "saved_model", "pb", "tflite", "edgetpu", "tfjs", 
        "paddle", "ncnn", "mnn"
    ] = Field(
        default="onnx", 
        description="Target export format for deployment environment"
    )
    split: Literal["val", "test", "train"] = Field(default="val", description="Dataset split to evaluate")
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
    conf: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold for predictions")
    max_det: int = Field(default=300, ge=1, description="Maximum number of detections per image")
    agnostic_nms: bool = Field(default=False, description="Class-agnostic NMS")
    iou: float = Field(default=0.7, ge=0.0, le=1.0, description="IoU threshold for NMS")
    end2end: Optional[bool] = Field(default=None, description="Use end2end head (e.g. YOLOv10/YOLO26)")

    def to_ultralytics_dict(self) -> dict:
        """Converts config instance into a clean dictionary for model.export(**kwargs)."""
        return self.model_dump(exclude_none=True)


class WorkflowConfig(BaseConfig):
    """Configuration class for Vision AI workflows"""
    source: Optional[str] = Field(default=None, description="Path to input source (video, stream, etc.)")
    model: Optional[str] = Field(default=None, description="Path to the model weights")
    classes: Optional[List[int]] = Field(default=None, description="Class indices to filter detections")
    show_conf: bool = Field(default=True, description="Show confidence scores on visual output")
    show_labels: bool = Field(default=True, description="Display class labels on visual output")
    show_boxes: bool = Field(default=True, description="Display bounding boxes on visual output")
    region: Optional[List[Tuple[int, int]]] = Field(default=None, description="Polygonal region or line coordinates")
    colormap: Optional[int] = Field(default=cv2.COLORMAP_DEEPGREEN, description="OpenCV colormap constant")
    show_in: bool = Field(default=True, description="Display count for objects entering region")
    show_out: bool = Field(default=True, description="Display count for objects leaving region")
    up_angle: float = Field(default=145.0, description="Upper angle threshold for pose monitoring")
    down_angle: int = Field(default=90, description="Lower angle threshold for pose monitoring")
    kpts: List[int] = Field(default_factory=lambda: [6, 8, 10], description="Keypoint indices to monitor")
    analytics_type: str = Field(default="line", description="Type of analytics chart ('line', 'bar', etc.)")
    figsize: Optional[Tuple[float, float]] = Field(default=(12.8, 7.2), description="Matplotlib figure size")
    blur_ratio: float = Field(default=0.5, ge=0.0, le=1.0, description="Blur ratio (0.0 to 1.0)")
    vision_point: Tuple[int, int] = Field(default=(20, 20), description="Reference point for directional tracking")
    crop_dir: str = Field(default="cropped-detections", description="Directory to save cropped detections")
    json_file: Optional[str] = Field(default=None, description="Path to JSON file for parking regions")
    line_width: int = Field(default=2, ge=1, description="Line width for drawing overlays")
    records: int = Field(default=5, ge=1, description="Threshold count for alerts")
    fps: float = Field(default=30.0, gt=0.0, description="Video frame rate for speed calculation")
    max_hist: int = Field(default=5, ge=1, description="Historical positions retained per track")
    meter_per_pixel: float = Field(default=0.05, gt=0.0, description="Real-world scale (meters per pixel)")
    max_speed: int = Field(default=120, gt=0, description="Speed limit threshold")
    show: bool = Field(default=False, description="Display GUI window during processing")
    iou: float = Field(default=0.7, ge=0.0, le=1.0, description="IoU threshold for NMS")
    conf: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold for predictions")
    device: Optional[str] = Field(default=None, description="Target device ('cpu', '0', etc.)")
    max_det: int = Field(default=300, ge=1, description="Maximum detections per frame")
    quantize: Union[int, str, None] = Field(default=None, description="Quantization precision (e.g. 16 for FP16)")
    imgsz: int = Field(default=640, gt=0, description="Inference image resolution")
    tracker: str = Field(default="botsort.yaml", description="Tracking config YAML path")
    verbose: bool = Field(default=True, description="Enable verbose logging output")
    data: str = Field(default="images", description="Directory path for similarity search or data")

    def update(self, **kwargs: Any) -> "WorkflowConfig":
        """Update configuration parameters with new values provided as keyword arguments."""
        if "half" in kwargs:
            warnings.warn(
                "'half' is deprecated, please use 'quantize' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["quantize"] = 16 if kwargs.pop("half") else None

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"'{key}' is not a valid argument for {self.__class__.__name__}.")

        return self

class AppConfig(BaseConfig):
    """Master application configuration aggregating sub-configs."""
    project_name: str = Field(default="vision_pipeline", description="Project workspace directory name")
    run_name: str = Field(default="exp", description="Experiment run identifier")
    data_path: str = Field(default="coco8.yaml", description="Path to dataset configuration YAML or directory")
    
    # Sub-configurations
    model: YOLOConfig = Field(default_factory=YOLOConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)