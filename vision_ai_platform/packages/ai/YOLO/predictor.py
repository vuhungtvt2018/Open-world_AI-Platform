from vision_ai_platform.packages.core.model import BasePredictor
from vision_ai_platform.packages.core.config import YOLOConfig
from vision_ai_platform.packages.core.results import Results
from vision_ai_platform.packages.utils import LOGGER
import vision_ai_platform.packages.utils.ops as ops
from vision_ai_platform.packages.utils.check import check_imgsz
from vision_ai_platform.packages.utils.device_utils import attempt_compile
from vision_ai_platform.packages.utils.nms import non_max_suppression
from vision_ai_platform.packages.ai.nn.autobackend import AutoBackend

from typing import Any, List, Optional, Union, Tuple, Dict, Callable
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import cv2


def letterbox(
    img: np.ndarray,
    new_shape: Union[int, List[int, int], Tuple[int, int]] = (640, 640),
    color: tuple = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Resize and pad image while meeting stride-multiple constraints (Ultralytics style)."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list):
        new_shape = tuple(new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border

    return img, ratio, (dw, dh)


def scale_boxes(
    img1_shape: tuple, boxes: torch.Tensor, img0_shape: tuple, ratio_pad=None
) -> torch.Tensor:
    """Rescale bounding boxes (xyxy) from letterboxed img1_shape back to original img0_shape."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    
    # Clip boxes to image bounds
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img0_shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img0_shape[0])  # y1, y2
    return boxes


class YOLOPredictor(BasePredictor):
    def __init__(
        self,
        cfg: YOLOConfig,
        save_dir: str | Path,
        task: str,
        model: Optional[nn.Module] = None,
        _callbacks: Optional[Dict[str, List[Callable]]] = None,
    ) -> None:
        super().__init__(cfg, save_dir, task, _callbacks)
        self._meta: dict[str, Any] = {}
    
    def setup_model(self, model: Union[str, Path, torch.nn.Module, None] = None, verbose: bool = True):
        """Initialize backend model and prepare for evaluation mode."""
        model_path = model or getattr(self.cfg, "model", "yolov8n.pt")
        
        if hasattr(model_path, "end2end"):
            if getattr(self.cfg, "end2end", None) is not None:
                model_path.end2end = getattr(self.cfg, "end2end")
            if model_path.end2end:
                model_path.set_head_attr(
                    max_det=max(self.cfg.max_det, 300), 
                    agnostic_nms=self.cfg.agnostic_nms
                )

        self.model = AutoBackend(
            model=model_path,
            device=getattr(self.cfg, "device", None),
            dnn=getattr(self.cfg, "dnn", False),
            data=self.data,
            fp16=getattr(self.cfg, "quantize", None) == 16,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.cfg.imgsz = self.model.imgsz

        self.model.eval()

        channels_last = (
            getattr(self.cfg, "channels_last", False)
            and self.device.type == "cuda"
            and self.model.format == "pt"
        )
        if channels_last:
            self.model.to(memory_format=torch.channels_last)

        self.model = attempt_compile(
            self.model, 
            device=self.device, 
            mode=getattr(self.cfg, "compile", None)
        )

    def setup_source(self, source: Any, stride: Optional[int] = None):
        """Configure video/image dataloader source."""
        stride_val = stride or getattr(self.model, "stride", 32)
        target_imgsz = getattr(self.cfg, "imgsz", 640)
        self.imgsz = check_imgsz(target_imgsz, stride=stride_val, min_dim=2)

        self.dataset = load_inference_source(
            source=source,
            batch=getattr(self.cfg, "batch", 1),
            vid_stride=self.cfg.vid_stride,
            buffer=self.cfg.stream_buffer,
            channels=getattr(self.model, "channels", 3),
        )
        self.source_type = self.dataset.source_type
        self.vid_writer = {}

    def postprocess(self, preds, img, orig_imgs, **kwargs) -> List[Results]:
        """Applies NMS and scales detection bounding boxes back to the original image dimensions.

        Args:
            preds: Raw model output tensor.
            orig_imgs: Original raw input passed into `predict`.

        Returns:
            List[Results]: List of Results objects per image.
        """
        save_feats = getattr(self, "_feats", None) is not None
        preds = non_max_suppression(
            preds,
            self.cfg.conf,
            kwargs.pop("iou", self.cfg.iou),  # allow callers (e.g. TrackTrack loose-NMS recovery) to override IoU
            self.cfg.classes,
            self.cfg.agnostic_nms,
            max_det=self.cfg.max_det,
            nc=0 if self.cfg.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.cfg.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    def construct_results(self, preds, img, orig_imgs):
        """Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])