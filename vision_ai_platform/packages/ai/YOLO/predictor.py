from vision_ai_platform.packages.core.model import BasePredictor
from vision_ai_platform.packages.core.config import PredictorConfig
from vision_ai_platform.packages.core.results import Results

from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import cv2


def letterbox(
    img: np.ndarray,
    new_shape: tuple = (640, 640),
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


def torchvision_nms(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """Non-Maximum Suppression"""
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)
    except ImportError:
        # Fallback torch implementation
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            i = order[0].item()
            keep.append(i)

            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])

            w = torch.maximum(torch.tensor(0.0, device=boxes.device), xx2 - xx1)
            h = torch.maximum(torch.tensor(0.0, device=boxes.device), yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= iou_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    max_det: int = 300,
) -> List[torch.Tensor]:
    """Applies Non-Maximum Suppression (NMS) on inference results.

    Args:
        prediction: Tensor of shape (batch, 4 + num_classes, num_anchors) or (batch, num_anchors, 4 + num_classes)
        conf_thres: Confidence threshold.
        iou_thres: IoU threshold for NMS.
        classes: Filter results by class indices.
        agnostic: Class-agnostic NMS.
        max_det: Maximum detections per image.

    Returns:
        List of Tensors (n, 6) containing [x1, y1, x2, y2, conf, class_id] per batch image.
    """
    # Transpose to (batch, num_anchors, 4 + num_classes) if formatted as (batch, 4 + num_classes, num_anchors)
    if prediction.shape[1] < prediction.shape[2]:
        prediction = prediction.transpose(1, 2)

    bs = prediction.shape[0]
    nc = prediction.shape[2] - 4  # number of classes
    output = [torch.zeros((0, 6), device=prediction.device)] * bs

    for idx, x in enumerate(prediction):
        # x shape: (num_anchors, 4 + num_classes)
        boxes = x[:, :4]
        scores, labels = x[:, 4:].max(1)

        # Filter by confidence threshold
        mask = scores > conf_thres
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        if not boxes.shape[0]:
            continue

        # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        xyxy = torch.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # Filter by class ID
        if classes is not None:
            class_mask = (labels.unsqueeze(1) == torch.tensor(classes, device=labels.device)).any(1)
            xyxy, scores, labels = xyxy[class_mask], scores[class_mask], labels[class_mask]

        if not xyxy.shape[0]:
            continue

        # Class offset for non-agnostic NMS
        offsets = labels.float() * (0 if agnostic else 4096.0)
        boxes_for_nms = xyxy + offsets.unsqueeze(1)

        keep = torchvision_nms(boxes_for_nms, scores, iou_thres)
        keep = keep[:max_det]

        detections = torch.cat([xyxy[keep], scores[keep].unsqueeze(1), labels[keep].float().unsqueeze(1)], dim=1)
        output[idx] = detections

    return output


class YOLOPredictor(BasePredictor):
    def __init__(
        self,
        cfg: PredictorConfig,
        model: Optional[nn.Module] = None,
        imgsz: tuple[int, int] = (640, 640),
    ) -> None:
        super().__init__(cfg, model)
        self.imgsz = imgsz
        self._meta: dict[str, Any] = {}
    
    def preprocess(self, source: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Loads and formats the input image into a normalized CHW PyTorch tensor.

        Args:
            source: Image file path, numpy array (BGR/RGB), or PyTorch tensor.

        Returns:
            torch.Tensor: Preprocessed tensor of shape (1, 3, H, W) normalized to [0.0, 1.0].
        """
        # 1. Load image to numpy array (BGR)
        if isinstance(source, str):
            img0 = cv2.imread(source)
            if img0 is None:
                raise ValueError(f"Could not load image from path: {source}")
        elif isinstance(source, np.ndarray):
            img0 = source.copy()
        elif isinstance(source, torch.Tensor):
            img0 = source.cpu().numpy()
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # Store original image dimensions and metadata for scaling during post-processing
        self._meta["orig_shape"] = img0.shape[:2]  # (height, width)

        # 2. Apply letterboxing (padding and scaling)
        img, ratio, pad = letterbox(img0, new_shape=self.imgsz, auto=False)
        self._meta["ratio_pad"] = (ratio, pad)
        self._meta["padded_shape"] = img.shape[:2]

        # 3. Convert BGR to RGB, HWC to CHW
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        img = np.ascontiguousarray(img)

        # 4. Cast to float tensor and normalize to [0, 1]
        tensor_img = torch.from_numpy(img).to(self.device).float()
        tensor_img /= 255.0

        # 5. Add batch dimension -> (1, C, H, W)
        if tensor_img.ndimension() == 3:
            tensor_img = tensor_img.unsqueeze(0)

        return tensor_img

    def postprocess(self, preds: Union[torch.Tensor, tuple, list], orig_imgs: Any) -> List[Results]:
        """Applies NMS and scales detection bounding boxes back to the original image dimensions.

        Args:
            preds: Raw model output tensor.
            orig_imgs: Original raw input passed into `predict`.

        Returns:
            List[Results]: List of detections per image. Each item is a numpy array of 
                           shape (N, 6) with columns: [x1, y1, x2, y2, confidence, class_id].
        """
        if preds is None:
            return []

        # Unpack tuple output if model returns loss or extra head outputs
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        # 1. Non-Maximum Suppression
        det_results = non_max_suppression(
            prediction=preds,
            conf_thres=self.cfg.conf_threshold,
            iou_thres=self.cfg.iou_threshold,
            classes=self.cfg.classes,
            agnostic=self.cfg.agnostic_nms,
            max_det=self.cfg.max_det,
        )

        formatted_results = []
        orig_shape = self._meta.get("orig_shape")
        padded_shape = self._meta.get("padded_shape", self.imgsz)
        ratio_pad = self._meta.get("ratio_pad")

        for det in det_results:
            if len(det):
                # 2. Rescale bboxes from letterboxed canvas back to original image shape
                det[:, :4] = scale_boxes(
                    img1_shape=padded_shape,
                    boxes=det[:, :4],
                    img0_shape=orig_shape,
                    ratio_pad=ratio_pad,
                )
                formatted_results.append(det.cpu().numpy())
            else:
                formatted_results.append(np.empty((0, 6), dtype=np.float32))

        return formatted_results

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
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])