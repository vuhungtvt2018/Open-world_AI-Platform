from __future__ import annotations

from functools import cached_property
from typing import Any
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import cv2

from .base import DataExportMixin, SimpleClass

def clip_boxes(boxes, shape):
    """Clip bounding boxes to image boundaries.

    Args:
        boxes (torch.Tensor | np.ndarray): Bounding boxes to clip.
        shape (tuple): Image shape as HWC or HW (supports both).

    Returns:
        (torch.Tensor | np.ndarray): Clipped bounding boxes.
    """
    h, w = shape[:2]  # supports both HWC or HW shapes
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0] = boxes[..., 0].clamp(0, w)
        boxes[..., 1] = boxes[..., 1].clamp(0, h)
        boxes[..., 2] = boxes[..., 2].clamp(0, w)
        boxes[..., 3] = boxes[..., 3].clamp(0, h)
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)  # y1, y2
    return boxes

def clip_coords(coords: torch.Tensor | np.ndarray, shape: tuple[int, ...]):
    """Clip line coordinates to image boundaries.

    Args:
        coords (torch.Tensor | np.ndarray): Line coordinates to clip.
        shape (tuple): Image shape as HWC or HW (supports both).

    Returns:
        (torch.Tensor | np.ndarray): Clipped coordinates.
    """
    h, w = shape[:2]  # supports both HWC or HW shapes
    if isinstance(coords, torch.Tensor):
        coords[..., 0] = coords[..., 0].clamp(0, w)
        coords[..., 1] = coords[..., 1].clamp(0, h)
    else:  # np.array
        coords[..., 0] = coords[..., 0].clip(0, w)  # x
        coords[..., 1] = coords[..., 1].clip(0, h)  # y
    return coords

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize: bool = False, padding: bool = True):
    """Rescale segment coordinates from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): Source image shape as HWC or HW (supports both).
        coords (torch.Tensor): Coordinates to scale with shape (N, 2).
        img0_shape (tuple): Image 0 shape as HWC or HW (supports both).
        ratio_pad (tuple, optional): Ratio and padding values as ((ratio_h, ratio_w), (pad_w, pad_h)).
        normalize (bool): Whether to normalize coordinates to range [0, 1].
        padding (bool): Whether coordinates are based on YOLO-style augmented images with padding.

    Returns:
        (torch.Tensor): Scaled coordinates.
    """
    img0_h, img0_w = img0_shape[:2]  # supports both HWC or HW shapes
    if ratio_pad is None:  # calculate from img0_shape
        img1_h, img1_w = img1_shape[:2]  # supports both HWC or HW shapes
        gain = min(img1_h / img0_h, img1_w / img0_w)  # gain  = old / new
        pad = round((img1_w - round(img0_w * gain)) / 2 - 0.1), round((img1_h - round(img0_h * gain)) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_w  # width
        coords[..., 1] /= img0_h  # height
    return coords

def merge_multi_segment(segments: list[list]):
    """Merge multiple segments into one list by connecting the coordinates with the minimum distance between each
    segment.

    This function connects these coordinates with a thin line to merge all segments into one.

    Args:
        segments (list[list]): Original segmentations in COCO's JSON file. Each element is a list of coordinates, like
            [segmentation1, segmentation2,...].

    Returns:
        (list[np.ndarray]): A list of connected segments represented as NumPy arrays.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        dis = ((segments[i - 1][:, None, :] - segments[i][None, :, :]) ** 2).sum(-1)
        idx1, idx2 = np.unravel_index(np.argmin(dis, axis=None), dis.shape)
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def masks2segments(masks: np.ndarray | torch.Tensor, strategy: str = "all") -> list[np.ndarray]:
    """Convert masks to segments using contour detection.

    Args:
        masks (np.ndarray | torch.Tensor): Binary masks with shape (N, H, W).
        strategy (str): Segmentation strategy, either 'all' or 'largest'.

    Returns:
        (list): List of segment masks as float32 arrays.
    """
    masks = masks.astype("uint8") if isinstance(masks, np.ndarray) else masks.byte().cpu().numpy()
    segments = []
    for x in np.ascontiguousarray(masks):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "all":  # merge and concatenate all segments
                c = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in c]))
                    if len(c) > 1
                    else c[0].reshape(-1, 2)
                )
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments

class BaseResultItem:
    """Base tensor class with additional methods for easy manipulation and device handling.

    This class provides a foundation for tensor-like objects with device management capabilities, supporting both
    PyTorch tensors and NumPy arrays. It includes methods for moving data between devices and converting between tensor
    types.
    """
    def __init__(self, data: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize BaseResultItem with prediction data and the original shape of the image.

        Args:
            data (torch.Tensor | np.ndarray): Prediction data such as bounding boxes, masks, or keypoints.
            orig_shape (tuple[int, int]): Original shape of the image in (height, width) format.
        """
        assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be torch.Tensor or np.ndarray"
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying data tensor.

        Returns:
            (tuple[int, ...]): The shape of the data tensor.

        Examples:
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseResultItem(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape

    def cpu(self):
        """Return a copy of the tensor stored in CPU memory.

        Returns:
            (BaseResultItem): A new BaseResultItem object with the data tensor moved to CPU memory.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseResultItem(data, orig_shape=(720, 1280))
            >>> cpu_tensor = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseResultItem)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """Return a copy of this object with its data converted to a NumPy array.

        Returns:
            (BaseResultItem): A new instance with `data` as a NumPy array.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape = (720, 1280)
            >>> base_tensor = BaseResultItem(data, orig_shape)
            >>> numpy_tensor = base_tensor.numpy()
            >>> print(type(numpy_tensor.data))
            <class 'numpy.ndarray'>
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """Move the tensor to GPU memory.

        Returns:
            (BaseResultItem): A new BaseResultItem instance with the data moved to GPU memory.

        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import BaseResultItem
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseResultItem(data, orig_shape=(720, 1280))
            >>> gpu_tensor = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Return a copy of the tensor with the specified device and dtype.

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

        Returns:
            (BaseResultItem): A new BaseResultItem instance with the data moved to the specified device and/or dtype.

        Examples:
            >>> base_tensor = BaseResultItem(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self) -> int:
        """Return the length of the underlying data tensor.

        Returns:
            (int): The number of elements in the first dimension of the data tensor.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseResultItem(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Return a new BaseResultItem instance containing the specified indexed elements of the data tensor.

        Args:
            idx (int | list[int] | torch.Tensor): Index or indices to select from the data tensor.

        Returns:
            (BaseResultItem): A new BaseResultItem instance containing the indexed data.

        Examples:
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseResultItem(data, orig_shape=(720, 1280))
            >>> result = base_tensor[0]  # Select the first row
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx], self.orig_shape)

class SemanticMask(BaseResultItem):
    """Semantic segmentation class map for one image."""

    def __len__(self) -> int:
        """Return one semantic segmentation result per image."""
        return 1


class DepthMap(BaseResultItem):
    """Per-pixel depth map (meters) for one image, shape (H, W)."""

    def __len__(self) -> int:
        """Return one depth map per image."""
        return 1

class BoundingBoxes(BaseResultItem):
    """A class for managing and manipulating detection boxes.

    This class provides comprehensive functionality for handling detection boxes, including their coordinates,
    confidence scores, class labels, and optional tracking IDs. It supports various box formats and offers methods for
    easy manipulation and conversion between different coordinate systems.

    Attributes:
        data (torch.Tensor | np.ndarray): The raw tensor containing detection boxes and associated data.
        orig_shape (tuple[int, int]): The original image dimensions (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xyxy (torch.Tensor | np.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | np.ndarray): Confidence scores for each box.
        cls (torch.Tensor | np.ndarray): Class labels for each box.
        id (torch.Tensor | None): Tracking IDs for each box (if available).
        xywh (torch.Tensor | np.ndarray): Boxes in [x, y, width, height] format.
        xyxyn (torch.Tensor | np.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
        xywhn (torch.Tensor | np.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.

    Methods:
        cpu: Return a copy of the object with all tensors on CPU memory.
        numpy: Return a copy of the object with all tensors as numpy arrays.
        cuda: Return a copy of the object with all tensors on GPU memory.
        to: Return a copy of the object with tensors on specified device and dtype.

    Examples:
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # height, width
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
    """
    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]):
        """Initialize the Boxes class with detection box data and the original image shape.

        This class manages detection boxes, providing easy access and manipulation of box coordinates, confidence
        scores, class identifiers, and optional tracking IDs. It supports multiple formats for box coordinates,
        including both absolute and normalized forms.

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array with detection boxes of shape (num_boxes, 6) or
                (num_boxes, 7). Columns should contain [x1, y1, x2, y2, (optional) track_id, confidence, class].
            orig_shape (tuple[int, int]): The original image shape as (height, width). Used for normalization.
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self) -> torch.Tensor | np.ndarray:
        """Return bounding boxes in [x1, y1, x2, y2] format.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or numpy array of shape (n, 4) containing bounding box coordinates in
                [x1, y1, x2, y2] format, where n is the number of boxes.
        """
        return self.data[:, :4]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        """Return the confidence scores for each detection box.

        Returns:
            (torch.Tensor | np.ndarray): A 1D tensor or array containing confidence scores for each detection, with
                shape (N,) where N is the number of detections.
        """
        return self.data[:, -2]

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        """Return the class ID tensor representing category predictions for each bounding box.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or numpy array containing the class IDs for each detection box. The
                shape is (N,), where N is the number of boxes.
        """
        return self.data[:, -1]

    @property
    def id(self) -> torch.Tensor | np.ndarray:
        """Return the tracking IDs for each detection box if available.

        Returns:
            (torch.Tensor | np.ndarray | None): A tensor or array containing tracking IDs for each box if tracking is
                enabled, otherwise None. Shape is (N,) where N is the number of boxes.
        """
        return self.data[:, -3] if self.is_track else None

    @cached_property
    def xywh(self) -> torch.Tensor | np.ndarray:
        """Convert bounding boxes from [x1, y1, x2, y2] format to [x, y, width, height] format.

        Returns:
            (torch.Tensor | np.ndarray): Boxes in [x_center, y_center, width, height] format, where x_center, y_center
                are the coordinates of the center point of the bounding box, width, height are the dimensions of the
                bounding box and the shape of the returned tensor is (N, 4), where N is the number of boxes.
        """
        xyxy = self.xyxy
        assert xyxy.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {xyxy.shape}"
        if isinstance(xyxy, torch.Tensor):
            y = torch.empty_like(xyxy)
        else:
            y = np.empty_like(xyxy)
        x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
        y[..., 0] = (x1 + x2) / 2  # x center
        y[..., 1] = (y1 + y2) / 2  # y center
        y[..., 2] = x2 - x1  # width
        y[..., 3] = y2 - y1  # height
        return y

    @cached_property
    def xyxyn(self) -> torch.Tensor | np.ndarray:
        """Return normalized bounding box coordinates relative to the original image size.

        This property calculates and returns the bounding box coordinates in [x1, y1, x2, y2] format, normalized to the
        range [0, 1] based on the original image dimensions.

        Returns:
            (torch.Tensor | np.ndarray): Normalized bounding box coordinates with shape (N, 4), where N is the number of
                boxes. Each row contains [x1, y1, x2, y2] values normalized to [0, 1].
        """
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @cached_property
    def xywhn(self) -> torch.Tensor | np.ndarray:
        """Return normalized bounding boxes in [x, y, width, height] format.

        This property calculates and returns the normalized bounding box coordinates in the format [x_center, y_center,
        width, height], where all values are relative to the original image dimensions.

        Returns:
            (torch.Tensor | np.ndarray): Normalized bounding boxes with shape (N, 4), where N is the number of boxes.
                Each row contains [x_center, y_center, width, height] values normalized to [0, 1] based on the original
                image dimensions.

        Examples:
            >>> boxes = Boxes(torch.tensor([[100, 50, 150, 100, 0.9, 0]]), orig_shape=(480, 640))
            >>> normalized = boxes.xywhn
            >>> print(normalized)
            tensor([[0.1953, 0.1562, 0.0781, 0.1042]])
        """
        xywh = self.xywh
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh

class Masks(BaseResultItem):
    """A class for storing and manipulating detection masks.

    This class extends BaseTensor and provides functionality for handling segmentation masks, including methods for
    converting between pixel and normalized coordinates.

    Attributes:
        data (torch.Tensor | np.ndarray): The raw tensor or array containing mask data.
        orig_shape (tuple[int, int]): Original image shape in (height, width) format.
        xy (list[np.ndarray]): A list of segments in pixel coordinates.
        xyn (list[np.ndarray]): A list of normalized segments.

    Methods:
        cpu: Return a copy of the Masks object with the mask tensor on CPU memory.
        numpy: Return a copy of the Masks object with the mask tensor as a numpy array.
        cuda: Return a copy of the Masks object with the mask tensor on GPU memory.
        to: Return a copy of the Masks object with the mask tensor on specified device and dtype.

    Examples:
        >>> masks_data = torch.rand(1, 160, 160)
        >>> orig_shape = (720, 1280)
        >>> masks = Masks(masks_data, orig_shape)
        >>> pixel_coords = masks.xy
        >>> normalized_coords = masks.xyn
    """

    def __init__(self, masks: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        """Initialize the Masks class with detection mask data and the original image shape.

        Args:
            masks (torch.Tensor | np.ndarray): Detection masks with shape (num_masks, height, width).
            orig_shape (tuple[int, int]): The original image shape as (height, width). Used for normalization.
        """
        if masks.ndim == 2:
            masks = masks[None, :]
        super().__init__(masks, orig_shape)

    @cached_property
    def xyn(self) -> list[np.ndarray]:
        """Return normalized xy-coordinates of the segmentation masks.

        This property calculates and caches the normalized xy-coordinates of the segmentation masks. The coordinates are
        normalized relative to the original image shape.

        Returns:
            (list[np.ndarray]): A list of numpy arrays, where each array contains the normalized xy-coordinates of a
                single segmentation mask. Each array has shape (N, 2), where N is the number of points in the
                mask contour.

        Examples:
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> normalized_coords = masks.xyn
            >>> print(normalized_coords[0])  # Normalized coordinates of the first mask
        """
        return [
            scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in masks2segments(self.data)
        ]

    @cached_property
    def xy(self) -> list[np.ndarray]:
        """Return the [x, y] pixel coordinates for each segment in the mask tensor.

        This property calculates and returns a list of pixel coordinates for each segmentation mask in the Masks object.
        The coordinates are scaled to match the original image dimensions.

        Returns:
            (list[np.ndarray]): A list of numpy arrays, where each array contains the [x, y] pixel coordinates for a
                single segmentation mask. Each array has shape (N, 2), where N is the number of points in the segment.

        Examples:
            >>> results = model("image.jpg")
            >>> masks = results[0].masks
            >>> xy_coords = masks.xy
            >>> print(len(xy_coords))  # Number of masks
            >>> print(xy_coords[0].shape)  # Shape of first mask's coordinates
        """
        return [
            scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=False)
            for x in masks2segments(self.data)
        ]

class Keypoints(BaseResultItem):
    """A class for storing and manipulating detection keypoints.

    This class encapsulates functionality for handling keypoint data, including coordinate manipulation, normalization,
    and confidence values. It supports keypoint detection results with optional visibility information.
    """
    def __init__(self, keypoints: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]):
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3

    @cached_property
    def xy(self) -> torch.Tensor | np.ndarray:
        """Return x, y coordinates of keypoints.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or array containing the x, y coordinates of keypoints with shape (N,
                K, 2), where N is the number of detections and K is the number of keypoints per detection.

        Examples:
            >>> results = model("image.jpg")
            >>> keypoints = results[0].keypoints
            >>> xy = keypoints.xy
            >>> print(xy.shape)  # (N, K, 2)
            >>> print(xy[0])  # x, y coordinates of keypoints for first detection

        Notes:
            - The returned coordinates are in pixel units relative to the original image dimensions.
            - This property uses LRU caching to improve performance on repeated access.
        """
        return self.data[:, :2]

    @cached_property
    def xyn(self) -> torch.Tensor | np.ndarray:
        """Return normalized coordinates (x, y) of keypoints relative to the original image size.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or array of shape (N, K, 2) containing normalized keypoint
                coordinates, where N is the number of instances, K is the number of keypoints, and the last dimension
                contains [x, y] values in the range [0, 1].

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 2), orig_shape=(480, 640))
            >>> normalized_kpts = keypoints.xyn
            >>> print(normalized_kpts.shape)
            torch.Size([1, 17, 2])
        """
        xy = self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[..., 0] /= self.orig_shape[1]
        xy[..., 1] /= self.orig_shape[0]
        return xy

    @cached_property
    def conf(self) -> torch.Tensor | np.ndarray | None:
        """Return confidence values for each keypoint.

        Returns:
            (torch.Tensor | np.ndarray | None): A tensor or array containing confidence scores for each keypoint if
                available, otherwise None. Shape is (num_detections, num_keypoints) for batched data or (num_keypoints,)
                for single detection.

        Examples:
            >>> keypoints = Keypoints(torch.rand(1, 17, 3), orig_shape=(640, 640))  # 1 detection, 17 keypoints
            >>> conf = keypoints.conf
            >>> print(conf.shape)  # torch.Size([1, 17])
        """
        return self.data[..., 2] if self.has_visible else None

class Probs(BaseResultItem):
    """A class for storing and manipulating classification probabilities.

    This class extends BaseResultItem and provides methods for accessing and manipulating classification probabilities,
    including top-1 and top-5 predictions.
    """

    def __init__(self, probs: torch.Tensor | np.ndarray, orig_shape: tuple[int, int] | None = None) -> None:
        super().__init__(probs, orig_shape)

    @cached_property
    def top1(self) -> int:
        return int(self.data.argmax())

    @cached_property
    def top5(self) -> list[int]:
        """Return the indices of the top 5 class probabilities.

        Returns:
            (list[int]): A list containing the indices of the top 5 class probabilities, sorted in descending order.
        """
        return (-self.data).argsort(0)[:5].tolist()  # this way works with both torch and numpy.

    @cached_property
    def top1conf(self) -> torch.Tensor | np.ndarray:
        """Return the confidence score of the highest probability class.

        This property retrieves the confidence score (probability) of the class with the highest predicted probability
        from the classification results.

        Returns:
            (torch.Tensor | np.ndarray): A tensor containing the confidence score of the top 1 class.
        """
        return self.data[self.top1]

    @cached_property
    def top5conf(self) -> torch.Tensor | np.ndarray:
        """Return confidence scores for the top 5 classification predictions.

        This property retrieves the confidence scores corresponding to the top 5 class probabilities predicted by the
        model. It provides a quick way to access the most likely class predictions along with their associated
        confidence levels.

        Returns:
            (torch.Tensor | np.ndarray): A tensor or array containing the confidence scores for the top 5 predicted
                classes, sorted in descending order of probability.
        """
        return self.data[self.top5]

class OBB(BaseResultItem):
    """A class for storing and manipulating Oriented Bounding Boxes (OBB).

    This class provides functionality to handle oriented bounding boxes, including conversion between different formats,
    normalization, and access to various properties of the boxes. It supports both tracking and non-tracking scenarios.
    """
    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int]):
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {7, 8}, f"expected 7 or 8 values but got {n}"  # xywh, rotation, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 8
        self.orig_shape = orig_shape

    @property
    def xywhr(self) -> torch.Tensor | np.ndarray:
        return self.data[:, :5]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        return self.data[:, -2]

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        return self.data[:, -1]

    @property
    def id(self) -> torch.Tensor | np.ndarray:
        return self.data[:, -3] if self.is_track else None

    @cached_property
    def xyxyxyxy(self) -> torch.Tensor | np.ndarray:
        cos, sin, cat, stack = (
            (torch.cos, torch.sin, torch.cat, torch.stack)
            if isinstance(self.xywhr, torch.Tensor)
            else (np.cos, np.sin, np.concatenate, np.stack)
        )
        ctr = self.xywhr[..., :2]
        w, h, angle = (self.xywhr[..., i : i + 1] for i in range(2, 5))
        cos_value, sin_value = cos(angle), sin(angle)
        vec1 = [w / 2 * cos_value, w / 2 * sin_value]
        vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
        vec1 = cat(vec1, -1)
        vec2 = cat(vec2, -1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2

        return stack([pt1, pt2, pt3, pt4], -2)

    @cached_property
    def xyxyxyxyn(self) -> torch.Tensor | np.ndarray:
        """Convert rotated bounding boxes to normalized xyxyxyxy format.

        Returns:
            (torch.Tensor | np.ndarray): Normalized rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2),
                where N is the number of boxes. Each box is represented by 4 points (x, y), normalized relative to the
                original image dimensions.
        """
        xyxyxyxyn = self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[..., 0] /= self.orig_shape[1]
        xyxyxyxyn[..., 1] /= self.orig_shape[0]
        return xyxyxyxyn

    @cached_property
    def xyxy(self) -> torch.Tensor | np.ndarray:
        """Convert oriented bounding boxes (OBB) to axis-aligned bounding boxes in xyxy format.

        This property calculates the minimal enclosing rectangle for each oriented bounding box and returns it in xyxy
        format (x1, y1, x2, y2). This is useful for operations that require axis-aligned bounding boxes, such as IoU
        calculation with non-rotated boxes.

        Returns:
            (torch.Tensor | np.ndarray): Axis-aligned bounding boxes in xyxy format with shape (N, 4), where N is the
                number of boxes. Each row contains [x1, y1, x2, y2] coordinates.
        """
        x = self.xyxyxyxy[..., 0]
        y = self.xyxyxyxy[..., 1]
        return (
            torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
            if isinstance(x, torch.Tensor)
            else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], -1)
        )

class BaseResults(SimpleClass, DataExportMixin):
    """A class for storing and manipulating inference results.

    This class provides comprehensive functionality for handling inference results from various Ultralytics models,
    including detection, instance segmentation, semantic segmentation, classification, pose estimation, and oriented
    bounding box detection. It supports visualization, data export, and various coordinate transformations.

    Attributes:
        orig_img (np.ndarray): The original image as a numpy array.
        orig_shape (tuple[int, int]): Original image shape in (height, width) format.
        boxes (Boxes | None): Detected bounding boxes.
        masks (Masks | None): Segmentation masks.
        probs (Probs | None): Classification probabilities.
        keypoints (Keypoints | None): Detected keypoints.
        obb (OBB | None): Oriented bounding boxes.
        semantic_mask (SemanticMask | None): Semantic segmentation class map.
        depth (DepthMap | None): Per-pixel depth map.
        speed (dict): Dictionary containing inference speed information.
        names (dict): Dictionary mapping class indices to class names.
        path (str): Path to the input image file.
        save_dir (str | None): Directory to save results.

    Methods:
        update: Update the Results object with new detection data.
        cpu: Return a copy of the Results object with all tensors moved to CPU memory.
        numpy: Convert all tensors in the Results object to numpy arrays.
        cuda: Move all tensors in the Results object to GPU memory.
        to: Move all tensors to the specified device and dtype.
        new: Create a new Results object with the same image, path, names, and speed attributes.
        plot: Plot detection results on an input BGR image.
        show: Display the image with annotated inference results.
        save: Save annotated inference results image to file.
        verbose: Return a log string for each task in the results.
        summary: Convert inference results to a summarized dictionary.
        save_txt: Save detection results to a text file.
        save_crop: Save cropped detection images to specified directory.
        to_df: Convert detection results to a Polars DataFrame.
        to_json: Convert detection results to JSON format.
        to_csv: Convert detection results to a CSV format.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> result = results[0]  # Get the first result
        >>> boxes = result.boxes  # Get the boxes for the first result
        >>> masks = result.masks  # Get the masks for the first result
        >>> for result in results:
        ...     result.plot()  # Plot detection results
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: dict[int, str],
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        keypoints: torch.Tensor | None = None,
        obb: torch.Tensor | None = None,
        speed: dict[str, float] | None = None,
        semantic_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ) -> None:
        """Initialize the Results class for storing and manipulating inference results.

        Args:
            orig_img (np.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (dict): A dictionary of class names.
            boxes (torch.Tensor | None): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.Tensor | None): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.Tensor | None): A 1D tensor of probabilities of each class for classification task.
            keypoints (torch.Tensor | None): A 2D tensor of keypoint coordinates for each detection.
            obb (torch.Tensor | None): A 2D tensor of oriented bounding box coordinates for each detection.
            semantic_mask (torch.Tensor | None): A 2D tensor of class IDs for semantic segmentation results.
            depth (torch.Tensor | None): A 2D float tensor of per-pixel depth values (H, W).
            speed (dict | None): A dictionary containing preprocess, inference, and postprocess speeds (ms/image).

        Notes:
            For the default pose model, keypoint indices for human body pose estimation are:
            0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
            5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
            9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
            13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = BoundingBoxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.masks = Masks(masks, self.orig_shape) if masks is not None else None  # native size or imgsz masks
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb = OBB(obb, self.orig_shape) if obb is not None else None
        self.semantic_mask = SemanticMask(semantic_mask, self.orig_shape) if semantic_mask is not None else None
        self.depth = DepthMap(depth, self.orig_shape) if depth is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb", "semantic_mask", "depth"

    def __getitem__(self, idx):
        """Return a Results object for a specific index of inference results.

        Args:
            idx (int | slice): Index or slice to retrieve from the Results object.

        Returns:
            (Results): A new Results object containing the specified subset of inference results.

        Examples:
            >>> results = model("path/to/image.jpg")  # Perform inference
            >>> single_result = results[0]  # Get the first result
            >>> subset_results = results[1:4]  # Get a slice of results
        """
        return self._apply("__getitem__", idx)

    def __len__(self) -> int:
        """Return the number of results in the Results object.

        Returns:
            (int): The number of results, determined by the length of the first non-empty attribute in (boxes, masks,
                probs, keypoints, obb, semantic_mask, or depth). Empty Results objects return 0.

        Examples:
            >>> results = Results(orig_img, path, names, boxes=torch.rand(5, 6))
            >>> len(results)
            5
        """
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)
        return 0

    def update(
        self,
        boxes: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        obb: torch.Tensor | None = None,
        keypoints: torch.Tensor | None = None,
        semantic_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ):
        """Update the Results object with new detection data.

        This method allows updating the boxes, masks, keypoints, probabilities, and oriented bounding boxes (OBB) of
        the Results object. It ensures that boxes are clipped to the original image shape.

        Args:
            boxes (torch.Tensor | None): A tensor of shape (N, 6) containing bounding box coordinates and confidence
                scores. The format is (x1, y1, x2, y2, conf, class).
            masks (torch.Tensor | None): A tensor of shape (N, H, W) containing segmentation masks.
            probs (torch.Tensor | None): A tensor of shape (num_classes,) containing class probabilities.
            obb (torch.Tensor | None): A tensor of shape (N, 7) or (N, 8) containing oriented bounding box coordinates.
            keypoints (torch.Tensor | None): A tensor of shape (N, K, 3) containing keypoints, where K=17 for persons.
            semantic_mask (torch.Tensor | None): A tensor of shape (H, W) containing class IDs for semantic
                segmentation.
            depth (torch.Tensor | None): A tensor of shape (H, W) containing per-pixel depth values.

        Examples:
            >>> results = model("image.jpg")
            >>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
            >>> results[0].update(boxes=new_boxes)
        """
        if boxes is not None:
            self.boxes = BoundingBoxes(clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if probs is not None:
            self.probs = Probs(probs)
        if obb is not None:
            self.obb = OBB(obb, self.orig_shape)
        if keypoints is not None:
            self.keypoints = Keypoints(keypoints, self.orig_shape)
        if semantic_mask is not None:
            self.semantic_mask = SemanticMask(semantic_mask, self.orig_shape)
        if depth is not None:
            self.depth = DepthMap(depth, self.orig_shape)

    def _apply(self, fn: str, *args, **kwargs):
        """Apply a function to all non-empty attributes and return a new Results object with modified attributes.

        This method is internally called by methods like .to(), .cuda(), .cpu(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args (Any): Variable length argument list to pass to the function.
            **kwargs (Any): Arbitrary keyword arguments to pass to the function.

        Returns:
            (Results): A new Results object with attributes modified by the applied function.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result_cuda = result.cuda()
            ...     result_cpu = result.cpu()
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is None:
                continue
            setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """Return a copy of the Results object with all its tensors moved to CPU memory.

        This method creates a new Results object with all tensor attributes (boxes, masks, probs, keypoints, obb)
        transferred to CPU memory. It's useful for moving data from GPU to CPU for further processing or saving.

        Returns:
            (Results): A new Results object with all tensor attributes on CPU memory.

        Examples:
            >>> results = model("path/to/image.jpg")  # Perform inference
            >>> cpu_result = results[0].cpu()  # Move the first result to CPU
            >>> print(cpu_result.boxes.device)  # Output: cpu
        """
        return self._apply("cpu")

    def numpy(self):
        """Convert all tensors in the Results object to numpy arrays.

        Returns:
            (Results): A new Results object with all tensors converted to numpy arrays.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> numpy_result = results[0].numpy()
            >>> type(numpy_result.boxes.data)
            <class 'numpy.ndarray'>

        Notes:
            This method creates a new Results object, leaving the original unchanged. It's useful for
            interoperability with numpy-based libraries or when CPU-based operations are required.
        """
        return self._apply("numpy")

    def cuda(self):
        """Move all tensors in the Results object to GPU memory.

        Returns:
            (Results): A new Results object with all tensors moved to CUDA device.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> cuda_results = results[0].cuda()  # Move first result to GPU
            >>> for result in results:
            ...     result_cuda = result.cuda()  # Move each result to GPU
        """
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """Move all tensors in the Results object to the specified device and dtype.

        Args:
            *args (Any): Variable length argument list to be passed to torch.Tensor.to().
            **kwargs (Any): Arbitrary keyword arguments to be passed to torch.Tensor.to().

        Returns:
            (Results): A new Results object with all tensors moved to the specified device and dtype.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> result_cuda = results[0].to("cuda")  # Move first result to GPU
            >>> result_cpu = results[0].to("cpu")  # Move first result to CPU
            >>> result_half = results[0].to(dtype=torch.float16)  # Convert first result to half precision
        """
        return self._apply("to", *args, **kwargs)

    @abstractmethod
    def new(self):
        """Create a new Results object with the same image, path, names, and speed attributes.

        Returns:
            (Results): A new Results object with copied attributes from the original instance.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        raise NotImplementedError()

    @abstractmethod
    def plot(
        self,
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: np.ndarray | torch.Tensor | None = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Plot detection results on an input BGR image.

        Args:
            conf (bool): Whether to plot detection confidence scores.
            line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
            font_size (float | None): Font size for text. If None, scaled to image size.
            font (str): Font to use for text.
            pil (bool): Whether to return the image as a PIL Image.
            img (np.ndarray | torch.Tensor | None): Image to plot on. Tensor images must be contiguous HWC BGR uint8. If
                None, uses the original image.
            kpt_radius (int): Radius of drawn keypoints.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot labels of bounding boxes.
            boxes (bool): Whether to plot bounding boxes.
            masks (bool): Whether to plot masks.
            probs (bool): Whether to plot classification probabilities.
            show (bool): Whether to display the annotated image.
            save (bool): Whether to save the annotated image.
            filename (str | None): Filename to save image if save is True.
            color_mode (str): Specify the color mode, e.g., 'instance' or 'class'.
            txt_color (tuple[int, int, int]): Text color in BGR format for classification output.

        Returns:
            (np.ndarray | PIL.Image.Image): Annotated image as a NumPy array (BGR) or PIL image (RGB) if `pil=True`.

        Examples:
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     im = result.plot(pil=True)
            ...     im.show()
        """
        raise NotImplementedError()

    def show(self, *args, **kwargs):
        """Display the image with annotated inference results.

        This method plots the detection results on the original image and displays it. It's a convenient way to
        visualize the model's predictions directly.

        Args:
            *args (Any): Variable length argument list to be passed to the `plot()` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot()` method.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> results[0].show()  # Display the first result
            >>> for result in results:
            ...     result.show()  # Display all results
        """
        self.plot(*args, show=True, **kwargs)

    def save(self, filename: str | None = None, *args, **kwargs) -> str:
        """Save annotated inference results image to file.

        This method plots the detection results on the original image and saves the annotated image to a file. It
        utilizes the `plot` method to generate the annotated image and then saves it to the specified filename.

        Args:
            filename (str | None): The filename to save the annotated image. If None, a default filename is generated
                based on the original image path.
            *args (Any): Variable length argument list to be passed to the `plot` method.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the `plot` method.

        Returns:
            (str): The filename where the image was saved.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save("annotated_image.jpg")
            >>> # Or with custom plot arguments
            >>> for result in results:
            ...     result.save("annotated_image.jpg", conf=False, line_width=2)
            >>> # Directory will be created automatically if it does not exist
            >>> result.save("path/to/annotated_image.jpg")
        """
        if not filename:
            filename = f"results_{Path(self.path).name}"
        Path(filename).absolute().parent.mkdir(parents=True, exist_ok=True)
        self.plot(*args, save=True, filename=filename, **kwargs)
        return filename

    def verbose(self) -> str:
        """Return a log string for each task in the results, detailing detection and classification outcomes.

        This method generates a human-readable string summarizing the detection and classification results. It includes
        the number of detections for each class and the top probabilities for classification tasks.

        Returns:
            (str): A formatted string containing a summary of the results. For detection tasks, it includes the number
                of detections per class. For classification tasks, it includes the top 5 class probabilities.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     print(result.verbose())
            2 persons, 1 car, 3 traffic lights,
            dog 0.92, cat 0.78, horse 0.64,

        Notes:
            - If there are no detections, the method returns "(no detections), " for detection tasks.
            - For classification tasks, it returns the top 5 class probabilities and their corresponding class names.
            - The returned string is comma-separated and ends with a comma and a space.
        """
        boxes = self.obb if self.obb is not None else self.boxes
        if len(self) == 0:
            return "" if self.probs is not None else "(no detections), "
        if self.probs is not None:
            return f"{', '.join(f'{self.names[j]} {self.probs.data[j]:.2f}' for j in self.probs.top5)}, "
        if boxes:
            counts = torch.as_tensor(boxes.cls, dtype=torch.int64).bincount()  # no-op for torch, converts numpy()
            return "".join(f"{n} {self.names[i]}{'s' * (n > 1)}, " for i, n in enumerate(counts) if n > 0)
        if self.depth is not None:
            d = self.depth.data
            d = d[d > 0]
            return f"depth {float(d.min()):.2f}-{float(d.max()):.2f}m, " if len(d) else "depth (no valid pixels), "
        if self.semantic_mask is not None:
            return ""

    @abstractmethod
    def save_txt(self, txt_file: str | Path, save_conf: bool = False) -> str:
        """Save detection results to a text file.

        Args:
            txt_file (str | Path): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo26n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_txt("output.txt")

        Notes:
            - The file will contain one line per detection or classification with the following structure:
              - For detections: `class x_center y_center width height [confidence] [track_id]`
              - For classifications: `confidence class_name`
              - For masks and keypoints, the specific formats will vary accordingly.
            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.
            - Existing contents of the file will not be overwritten; new results will be appended.
            - This method does not support Semantic Segmentation tasks.
        """
        pass

    @abstractmethod
    def save_crop(self, save_dir: str | Path, file_name: str | Path = Path("im.jpg")):
        """Save cropped detection images to specified directory.

        This method saves cropped images of detected objects to a specified directory. Each crop is saved in a
        subdirectory named after the object's class, with the filename based on the input file_name.

        Args:
            save_dir (str | Path): Directory path where cropped images will be saved.
            file_name (str | Path): Base filename for the saved cropped images.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_crop(save_dir="path/to/crops", file_name="detection")

        Notes:
            - This method does not support Classify, Oriented Bounding Box (OBB), or Semantic Segmentation tasks.
            - Crops are saved as 'save_dir/class_name/file_name.jpg'.
            - The method will create necessary subdirectories if they don't exist.
            - Original image is copied before cropping to avoid modifying the original.
        """
        pass

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, Any]]:
        """Convert inference results to a summarized dictionary with optional normalization for box coordinates.

        This method creates a list of detection dictionaries, each containing information about a single detection or
        classification result. For classification tasks, it returns the top 5 classes and their
        confidences. For detection tasks, it includes class information, bounding box coordinates, and
        optionally mask segments and keypoints.

        Args:
            normalize (bool): Whether to normalize bounding box coordinates by image dimensions.
            decimals (int): Number of decimal places to round the output values to.

        Returns:
            (list[dict[str, Any]]): A list of dictionaries, each containing summarized information for a single
                detection or classification result. The structure of each dictionary varies based on the task type
                (classification or detection) and available information (boxes, masks, keypoints).

        Examples:
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     summary = result.summary()
            ...     print(summary)
        """
        # Create list of detection dictionaries
        results = []
        if self.probs is not None:
            # Return top 5 classification results
            for class_id, conf in zip(self.probs.top5, self.probs.top5conf.tolist()):
                class_id = int(class_id)
                results.append(
                    {
                        "name": self.names[class_id],
                        "class": class_id,
                        "confidence": round(conf, decimals),
                    }
                )
            return results

        if self.semantic_mask is not None:
            # Return per-class pixel coverage for semantic segmentation
            mask = self.semantic_mask.data
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            unique, counts = np.unique(mask, return_counts=True)
            total = mask.size
            for class_id, count in zip(unique.tolist(), counts.tolist()):
                if len(self.names) == 1:
                    if class_id != 1:  # skip binary background and ignore label
                        continue
                    class_id = 0
                elif class_id not in self.names:  # skip ignore label (e.g., 255)
                    continue
                results.append(
                    {
                        "name": self.names[class_id],
                        "class": class_id,
                        "pixel_ratio": round(count / total, decimals),
                    }
                )
            return results

        if self.depth is not None:
            # Depth is a dense per-pixel map, not a per-instance result; excluded from summary.
            return results

        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            class_id, conf = int(row.cls.item()), round(row.conf.item(), decimals)
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
            if data.is_track:
                result["track_id"] = int(row.id.item())  # track ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).astype(float).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).astype(float).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                kpt = self.keypoints[i]
                k = kpt.data[0]
                k = k.cpu().numpy() if isinstance(k, torch.Tensor) else k
                result["keypoints"] = {
                    "x": (k[:, 0] / w).astype(float).round(decimals).tolist(),
                    "y": (k[:, 1] / h).astype(float).round(decimals).tolist(),
                }
                if kpt.has_visible:
                    result["keypoints"]["visible"] = k[:, 2].astype(float).round(decimals).tolist()
            results.append(result)

        return results