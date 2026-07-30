from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import glob
import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod


class BaseDataset(Dataset, ABC):
    """Abstract Base Dataset for computer vision tasks."""

    def __init__(
        self,
        img_path: str,
        imgsz: int = 640,
        augment: bool = True,
        hyp: Optional[Dict[str, Any]] = None,
        rect: bool = False,
        cache: bool = False,
        stride: int = 32,
        pad: float = 0.5,
        prefix: str = "",
    ):
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.hyp = hyp or {}
        self.rect = rect
        self.cache = cache
        self.stride = stride
        self.pad = pad
        self.prefix = prefix

        # Find image files
        self.im_files = self.get_img_files(self.img_path)
        if not self.im_files:
            raise FileNotFoundError(f"{self.prefix}No images found in {self.img_path}")

        # Load labels metadata
        self.labels = self.load_annotations()

    @abstractmethod
    def get_img_files(self, img_path: str) -> List[str]:
        """Gather list of image paths from specified path."""
        raise NotImplementedError

    @abstractmethod
    def load_annotations(self) -> List[Dict[str, Any]]:
        """Load bounding box / segmentation / keypoint labels corresponding to im_files."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.im_files)

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Fetch preprocessed image and targets at given index."""
        raise NotImplementedError


class YOLODataset(BaseDataset):
    """YOLO Dataset for object detection supporting standard .txt annotations."""

    def get_img_files(self, img_path: str) -> List[str]:
        path = Path(img_path)
        if path.is_dir():
            files = sorted(glob.glob(os.path.join(img_path, "**/*.*"), recursive=True))
        elif path.is_file() and path.suffix == ".txt":  # text file listing image paths
            with open(path, "r") as f:
                files = [x.strip() for x in f.readlines() if x.strip()]
        else:
            files = [str(path)]

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        return [f for f in files if Path(f).suffix.lower() in valid_extensions]

    def load_annotations(self) -> List[Dict[str, Any]]:
        labels = []
        for img_file in self.im_files:
            # Map image path to label path: images/x.jpg -> labels/x.txt
            label_file = (
                img_file.replace("/images/", "/labels/")
                .replace("\\images\\", "\\labels\\")
                .rsplit(".", 1)[0]
                + ".txt"
            )

            bboxes, clss = [], []
            if os.path.isfile(label_file):
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            clss.append(int(parts[0]))
                            # YOLO format: cx, cy, w, h normalized [0, 1]
                            bboxes.append([float(x) for x in parts[1:5]])

            labels.append(
                {
                    "im_file": img_file,
                    "cls": np.array(clss, dtype=np.float32).reshape(-1, 1),
                    "bboxes": np.array(bboxes, dtype=np.float32).reshape(-1, 4),
                }
            )
        return labels

    def load_image(self, i: int) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Loads 1 image from dataset index 'i', returning (img, original_shape, resized_shape)."""
        f = self.im_files[i]
        img = cv2.imread(f)
        if img is None:
            raise ValueError(f"Image Not Found {f}")
        h0, w0 = img.shape[:2]
        r = self.imgsz / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Loads image, applies augmentations, and returns transformed data."""
        # Load image & labels
        img, (h0, w0), (h, w) = self.load_image(index)
        label = self.labels[index].copy()
        bboxes = label["bboxes"].copy()
        cls = label["cls"].copy()

        # Letterbox padding
        shape = (self.imgsz, self.imgsz)
        r = min(shape[0] / h, shape[1] / w)
        unpad_h, unpad_w = int(round(h * r)), int(round(w * r))
        dh, dw = (shape[0] - unpad_h) / 2, (shape[1] - unpad_w) / 2

        if (w, h) != (unpad_w, unpad_h):
            img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Convert bboxes from normalized cxcywh to letterboxed pixel xywh
        if len(bboxes) > 0:
            bboxes[:, 0] = bboxes[:, 0] * unpad_w + dw  # cx
            bboxes[:, 1] = bboxes[:, 1] * unpad_h + dh  # cy
            bboxes[:, 2] = bboxes[:, 2] * unpad_w       # w
            bboxes[:, 3] = bboxes[:, 3] * unpad_h       # h

        # Augmentation: Random Flip Left-Right
        if self.augment and np.random.random() < self.hyp.get("fliplr", 0.5):
            img = np.fliplr(img)
            if len(bboxes) > 0:
                bboxes[:, 0] = self.imgsz - bboxes[:, 0]

        # Convert BGR to RGB, normalize to CHW PyTorch tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float() / 255.0

        # Construct targets tensor: [batch_idx, class_id, cx, cy, w, h]
        nl = len(bboxes)
        targets = torch.zeros((nl, 6))
        if nl > 0:
            targets[:, 1] = torch.from_numpy(cls.squeeze(-1))
            # Normalize bounding box coordinates to [0, 1] relative to imgsz canvas
            targets[:, 2:] = torch.from_numpy(bboxes) / self.imgsz

        return {
            "img": tensor_img,
            "target": targets,
            "im_file": self.im_files[index],
            "ori_shape": (h0, w0),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for batching bounding boxes of variable length."""
        imgs = torch.stack([b["img"] for b in batch], 0)
        targets = []
        for i, item in enumerate(batch):
            target = item["target"]
            if len(target) > 0:
                target[:, 0] = i  # Assign batch index
                targets.append(target)

        targets = torch.cat(targets, 0) if len(targets) > 0 else torch.zeros((0, 6))
        im_files = [b["im_file"] for b in batch]
        ori_shapes = [b["ori_shape"] for b in batch]

        return {
            "img": imgs,
            "target": targets,
            "im_file": im_files,
            "ori_shape": ori_shapes,
        }