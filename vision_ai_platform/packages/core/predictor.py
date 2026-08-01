from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, List, Union, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import YOLOConfig

class BasePredictor(ABC):
    """A base class for creating predictors.

    This class provides the foundation for prediction functionality, handling model setup, inference, and result
    processing across various input sources.

    Attributes:
        cfg (YOLOConfig): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (torch.nn.Module): Model used for prediction.
        data (str): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_writer (dict[Path, cv2.VideoWriter]): Dictionary of {save_path: video_writer} for saving video output.
        plotted_img (np.ndarray): Last plotted image.
        source_type (SimpleNamespace): Type of input source.
        seen (int): Number of images processed.
        windows (list[str]): List of window names for visualization.
        batch (tuple): Current batch data.
        results (list[Any]): Current batch results.
        transforms (Callable): Image transforms for classification.
        callbacks (dict[str, list[Callable]]): Callback functions for different events.
        txt_path (Path): Path to save text results.
        _lock (threading.Lock): Lock for thread-safe inference.

    Methods:
        preprocess: Prepare input image before inference.
        inference: Run inference on a given image.
        postprocess: Process raw predictions into structured results.
        setup_source: Set up input source and inference mode.
        stream_inference: Stream inference on input source.
        setup_model: Initialize and configure the model.
        write_results: Write inference results to files.
        save_predicted_images: Save prediction visualizations.
        show: Display results in a window.
        run_callbacks: Execute registered callbacks for an event.
        add_callback: Register a new callback function.
    """

    def __init__(
        self,
        cfg: YOLOConfig,
        save_dir: Optional[str | Path] = None,
        _callbacks: Optional[Dict[str, List[Callable]]] = None,
    ):
        """Initialize the custom predictor with PredictorConfig.

        Args:
            cfg (PredictorConfig): Pydantic or structured configuration instance.
            save_dir (str | Path): Directory to save the prediction results
            _callbacks (dict, optional): Registered callback dictionary.
        """
        self.cfg = cfg

        # Set default values if unassigned
        if self.cfg.conf is None:
            self.cfg.conf = 0.25

        self.save_dir = Path(save_dir) if save_dir else None
        self.done_warmup = False

        # Model and Runtime state attributes
        self.model: Optional[nn.Module] = None
        self.data: Any = getattr(self.cfg, "data", None)
        self.imgsz: Any = None
        self.device: Optional[torch.device] = None
        self.dataset: Any = None
        self.vid_writer: Dict[Path, cv2.VideoWriter] = {}
        self.plotted_img: Optional[np.ndarray] = None
        self.source_type: Any = None
        self.seen: int = 0
        self.speed: Optional[Dict[str, float]] = None
        self.pixels: Optional[int] = None
        self.windows: List[str] = []
        self.screen: Optional[tuple] = None
        self.batch: Any = None
        self.results: Any = None
        self.transforms: Optional[Callable] = None
        self.txt_path: Optional[Path] = None

        # Callbacks and Threading
        self.callbacks: Optional[Dict[str, List[Callable]]] = None
        self._lock = threading.Lock()

    @abstractmethod
    def setup_model(self, model: Union[str, Path, torch.nn.Module, None] = None, verbose: bool = True):
        """Initialize backend model and prepare for evaluation mode."""
        raise NotImplementedError()

    @abstractmethod
    def setup_source(self, source: Any, stride: Optional[int] = None):
        """Configure video/image dataloader source."""
        pass
    
    @abstractmethod
    def pre_transform(self, im: List[np.ndarray]) -> List[np.ndarray]:
        """Apply letterbox resize transform."""
        raise NotImplementedError()

    def preprocess(self, im: Union[torch.Tensor, List[np.ndarray]]) -> torch.Tensor:
        """Preprocess inputs prior to forwarding to model."""
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # BGR to RGB
            im = im.transpose((0, 3, 1, 2))  # BHWC -> BCHW
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if getattr(self.model, "fp16", False) else im.float()
        if not_tensor:
            im /= 255.0  # Normalize [0, 255] to [0.0, 1.0]
        return im

    @abstractmethod
    def inference(self, im: torch.Tensor, *args, **kwargs):
        """Execute model forward pass."""
        raise NotImplementedError()

    def postprocess(self, preds: Any, img: torch.Tensor, orig_imgs: Any) -> Any:
        """Override in subclasses to perform Task-specific NMS / post-processing."""
        return preds

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: List[str]) -> str:
        """Write bounding box metadata / visual outputs to file."""
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # None if frame undetermined

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # Add predictions to image
        if self.cfg.save or self.cfg.show:
            self.plotted_img = result.plot(
                line_width=self.cfg.line_width,
                boxes=self.cfg.show_boxes,
                conf=self.cfg.show_conf,
                labels=self.cfg.show_labels,
            )

        # Save results
        if self.cfg.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.cfg.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.cfg.show:
            self.show(str(p))
        if self.cfg.save:
            self.save_predicted_images(self.save_dir / p.name, frame)

        return string

    @abstractmethod
    def save_predicted_images(self, save_path: Path, frame: int = 0):
        """Save predictions to image or video files."""
        pass

    def show(self, p: str = ""):
        """Display image prediction window."""
        im = self.plotted_img
        if im is None:
            return
        if platform.system() in {"Linux", "Windows"} and p not in self.windows:
            self.windows.append(p)
            name = p.encode("unicode_escape").decode()
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            h, w = im.shape[:2]
            try:
                if self.screen is None:
                    root = __import__("tkinter").Tk()
                    root.withdraw()
                    self.screen = 0.9 * root.winfo_screenwidth(), 0.9 * root.winfo_screenheight()
                    root.destroy()
                r = min(self.screen[0] / w, self.screen[1] / h, 1.0)
                cv2.resizeWindow(name, max(1, int(w * r)), max(1, int(h * r)))
            except Exception:
                cv2.resizeWindow(name, w, h)
        cv2.imshow(p, im)
        if cv2.waitKey(300 if self.dataset.mode == "image" else 1) & 0xFF == ord("q"):
            raise StopIteration

    def run_callbacks(self, event: str):
        """Execute callbacks registered for `event`."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func: Callable):
        """Register a new event callback."""
        self.callbacks.setdefault(event, []).append(func)

    @abstractmethod
    def __call__(self, source: Any = None, model: Any = None, stream: bool = False, *args, **kwargs):
        """Call method enabling predictor execution as callable."""
        raise NotImplementedError()