from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, List

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import PredictorConfig

class BasePredictor(ABC):
    """Base class for making inference predictions on various sources.

    Handles pre-processing, model forward pass, and post-processing (e.g. NMS).

    Attributes:
        cfg (PredictorConfig): Inference settings.
        model (nn.Module): PyTorch model used for evaluation.
        device (torch.device): Device on which model inference is run.
    """

    def __init__(self, cfg: "PredictorConfig", model: Optional[nn.Module] = None) -> None:
        """Initialize predictor with configuration and target model."""
        self.cfg = cfg
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup_model(self, model: nn.Module) -> None:
        """Assign model instance and set evaluation mode."""
        self.model = model.to(self.device)
        self.model.eval()

    @abstractmethod
    def preprocess(self, source: Any) -> torch.Tensor:
        """Preprocess inputs (resizing, normalization, batching)."""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, preds: torch.Tensor, orig_imgs: Any) -> List[Any]:
        """Apply NMS, threshold filtering, and restore original coordinate bounds."""
        raise NotImplementedError

    def __call__(self, source: Any, model: Optional[nn.Module] = None) -> List[Any]:
        """Callable wrapper executing full prediction pipeline."""
        return self.predict(source, model=model)

    def predict(self, source: Any, model: Optional[nn.Module] = None) -> List[Any]:
        """Run end-to-end inference on the input source.

        Args:
            source: Image input path, tensor, numpy array, or stream.
            model (Optional[nn.Module]): Override target model instance.

        Returns:
            List[Any]: Formatted prediction outputs (e.g. bounding boxes/masks).
        """
        if model is not None:
            self.setup_model(model)

        inputs = self.preprocess(source)
        with torch.no_grad():
            preds = self.model(inputs) if self.model else None
        
        results = self.postprocess(preds, source)
        return results