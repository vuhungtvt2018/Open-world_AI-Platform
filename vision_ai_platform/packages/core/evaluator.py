from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import EvaluatorConfig

class BaseEvaluator(ABC):
    """Base class for validating models and calculating evaluation metrics.

    Attributes:
        cfg (EvaluatorConfig): Validation parameters.
        model (nn.Module): Model being evaluated.
        metrics (Dict[str, float]): Computed evaluation metrics (mAP, Precision, Recall).
    """

    def __init__(self, cfg: "EvaluatorConfig", model: Optional[nn.Module] = None) -> None:
        """Initialize evaluator with configuration."""
        self.cfg = cfg
        self.model = model
        self.metrics: Dict[str, float] = {}

    @abstractmethod
    def init_metrics(self) -> None:
        """Initialize metric tracking containers."""
        raise NotImplementedError

    @abstractmethod
    def update_metrics(self, preds: Any, targets: Any) -> None:
        """Update metrics state with a batch of predictions and ground truth targets."""
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        """Compute final evaluation metrics across all processed batches."""
        raise NotImplementedError

    def evaluate(self, model: nn.Module, dataloader: Any) -> Dict[str, float]:
        """Run complete validation pass over a dataloader.

        Args:
            model (nn.Module): Evaluation target model.
            dataloader: Validation dataloader yielding (batch_inputs, targets).

        Returns:
            Dict[str, float]: Calculated metrics dictionary (e.g., mAP50, mAP50-95).
        """
        self.model = model
        self.model.eval()
        self.init_metrics()

        with torch.no_grad():
            for inputs, targets in dataloader:
                preds = self.model(inputs)
                self.update_metrics(preds, targets)

        self.metrics = self.compute_metrics()
        return self.metrics