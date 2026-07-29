from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import TrainerConfig

class BaseTrainer(ABC):
    """Base class for training neural network models.

    Handles dataset setup, training loops, optimization, and saving checkpoints.

    Attributes:
        cfg (TrainerConfig): Training configuration parameters.
        model (nn.Module): PyTorch model instance.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        epoch (int): Current epoch counter.
    """

    def __init__(self, cfg: "TrainerConfig", model: Optional[nn.Module] = None) -> None:
        """Initialize trainer with configuration and optional model instance."""
        self.cfg = cfg
        self.model = model
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.epoch: int = 0
        self.best_fitness: float = 0.0

    @abstractmethod
    def build_dataset(self, data_path: str, mode: str = "train") -> Any:
        """Construct data loader or dataset object.

        Args:
            data_path (str): Path to dataset or dataset config.
            mode (str): Data split mode ('train', 'val').
        """
        raise NotImplementedError

    @abstractmethod
    def build_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer based on self.cfg parameters."""
        raise NotImplementedError

    def train(self, data_path: str) -> Dict[str, Any]:
        """Execute the full training pipeline across specified epochs.

        Args:
            data_path (str): Path to training dataset configuration or folder.

        Returns:
            Dict[str, Any]: Final training metrics and execution statistics.
        """
        self.optimizer = self.build_optimizer()
        train_loader = self.build_dataset(data_path, mode="train")

        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            self.train_one_epoch(train_loader)
            
            # Save checkpoint logic or early stopping evaluation can be placed here

        return {"status": "success", "epochs_completed": self.cfg.epochs}

    @abstractmethod
    def train_one_epoch(self, dataloader: Any) -> None:
        """Run a single epoch training loop over the dataloader."""
        raise NotImplementedError