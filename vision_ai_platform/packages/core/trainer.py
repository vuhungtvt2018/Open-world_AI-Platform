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
import torch.optim as optim

from .config import TrainerConfig

class BaseTrainer(ABC):
    """Base class for training neural network models.

    Handles dataset setup, training loops, optimization, and saving checkpoints.

    Attributes:
        cfg (TrainerConfig): Training configuration parameters.
        evaluator (BaseEvaluator): Evaluator instance.
        model (nn.Module): PyTorch model instance.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        epoch (int): Current epoch counter.        
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (torch.Tensor): Current loss value.
        tloss (dict): Running mean of loss items.
        loss_names (tuple): Names of loss items, derived from the loss dict returned by the criterion on the first
            batch.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of metrics.
        plots (dict): Dictionary of plots.
    """

    def __init__(self, cfg: "TrainerConfig", model: Optional[nn.Module] = None) -> None:
        """Initialize trainer with configuration and optional model instance."""
        self.cfg = cfg
        self.evaluator = None
        self.metrics = None
        self.plots = {}

        # Dirs
        self.save_dir = Path(cfg.save_dir).resolve()
        self.weight_dir = self.save_dir / "weights"
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"

        self.batch_size = self.cfg.batch_size
        self.epochs = self.cfg.epochs or 100
        self.epoch: int = 0
        self.save_period = self.cfgs.save_period

        # Model and dataset
        self.model = model
        self.data = None
        self.ema = None

        # Optimization init
        self.optimizer: Optional[optim.Optimizer] = None
        self.lf = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ()
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.cfg.resume:
            self.csv.unlink()
        self.plot_idx = [0, 1, 2]
        self.nan_recovery_attempts = 0

    @abstractmethod
    def build_dataset(self, data_path: str, mode: str = "train") -> Any:
        """Construct data loader or dataset object.

        Args:
            data_path (str): Path to dataset or dataset config.
            mode (str): Data split mode ('train', 'val').
        """
        raise NotImplementedError

    @abstractmethod
    def build_optimizer(self) -> optim.Optimizer:
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

    @abstractmethod
    def get_validator(self):
        """Raise NotImplementedError (must be implemented by subclasses)."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    @abstractmethod
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Raise NotImplementedError (must return a `torch.utils.data.DataLoader` in subclasses)."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return a loss dict with labeled training loss items, or a list of loss names if loss_items is None."""
        if loss_items is None:
            return [f"{prefix}/{x}" for x in self.loss_names]
        return {f"{prefix}/{k}": round(float(v), 5) for k, v in loss_items.items()}

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance. Override in subclasses."""

    def build_targets(self, preds, targets):
        """Build target tensors for training YOLO model."""

    def progress_string(self):
        """Return a string describing training progress."""
        return ""
    
    