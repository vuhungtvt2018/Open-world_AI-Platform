from __future__ import annotations

from collections import defaultdict
from typing import Optional, Tuple, Dict
from pathlib import Path
import os

import cv2
import numpy as np

from vision_ai_platform.packages.workflow.base import Workflow
from vision_ai_platform.packages.workflow.annotator import WorkflowAnnotator
from vision_ai_platform.packages.core import WorkflowConfig, WorkflowResults
from vision_ai_platform.packages.utils import LOGGER
from vision_ai_platform.packages.utils.annotator import colors


class TrainingWorkflow(Workflow):
    """A Workflow subclass to execute and monitor model training tasks.

    This class encapsulates data initialization, model fine-tuning/training,
    epoch loss tracking, validation evaluation, and model artifact exporting.

    Attributes:
        current_epoch (int): Current training epoch count.
        best_fitness (float): Highest evaluation score/fitness achieved.
        loss_history (dict[str, list[float]]): Epoch-wise tracking of training and validation losses.
        is_trained (bool): Status flag confirming whether model training has completed.
        save_dir (str): Directory where trained checkpoints and training runs are stored.

    Methods:
        train_epoch: Execute a single epoch step across training data.
        validate: Run evaluation on the validation split and record metrics.
        export_model: Export trained weights to specified deployment formats (e.g., ONNX, TensorRT).
        plot_training_progress: Render real-time loss curves and training metrics onto a frame buffer.
        process: Execute the complete training lifecycle workflow.
    """
    def __init__(self, cfg: WorkflowConfig):
        super().__init__(cfg)

        self.current_epoch = 0
        self.best_fitness = 0.0
        self.is_trained = False

        self.loss_history = defaultdict(list)
        
        # Configure output experiment output directory
        self.save_dir = Path(self.cfg.na)
        self.save_dir = os.path.join(
            getattr(self.cfg, "project", "runs/train"),
            getattr(self.cfg, "name", "exp")
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.logger.info(f"Training Workflow initialized. Artifacts will be saved to: {self.save_dir}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute training operations for a single epoch.

        Args:
            epoch (int): Index of the current epoch.

        Returns:
            Dict[str, float]: Dictionary containing loss metrics (box_loss, cls_loss, dfl_loss, total_loss)
                for the current epoch.
        """
        self.logger.info(f"Epoch {epoch}/{self.cfg.epochs} - Training in progress...")
        self.model.train()

    def process(self, data_path: Optional[Path | str] = None, **kwargs):
        dataset = Path(data_path) or getattr(self.cfg, "data", "coco8.yaml")
