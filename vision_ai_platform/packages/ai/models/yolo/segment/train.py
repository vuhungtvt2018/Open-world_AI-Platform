# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Optional

from vision_ai_platform.packages.core import YOLOConfig
from vision_ai_platform.packages.ai.models import yolo
from vision_ai_platform.packages.ai.nn.tasks import SegmentationModel
from vision_ai_platform.packages.utils import DEFAULT_CFG, RANK


class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training based on a segmentation model.

    This trainer specializes in handling segmentation tasks, extending the detection trainer with segmentation-specific
    functionality including model initialization, validation, and visualization.

    Attributes:
        loss_names (tuple[str]): Names of the loss components used during training.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationTrainer
        >>> args = dict(model="yolo26n-seg.pt", data="coco8-seg.yaml", epochs=3)
        >>> trainer = SegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(
        self,
        cfg: YOLOConfig = DEFAULT_CFG,
        save_dir: Optional[str | Path] = None,
        _callbacks: dict | None = None
    ):
        """Initialize a SegmentationTrainer object.

        Args:
            cfg (dict): Configuration dictionary with default training settings.
            save_dir (str | Path, optional): Directory path to save training results.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.
        """
        trainer_cfg = cfg.model_copy(update={"task": "segment"})
        super().__init__(trainer_cfg, save_dir, _callbacks)

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """Initialize and return a SegmentationModel with specified configuration and weights.

        Args:
            cfg (dict | str, optional): Model configuration. Can be a dictionary, a path to a YAML file, or None.
            weights (str | Path, optional): Path to pretrained weights file.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (SegmentationModel): Initialized segmentation model with loaded weights if specified.

        Examples:
            >>> trainer = SegmentationTrainer()
            >>> model = trainer.get_model(cfg="yolo26n-seg.yaml")
            >>> model = trainer.get_model(weights="yolo26n-seg.pt", verbose=False)
        """
        model = self.set_model_names_for_load(
            SegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return yolo.segment.SegmentationValidator(
            copy(self.cfg), self.test_loader, save_dir=self.save_dir, _callbacks=self.callbacks
        )
