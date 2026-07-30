from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .predictor import BasePredictor
from .evaluator import BaseEvaluator
from .trainer import BaseTrainer
from .tracker import BaseTracker

from .config import ModelConfig, TrackerConfig, TrainerConfig, EvaluatorConfig, PredictorConfig

class BaseModel(ABC):
    """Unified Base Class for models (referencing Ultralytics YOLO API).

    Aggregates training, prediction, evaluation, and tracking workflows.

    Attributes:
        cfg (ModelConfig): Model architectural parameters.
        model (Optional[nn.Module]): Underlying PyTorch network.
        predictor (Optional[BasePredictor]): Active predictor instance.
        trainer (Optional[BaseTrainer]): Active trainer instance.
        evaluator (Optional[BaseEvaluator]): Active evaluator instance.
        tracker (Optional[BaseTracker]): Active tracker instance.
        callbacks (Dict[str, List[Callable]]): Hooks for pipeline event customization.
    """

    def __init__(
        self, 
        model_cfg: "ModelConfig", 
        weights_path: Optional[str] = None
    ) -> None:
        """Initialize model architecture and optional weight checkpoint."""
        self.cfg = model_cfg
        self.weights_path = weights_path
        self.model: Optional[nn.Module] = None
        self.device = model_cfg.device
        
        # Internal modules
        self.predictor: Optional[BasePredictor] = None
        self.trainer: Optional[BaseTrainer] = None
        self.evaluator: Optional[BaseEvaluator] = None
        self.tracker: Optional["BaseTracker"] = None
        
        # Event callbacks map
        self.callbacks: Dict[str, List[Callable]] = {
            "on_train_start": [],
            "on_train_end": [],
            "on_predict_start": [],
            "on_predict_end": [],
        }

        if weights_path:
            self._load(weights_path)
        else:
            self._new()

    def _new(self) -> None:
        """Instantiate new model architecture from config."""
        pass

    def _load(self, weights_path: str) -> None:
        """Load model state and weights from checkpoint path."""
        pass

    def __call__(self, source: Any, **kwargs) -> List[Any]:
        """Alias for predict method."""
        return self.predict(source, **kwargs)

    def predict(
        self, 
        source: Any, 
        predict_cfg: Optional["PredictorConfig"] = None, 
        **kwargs
    ) -> List[Any]:
        """Perform predictions on given image/video sources."""
        if self.predictor is None:
            # Fallback initialization using given or default PredictorConfig
            cfg = predict_cfg or PredictorConfig(**kwargs)
            self.predictor = self.get_predictor(cfg)

        self.run_callbacks("on_predict_start")
        results = self.predictor.predict(source, model=self.model)
        self.run_callbacks("on_predict_end")
        return results

    def track(
        self, 
        source: Any, 
        tracker_cfg: Optional["TrackerConfig"] = None, 
        **kwargs
    ) -> List[Any]:
        """Perform object tracking on video sequence or stream."""
        if self.tracker is None:
            cfg = tracker_cfg or TrackerConfig(**kwargs)
            self.tracker = self.get_tracker(cfg)

        # Runs model prediction and passes detected bboxes into tracker instance
        detections = self.predict(source, **kwargs)
        tracked_results = []
        for det in detections:
            tracks = self.tracker.update(det)
            tracked_results.append(tracks)

        return tracked_results

    def evaluate(
        self, 
        dataloader: Any, 
        eval_cfg: Optional["EvaluatorConfig"] = None, 
        **kwargs
    ) -> Dict[str, float]:
        """Validate/evaluate the model on a target dataset split."""
        if self.evaluator is None:
            cfg = eval_cfg or EvaluatorConfig(**kwargs)
            self.evaluator = self.get_evaluator(cfg)

        return self.evaluator.evaluate(self.model, dataloader)

    def train(
        self, 
        data_path: str, 
        train_cfg: Optional["TrainerConfig"] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """Train the model on a given dataset."""
        if self.trainer is None:
            from __main__ import TrainerConfig
            cfg = train_cfg or TrainerConfig(**kwargs)
            self.trainer = self.get_trainer(cfg)

        self.run_callbacks("on_train_start")
        results = self.trainer.train(data_path)
        self.run_callbacks("on_train_end")
        return results

    # Builder factory methods (to be overridden by subclasses like YOLO)
    @abstractmethod
    def get_predictor(self, cfg: "PredictorConfig") -> BasePredictor:
        """Factory method returning concrete Predictor implementation."""
        raise NotImplementedError

    @abstractmethod
    def get_trainer(self, cfg: "TrainerConfig") -> BaseTrainer:
        """Factory method returning concrete Trainer implementation."""
        raise NotImplementedError

    @abstractmethod
    def get_evaluator(self, cfg: "EvaluatorConfig") -> BaseEvaluator:
        """Factory method returning concrete Evaluator implementation."""
        raise NotImplementedError

    @abstractmethod
    def get_tracker(self, cfg: "TrackerConfig") -> "BaseTracker":
        """Factory method returning concrete Tracker implementation."""
        raise NotImplementedError

    # Callback management
    def add_callback(self, event: str, callback: Callable) -> None:
        """Register callback hook for specific workflow events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def run_callbacks(self, event: str) -> None:
        """Trigger registered callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            callback(self)