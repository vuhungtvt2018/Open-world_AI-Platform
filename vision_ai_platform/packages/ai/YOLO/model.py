from vision_ai_platform.packages.core.model import BaseModel
from vision_ai_platform.packages.core.config import ModelConfig, PredictorConfig, EvaluatorConfig, TrainerConfig, TrackerConfig
from .predictor import YOLOPredictor
from .evaluator import YOLOEvaluator
from .trainer import YOLOTrainer

import torch
import torch.nn as nn

class YOLOModel(BaseModel):
    def __init__(self, model_cfg: ModelConfig, weights_path = None):
        super().__init__(model_cfg, weights_path)

    def _new(self) -> None:
        """Instantiate new model architecture from config."""
        pass
    
    def _load(self, weights_path: str) -> None:
        """Load model state and weights from checkpoint path."""
        self.model.load_state_dict(torch.load(self.weights_path, map_location="cpu", weights_only=True))

    def get_predictor(self):
        if self.predictor is None:
            return YOLOPredictor(self.cfg.predictor, self.model, self.cfg.imgsz, self.cfg.device)
        return self.predictor

    def get_evaluator(self, cfg: "EvaluatorConfig"):
        if self.evaluator is None:
            return YOLOEvaluator(self.cfg.evaluator, self.model, self.cfg.device)
        return self.evaluator

    def get_trainer(self, cfg: "TrainerConfig"):
        if self.trainer is None:
            return YOLOTrainer(self.cfg.trainer, self.model, self.cfg.task, self.cfg.device)
        return self.trainer