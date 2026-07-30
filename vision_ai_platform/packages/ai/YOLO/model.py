from vision_ai_platform.packages.core.model import BaseModel
from vision_ai_platform.packages.core.config import ModelConfig, PredictorConfig, EvaluatorConfig, TrainerConfig, TrackerConfig
from .predictor import YOLOPredictor
from .evaluator import YOLOEvaluator
from .trainer import YOLOTrainer
from .tracker import YOLOTracker

class YOLOModel(BaseModel):
    def __init__(self, model_cfg: ModelConfig, weights_path = None):
        super().__init__(model_cfg, weights_path)

    def get_predictor(self, cfg: "PredictorConfig"):
        if self.predictor is None:
            return YOLOPredictor(cfg, self.model, self.cfg.imgsz)
        return self.predictor

    def get_evaluator(self, cfg: "EvaluatorConfig"):
        if self.evaluator is None:
            return YOLOEvaluator(cfg, self.model)
        return self.evaluator

    def get_trainer(self, cfg: "TrainerConfig"):
        if self.trainer is None:
            return YOLOTrainer(cfg, self.model, self.device)
        return self.trainer

    def get_tracker(self, cfg: "TrackerConfig"):
        if self.tracker is None:
            return YOLOTracker(cfg)
        return self.tracker