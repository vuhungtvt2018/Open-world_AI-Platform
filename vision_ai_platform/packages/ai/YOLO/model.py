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
        return YOLOPredictor(cfg, self.model)

    def get_evaluator(self, cfg: "EvaluatorConfig"):
        return YOLOEvaluator(cfg, self.model)

    def get_trainer(self, cfg: "TrainerConfig"):
        return YOLOTrainer(cfg, self.model)

    def get_tracker(self, cfg: "TrackerConfig"):
        return YOLOTracker(cfg)