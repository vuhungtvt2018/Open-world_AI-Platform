from vision_ai_platform.packages.core.model import BaseEvaluator
from vision_ai_platform.packages.core.config import EvaluatorConfig

class YOLOEvaluator(BaseEvaluator):
    def __init__(self, cfg: EvaluatorConfig, model = None):
        super().__init__(cfg, model)