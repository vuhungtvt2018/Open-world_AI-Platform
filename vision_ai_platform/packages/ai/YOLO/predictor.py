from vision_ai_platform.packages.core.model import BasePredictor
from vision_ai_platform.packages.core.config import PredictorConfig

class YOLOPredictor(BasePredictor):
    def __init__(self, cfg: PredictorConfig, model = None):
        super().__init__(cfg, model)