from vision_ai_platform.packages.core.model import BaseModel
from vision_ai_platform.packages.core.config import ModelConfig

class YOLOModel(BaseModel):
    def __init__(self, model_cfg: ModelConfig, weights_path = None):
        super().__init__(model_cfg, weights_path)