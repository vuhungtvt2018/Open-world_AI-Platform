from vision_ai_platform.packages.core.model import BaseTrainer
from vision_ai_platform.packages.core.config import TrainerConfig

class YOLOTrainer(BaseTrainer):
    def __init__(self, cfg: TrainerConfig, model = None):
        super().__init__(cfg, model)