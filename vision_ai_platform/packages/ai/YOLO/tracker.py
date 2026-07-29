from vision_ai_platform.packages.core.model import BaseTracker
from vision_ai_platform.packages.core.config import TrackerConfig

class YOLOTracker(BaseTracker):
    def __init__(self, cfg: TrackerConfig):
        super().__init__(cfg)