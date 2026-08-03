from __future__ import annotations

from abc import ABC

from vision_ai_platform import YOLO
from vision_ai_platform.packages.core import WorkflowConfig
from vision_ai_platform.packages.core.workflow import BaseWorkflow
from vision_ai_platform.packages.utils import LOGGER, ops
from vision_ai_platform.packages.utils.check import check_imshow, check_requirements


class Workflow(BaseWorkflow, ABC):
    def __init__(self, cfg: WorkflowConfig):
        check_requirements("shapely>=2.0.0")
        super().__init__(cfg)
        self.logger = LOGGER
        self.logger.info(f"Vision AI Platform Solutions: ✅ {self.cfg}")
        self.model = YOLO(self.cfg.model)
        self.names = self.model.names

        self.env_check = check_imshow(warn=True)
        self.profilers = (
            ops.Profile(device=self.device),  # track
            ops.Profile(device=self.device),  # solution
        )