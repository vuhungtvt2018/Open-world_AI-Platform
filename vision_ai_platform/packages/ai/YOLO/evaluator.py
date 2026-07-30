from vision_ai_platform.packages.core.model import BaseEvaluator
from vision_ai_platform.packages.core.config import EvaluatorConfig
from vision_ai_platform.packages.utils.metrics import Metric, DetMetrics

from typing import Any, Dict

class YOLOEvaluator(BaseEvaluator):
    def __init__(self, cfg: EvaluatorConfig, model = None):
        super().__init__(cfg, model)

    def init_metrics(self) -> None:
        """Initialize metric tracking containers."""

    def update_metrics(self, preds: Any, targets: Any) -> None:
        """Update metrics state with a batch of predictions and ground truth targets."""

    def compute_metrics(self) -> Dict[str, float]:
        """Compute final evaluation metrics across all processed batches."""