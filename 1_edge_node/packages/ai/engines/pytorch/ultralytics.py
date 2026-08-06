from pathlib import Path
from typing import Any
from ultralytics import YOLO
import numpy as np

from packages.ai.core.backend import BaseBackend


class UltralyticsBackend(BaseBackend):
    """
    06082026 - KIET - Thực thi model PyTorch bằng thư viện Ultralytics.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        confidence: float = 0.25,
        image_size: int = 640,
    ) -> None:
        """
        06082026 - KIET - Khởi tạo cấu hình cho Ultralytics backend.
        """

        super().__init__()

        self.checkpoint = checkpoint
        self.device = device
        self.confidence = confidence
        self.image_size = image_size

        self.model: Any = None
        self.model_task: str | None = None

    def load(self) -> None:
        """
        06082026 - KIET - Kiểm tra checkpoint và tải model Ultralytics.
        """

        checkpoint_path = Path(self.checkpoint)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        self.model = YOLO(str(checkpoint_path))
        self.model_task = getattr(self.model, "task", None)
        self.is_loaded = True

    def predict(self, image: np.ndarray) -> Any:
        """
        06082026 - KIET - Chạy inference trên một ảnh và trả raw output.
        """

        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                "UltralyticsBackend is not loaded. Call load() first."
            )

        results = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        if not results:
            return None
        
        return results[0]

    def close(self) -> None:
        """
        06082026 - KIET - Giải phóng model khỏi Ultralytics backend.
        """

        self.model = None
        self.model_task = None

        super().close()