from typing import Any
import numpy as np

from packages.ai.core.backend import BaseBackend

class FakeBackend(BaseBackend):
    """
    06082026 - NTKIET - Mockup test
    """
    def __init__(self, raw_output: Any = None):
        super().__init__()
        self.raw_output = raw_output

    def load(self):
        self.is_loaded = True

    def set_output(self, raw_output):
        self.raw_output = raw_output

    def predict(self, image: np.ndarray) -> Any:
        """
        06082026 - KIET - Trả về raw output giả lập.
        """

        if not self.is_loaded:
            raise RuntimeError(
                "FakeBackend is not loaded. Call load() first."
            )

        return self.raw_output

    def close(self) -> None:
        """
        06082026 - KIET - Đóng backend và giải phóng raw output giả lập.
        """

        self.raw_output = None
        super().close()