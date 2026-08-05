from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

import numpy as np

from packages.ai.core.backend import BaseBackend
from packages.ai.core.schemas import InferenceResult


class BaseVisionTask(ABC):
    """
    05082026 - KIET - Điều phối preprocess, inference và postprocess cho một AI task.
    """

    def __init__(self, backend: BaseBackend) -> None:
        """
        05082026 - KIET - Khởi tạo task với backend được cung cấp.
        """

        self.backend = backend

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        05082026 - KIET - Kiểm tra và tiền xử lý ảnh trước khi inference.
        """

        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy.ndarray")

        if image.size == 0:
            raise ValueError("Input image is empty")

        if image.ndim not in (2, 3):
            raise ValueError(
                "Input image must have 2 or 3 dimensions"
            )

        return image

    @abstractmethod
    def postprocess(
        self,
        raw_output: Any,
        original_image: np.ndarray,
    ) -> InferenceResult:
        """
        05082026 - KIET - Chuyển raw output thành InferenceResult chuẩn.
        """

        raise NotImplementedError

    def run(self, image: np.ndarray) -> InferenceResult:
        """
        05082026 - KIET - Thực hiện đầy đủ một lượt inference trên ảnh đầu vào.
        """

        started_at = perf_counter()

        model_input = self.preprocess(image)
        raw_output = self.backend.predict(model_input)

        result = self.postprocess(
            raw_output=raw_output,
            original_image=image,
        )

        result.processing_time_ms = (
            perf_counter() - started_at
        ) * 1000.0

        return result