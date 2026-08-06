from typing import Any

import numpy as np

from packages.ai.core.schemas import (
    BoundingBox,
    InferenceResult,
    Prediction,
    TaskType,
)
from packages.ai.core.task import BaseVisionTask


class DetectionTask(BaseVisionTask):
    """
    06082026 - KIET - Chuyển output Object Detection thành schema chuẩn của AI core.
    """

    def _get_class_name(
        self,
        class_names: dict | list,
        class_id: int,
    ) -> str:
        """
        06082026 - KIET - Lấy tên class tương ứng với class ID từ output model.
        """

        if isinstance(class_names, dict):
            return str(
                class_names.get(class_id, "unknown")
            )

        if isinstance(class_names, list):
            if 0 <= class_id < len(class_names):
                return str(class_names[class_id])

        return "unknown"

    def postprocess(
        self,
        raw_output: Any,
        original_image: np.ndarray,
    ) -> InferenceResult:
        """
        06082026 - KIET - Chuyển raw output của Ultralytics thành InferenceResult.
        """

        image_height, image_width = original_image.shape[:2]

        result = InferenceResult(
            task=TaskType.DETECTION,
            image_width=image_width,
            image_height=image_height,
        )

        if raw_output is None:
            return result

        boxes_output = getattr(
            raw_output,
            "boxes",
            None,
        )

        if boxes_output is None:
            return result

        if len(boxes_output) == 0:
            return result

        boxes = (
            boxes_output.xyxy.detach().cpu().numpy()
        )

        confidences = (
            boxes_output.conf.detach().cpu().numpy()
        )

        class_ids = (
            boxes_output.cls.detach().cpu().numpy().astype(int)
        )

        class_names = getattr(
            raw_output,
            "names",
            {},
        )

        for box, confidence, class_id in zip(
            boxes,
            confidences,
            class_ids,
        ):
            x1, y1, x2, y2 = box.tolist()

            # Giới hạn bbox không vượt ra ngoài kích thước ảnh.
            x1 = max(0.0, min(float(x1), float(image_width)))
            y1 = max(0.0, min(float(y1), float(image_height)))
            x2 = max(0.0, min(float(x2), float(image_width)))
            y2 = max(0.0, min(float(y2), float(image_height)))

            prediction = Prediction(
                class_id=int(class_id),
                class_name=self._get_class_name(
                    class_names=class_names,
                    class_id=int(class_id),
                ),
                confidence=float(confidence),
                bbox=BoundingBox(
                    xmin=x1,
                    ymin=y1,
                    xmax=x2,
                    ymax=y2,
                ),
            )

            result.predictions.append(prediction)

        result.raw_output = raw_output
        return result