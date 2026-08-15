# packages/ai/tasks/detection/__init__.py
import numpy as np
from typing import Any, Dict
from ultralytics import YOLO
from packages.core.config import AppConfig
from packages.ai.tasks.base_task import BaseVisionTask, InferenceResult, BoundingBox

class YOLODetector(BaseVisionTask):
    """YOLO-based object detector (Detection, Segmentation, Pose)."""

    def load_model(self) -> None:
        """
        15082026 - KHAI - Add to(device) to model
        """
        self.model = YOLO(self.model_path).to(self.device)
        # Warmup
        self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)

    def preprocess(self, image: np.ndarray) -> Any:
        return image # YOLO handles its own preprocessing internally

    def predict(self, preprocessed_data: Any, keypoint_detection: bool = False, **kwargs) -> Any:
        results = self.model.predict(
            source=preprocessed_data,
            imgsz=640,
            conf=self.config.get("conf_threshold", 0.8),
            verbose=False,
        )
        # For our MVP, we usually just process the first result (one image at a time)
        return results if keypoint_detection else (results[0] if isinstance(results, list) and len(results) > 0 else results)

    def postprocess(self, raw_output: Any, original_image: np.ndarray) -> InferenceResult:
        result = InferenceResult()
        if not raw_output or not hasattr(raw_output, "boxes") or len(raw_output.boxes) == 0:
            return result
        
        # Bounding boxes
        xyxy = raw_output.boxes.xyxy.cpu().numpy().tolist()
        confs = raw_output.boxes.conf.cpu().numpy().tolist()
        cls_ids = raw_output.boxes.cls.cpu().numpy().astype(int).tolist()
        
        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            class_name = raw_output.names.get(cls_id, "unknown")
            result.boxes.append(BoundingBox(
                xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3],
                confidence=conf, class_id=cls_id, class_name=class_name
            ))
            
        # Masks (Segmentation)
        if hasattr(raw_output, "masks") and raw_output.masks is not None:
            # We store the mask data. The caller can reshape it to H, W if needed
            result.masks = raw_output.masks.data.cpu().numpy()
            
        # Keypoints (Pose)
        if hasattr(raw_output, "keypoints") and raw_output.keypoints is not None:
            result.keypoints = raw_output.keypoints.data.cpu().numpy()
            
        result.raw_output = raw_output
        return result