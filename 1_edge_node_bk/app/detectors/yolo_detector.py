import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path: str):
        # Load YOLO segmentation model
        self.model = YOLO(model_path)
        # Warmup để tránh trễ lần đầu
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model.predict(dummy, imgsz=640, verbose=False)

    def predict(self, frame, keypoint_detection=False):
        """
        Chạy YOLO segmentation và keypoint trên frame.
        Trả về đối tượng Results (ultralytics.engine.results.Results).
        """
        results = self.model.predict(
            source=frame,
            imgsz=640,
            verbose=False,
            conf=0.8,
        )
        
        return results if keypoint_detection else results[0]
