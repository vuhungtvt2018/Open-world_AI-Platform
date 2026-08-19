import numpy as np

# 1. Test Base Classes
from packages.ai.tasks.base_task import BaseVisionTask, InferenceResult, BoundingBox

class DummyYoloDetector(BaseVisionTask):
    def load_model(self):
        print("[DummyYoloDetector] Đã load model thành công.")
        
    def preprocess(self, image):
        print("[DummyYoloDetector] Đang tiền xử lý ảnh...")
        return image
        
    def predict(self, data):
        print("[DummyYoloDetector] Đang chạy mô hình AI...")
        return {"boxes": [[10, 20, 50, 60, 0.9, 1]]}
        
    def postprocess(self, raw, img):
        print("[DummyYoloDetector] Đang hậu xử lý...")
        box = BoundingBox(xmin=10, ymin=20, xmax=50, ymax=60, confidence=0.9, class_id=1, class_name="bulong")
        res = InferenceResult(boxes=[box])
        return res


# 2. Test AutoBackend
from packages.ai.engines.autobackend import AutoBackend

# 3. Test Config Schema
from packages.core.config_schema import PlatformConfig

# 4. Test Event Bus & Callbacks
from packages.workflow.events import default_event_bus, EventBus
from packages.workflow.callbacks import LoggerCallback, AlertCallback


def run_test():
    print("="*50)
    print("KIỂM TRA CÁC MODULE CHUẨN HÓA MỚI (Học từ Ultralytics)")
    print("="*50)

    # Khởi tạo Config
    print("\n1. Kiểm tra Cấu hình (Config Schema)...")
    config_data = {
        "camera": {"camera_id": "0", "fps": 60},
        "ai": {"models": {"yolo_main": {"name": "Yolov8", "weights_path": "yolov8n.pt"}}},
        "database": {"dsn": "sqlite:///test.db"}
    }
    platform_config = PlatformConfig(**config_data)
    print(f"Cấu hình hợp lệ: Camera chạy ở {platform_config.camera.fps} FPS.")

    # Khởi tạo Event Bus và gắn Callbacks
    print("\n2. Kiểm tra Event Bus và Callbacks...")
    logger_cb = LoggerCallback(default_event_bus)
    alert_cb = AlertCallback(default_event_bus)

    # Khởi tạo AutoBackend thử (giả lập file weights có đuôi .onnx)
    print("\n3. Kiểm tra AutoBackend...")
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_path = f.name
    try:
        backend = AutoBackend(weights=temp_path)
        out = backend.predict("fake_image_data")
        print(f"Kết quả backend: {out}")
    finally:
        os.remove(temp_path)

    # Khởi tạo và chạy BaseVisionTask
    print("\n4. Kiểm tra luồng chạy với BaseVisionTask...")
    detector = DummyYoloDetector(model_path="dummy.pt")
    fake_image = np.zeros((100, 100, 3))
    
    # Chạy quy trình
    result = detector.run(fake_image)
    
    # Phát sự kiện để trigger Callbacks
    print("\n5. Phát sự kiện INFERENCE_DONE...")
    default_event_bus.publish(EventBus.EVENT_INFERENCE_DONE, result=result)
    
    if result.boxes[0].confidence > 0.8:
        print("\n6. Phát sự kiện ANOMALY_DETECTED...")
        default_event_bus.publish(EventBus.EVENT_ANOMALY_DETECTED, info="Phát hiện dị vật (bulong lỏng)!")
        
    print("\n[OK] TẤT CẢ CÁC MODULE CHUẨN HÓA CHẠY THÀNH CÔNG!")

if __name__ == "__main__":
    run_test()
