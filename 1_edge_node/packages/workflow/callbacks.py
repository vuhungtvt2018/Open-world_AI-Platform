from typing import Any
from .events import EventBus

class BaseCallback:
    """
    Lớp cơ sở cho mọi Callback trong hệ thống, tương tự như callback của Ultralytics.
    Các callback này sẽ tự động gắn vào EventBus.
    """
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.register_events()

    def register_events(self):
        """Ghi đè hàm này để đăng ký (subscribe) các sự kiện cụ thể."""
        pass


class LoggerCallback(BaseCallback):
    """Một Callback ví dụ chuyên làm nhiệm vụ ghi log khi có suy luận xong."""
    
    def register_events(self):
        self.event_bus.subscribe(EventBus.EVENT_INFERENCE_DONE, self.on_inference_done)

    def on_inference_done(self, **kwargs: Any):
        result = kwargs.get("result")
        print(f"[LoggerCallback] Nhận được kết quả suy luận. Tìm thấy {len(result.boxes)} vật thể.")


class AlertCallback(BaseCallback):
    """Callback gửi cảnh báo khi phát hiện bất thường."""
    
    def register_events(self):
        self.event_bus.subscribe(EventBus.EVENT_ANOMALY_DETECTED, self.on_anomaly)

    def on_anomaly(self, **kwargs: Any):
        anomaly_info = kwargs.get("info", "Unknown")
        print(f"[AlertCallback] 🚨 CẢNH BÁO BẤT THƯỜNG: {anomaly_info}")
