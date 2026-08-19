from typing import Callable, Dict, List, Any

class EventBus:
    """
    Hệ thống phát và nhận sự kiện (Pub/Sub) nội bộ.
    Giúp các module (Camera, AI, DB) tách biệt hoàn toàn và giao tiếp qua sự kiện.
    """
    
    # Định nghĩa các tên sự kiện chuẩn
    EVENT_FRAME_READY = "FRAME_READY"
    EVENT_INFERENCE_DONE = "INFERENCE_DONE"
    EVENT_ANOMALY_DETECTED = "ANOMALY_DETECTED"
    EVENT_ERROR_OCCURRED = "ERROR_OCCURRED"

    def __init__(self):
        # Map: Event Name -> Danh sách các hàm callback
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_name: str, callback: Callable):
        """Đăng ký lắng nghe một sự kiện."""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(callback)

    def publish(self, event_name: str, **kwargs: Any):
        """Phát một sự kiện kèm theo payload (kwargs)."""
        if event_name in self._subscribers:
            for callback in self._subscribers[event_name]:
                try:
                    callback(**kwargs)
                except Exception as e:
                    print(f"[EventBus] Lỗi khi thực thi callback cho sự kiện {event_name}: {e}")

# Biến global mặc định (Singleton pattern) để dùng chung trên toàn hệ thống
default_event_bus = EventBus()
