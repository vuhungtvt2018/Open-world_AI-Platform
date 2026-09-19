"""
19092026 - KHAI - Create class BaseCamera as base class for Basler and RTSP cameras
"""
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

class BaseCamera(ABC):
    def __init__(self, *args, **kwargs):
        self.lock = threading.Lock()
        self.frame = None
        self.last_frame_at = None
        self.last_error = None
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def read(self):
        """
        19082026 - KIET - Trả bản sao frame mới nhất để nhiều consumer đọc an toàn.
        """
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    @abstractmethod
    def _loop(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def __del__(self):
        """
        19082026 - KIET - Đảm bảo giải phóng Basler camera khi instance bị hủy.
        """
        try:
            self.stop()
        except Exception:
            pass