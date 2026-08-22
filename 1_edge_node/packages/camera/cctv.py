import threading
import time
from typing import Optional

import cv2


class RTSP_Threaded_Camera:
    """
    19082026 - KIET - Đọc RTSP/local camera độc lập trên thread và tự kết nối lại khi mất frame.
    """

    def __init__(self, rtsp_url: str | int, width: Optional[int] = None, height: Optional[int] = None):
        """
        19082026 - KIET - Khởi tạo camera từ URL/index riêng của từng camera ID.
        """

        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.cap = self._open_capture()
        if self.cap is None:
            raise RuntimeError(f"Cannot open RTSP: {rtsp_url}")
        self.lock = threading.Lock()
        self.frame = None
        self.last_frame_at = None
        self.last_error = None
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _open_capture(self):
        """
        19082026 - KIET - Mở VideoCapture và áp dụng kích thước frame được cấu hình.
        """

        capture = cv2.VideoCapture(self.rtsp_url)
        if not capture.isOpened():
            capture.release()
            return None

        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        return capture

    def _loop(self):
        """
        19082026 - KIET - Cập nhật frame mới nhất và reconnect RTSP khi kết nối bị gián đoạn.
        """

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open_capture()
                if self.cap is None:
                    self.last_error = f"Cannot reconnect RTSP: {self.rtsp_url}"
                    time.sleep(1.0)
                    continue

            ok, f = self.cap.read()
            if not ok:
                with self.lock:
                    # 19082026 - KIET - Xóa frame cũ để API không báo online khi RTSP đã mất kết nối.
                    self.frame = None
                    self.last_error = "Frame read failed"
                self.cap.release()
                self.cap = None
                time.sleep(0.5)
                continue

            with self.lock:
                self.frame = f
                self.last_frame_at = time.time()
                self.last_error = None

    def read(self):
        """
        19082026 - KIET - Trả bản sao frame mới nhất để nhiều consumer đọc an toàn.
        """

        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        """
        19082026 - KIET - Dừng thread và giải phóng kết nối RTSP/local camera.
        """

        self.running = False
        self.t.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
