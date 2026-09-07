import threading
import time
from typing import Optional

from pypylon import pylon


def _get_device_value(device_info, getter_name: str) -> str | None:
    """
    19082026 - KIET - Đọc metadata Basler an toàn giữa các loại USB và GigE camera.
    """

    getter = getattr(device_info, getter_name, None)
    if getter is None:
        return None

    try:
        value = getter()
        return str(value) if value not in (None, "") else None
    except Exception:
        return None


def discover_basler_cameras() -> list[dict]:
    """
    19082026 - KIET - Liệt kê các Basler camera khả dụng để Live Stream UI lựa chọn.
    """

    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    return [
        {
            "device_key": f"basler:{_get_device_value(device, 'GetSerialNumber')}",
            "source_type": "basler",
            "serial_number": _get_device_value(device, "GetSerialNumber"),
            "model_name": _get_device_value(device, "GetModelName"),
            "display_name": (
                _get_device_value(device, "GetFriendlyName")
                or _get_device_value(device, "GetModelName")
                or "Basler Camera"
            ),
            "ip_address": _get_device_value(device, "GetIpAddress"),
            "available": True,
        }
        for device in devices
    ]


class Basler_Threaded_Camera:
    """
    19082026 - KIET - Đọc đúng Basler camera theo serial number trên thread riêng.
    """

    def __init__(
        self,
        serial_number: Optional[str] = None,
        exposure_time_us: Optional[float] = None,
        gain: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        auto_resolution: bool = True,
    ):
        """
        19082026 - KIET - Khởi tạo Basler camera cụ thể để hỗ trợ nhiều thiết bị đồng thời.
        """

        factory = pylon.TlFactory.GetInstance()
        device_info = None

        if serial_number:
            for available_device in factory.EnumerateDevices():
                if _get_device_value(available_device, "GetSerialNumber") == str(serial_number):
                    device_info = available_device
                    break

            if device_info is None:
                raise RuntimeError(f"Basler camera not found: serial={serial_number}")

            device = factory.CreateDevice(device_info)
        else:
            device = factory.CreateFirstDevice()

        self.serial_number = str(serial_number) if serial_number else None
        self.camera = pylon.InstantCamera(device)
        try:
            self.camera.Open()

            if exposure_time_us is not None and self.camera.ExposureTime.IsWritable():
                self.camera.ExposureTime.SetValue(float(exposure_time_us))

            if gain is not None and self.camera.Gain.IsWritable():
                self.camera.Gain.SetValue(float(gain))

            if auto_resolution:
                if self.camera.Width.TrySetToMaximum():
                    pass
                if self.camera.Height.TrySetToMaximum():
                    pass
            elif width is not None and self.camera.Width.IsWritable():
                self.camera.Width.SetValue(int(width))
            elif height is not None and self.camera.Height.IsWritable():
                self.camera.Height.SetValue(int(height))

            if width is not None and self.camera.Width.IsWritable():
                self.camera.Width.SetValue(int(width))
            if height is not None and self.camera.Height.IsWritable():
                self.camera.Height.SetValue(int(height))

            if fps is not None:
                frame_rate_enable = getattr(self.camera, "AcquisitionFrameRateEnable", None)
                frame_rate = getattr(self.camera, "AcquisitionFrameRate", None)
                if frame_rate_enable is not None and frame_rate_enable.IsWritable():
                    frame_rate_enable.SetValue(True)
                if frame_rate is not None and frame_rate.IsWritable():
                    frame_rate.SetValue(float(fps))

            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            if self.camera.IsOpen():
                self.camera.Close()
            raise

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.lock = threading.Lock()
        self.frame = None
        self.last_frame_at = None
        self.last_error = None
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        """
        19082026 - KIET - Cập nhật frame Basler mới nhất cho camera registry.
        """

        while self.running and self.camera.IsGrabbing():
            try:
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    image = self.converter.Convert(grab_result)
                    with self.lock:
                        self.frame = image.GetArray()
                        self.last_frame_at = time.time()
                        self.last_error = None
                grab_result.Release()
            except Exception as exc:
                self.last_error = str(exc)
                time.sleep(0.01)

    def read(self):
        """
        19082026 - KIET - Trả bản sao frame mới nhất của Basler camera.
        """

        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        """
        19082026 - KIET - Dừng grabbing và đóng Basler camera an toàn.
        """

        self.running = False
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        if self.t.is_alive():
            self.t.join(timeout=6.0)
        if self.camera.IsOpen():
            self.camera.Close()

    def __del__(self):
        """
        19082026 - KIET - Đảm bảo giải phóng Basler camera khi instance bị hủy.
        """

        try:
            self.stop()
        except Exception:
            pass
