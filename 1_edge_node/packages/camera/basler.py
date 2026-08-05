import threading
import time
from typing import Optional

import cv2
from pypylon import pylon


class Basler_Threaded_Camera:
    def __init__(
        self,
        exposure_time_us: Optional[float] = None,
        gain: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        auto_resolution: bool = True,
    ):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
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

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.running and self.camera.IsGrabbing():
            try:
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    image = self.converter.Convert(grab_result)
                    with self.lock:
                        self.frame = image.GetArray()
                grab_result.Release()
            except Exception:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        if self.camera.IsOpen():
            self.camera.Close()
        self.t.join(timeout=1.0)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
