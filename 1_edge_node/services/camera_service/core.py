import os
import sys
from pathlib import Path
from packages.core.config import AppConfig
from packages.camera import RTSP_Threaded_Camera

# Resolve ROOT (1_edge_node/) – 2 levels up from services/camera_service/
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cfg = AppConfig.from_yaml(os.path.join(str(ROOT), "config.yaml"))


class CameraManager:
    def __init__(self, config: AppConfig):
        self.cfg = config
        self.cameras = {}      # id -> Camera_Instance
        self.input_mode = {}   # id -> "stream", "basler", "folder"
        self.current_frame = None
        self.current_image_name = "in_memory_image"

    def set_camera_mode(self, cam_id: str, mode: str):
        if self.input_mode.get(cam_id) == mode:
            return
        if cam_id in self.cameras and self.cameras[cam_id] is not None:
            self.cameras[cam_id].stop()
            self.cameras.pop(cam_id)
        self.input_mode[cam_id] = mode
        if mode == "stream":
            rtsp_val = self.cfg.RTSP_URL
            if str(rtsp_val).isdigit():
                rtsp_val = int(rtsp_val)
            self.cameras[cam_id] = RTSP_Threaded_Camera(rtsp_val, width=self.cfg.CAM_WIDTH, height=self.cfg.CAM_HEIGHT)
        elif mode == "basler":
            from packages.camera.basler import Basler_Threaded_Camera
            self.cameras[cam_id] = Basler_Threaded_Camera(width=self.cfg.CAM_WIDTH, height=self.cfg.CAM_HEIGHT)

    def get_frame(self, cam_id: str = "default"):
        if self.input_mode.get(cam_id) in ["stream", "basler"] and cam_id in self.cameras:
            return self.cameras[cam_id].read()
        elif self.input_mode.get(cam_id) == "none":
            return None
        else:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()

# Singleton instance - shared across the API
camera_manager = CameraManager(cfg)
