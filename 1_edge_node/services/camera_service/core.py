import os
import sys
import threading
from pathlib import Path
from packages.core.config import AppConfig
from packages.camera import RTSP_Threaded_Camera, Basler_Threaded_Camera
from services.database.crud import list_camera_configs

# Resolve ROOT (1_edge_node/) – 2 levels up from services/camera_service/
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cfg = AppConfig.from_yaml(os.path.join(str(ROOT), "config.yaml"))


class CameraManager:
    """
    23082026 - KHAI - Add new attributes and methods, fix set_camera_mode and get_frame based on branch 23082026-KIET-COUNTING-FINAL
    """
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.cameras = {}      # id -> Camera_Instance
        self.input_mode = {}   # id -> "stream", "basler", "folder"
        # 19082026 - KIET - Lưu cấu hình và trạng thái runtime độc lập theo camera ID.
        self.camera_configs = {}
        self.camera_status = {}
        self.camera_lock = threading.RLock()
        self.current_frame = None
        self.current_image_name = "in_memory_image"

    def register_camera_config(self, camera_config: dict) -> None:
        """
        19082026 - KIET - Đăng ký cấu hình camera vào runtime registry.
        """

        camera_id = str(camera_config["camera_id"])
        with self.camera_lock:
            self.camera_configs[camera_id] = dict(camera_config)
            self.camera_status.setdefault(
                camera_id,
                {"status": "disconnected", "error": None},
            )

    def load_saved_camera_configs(self) -> None:
        """
        19082026 - KIET - Load camera từ database và tự connect các camera đã enable.
        """

        for camera_config in list_camera_configs():
            self.register_camera_config(camera_config)
            if not camera_config.get("enabled", False):
                continue

            try:
                self.connect_camera(camera_config["camera_id"])
            except Exception as exc:
                with self.camera_lock:
                    self.camera_status[camera_config["camera_id"]] = {
                        "status": "error",
                        "error": str(exc),
                    }

    def connect_camera(self, camera_id: str) -> None:
        """
        19082026 - KIET - Kết nối đúng RTSP URL hoặc Basler serial theo camera ID.
        """

        camera_id = str(camera_id)
        with self.camera_lock:
            camera_config = self.camera_configs.get(camera_id)

        if camera_config is None:
            raise KeyError(f"Camera config not found: {camera_id}")

        self.disconnect_camera(camera_id, preserve_status=True)
        with self.camera_lock:
            self.camera_status[camera_id] = {
                "status": "connecting",
                "error": None,
            }

        try:
            source_type = camera_config["source_type"]
            width = camera_config.get("width") or self.cfg.CAM_WIDTH
            height = camera_config.get("height") or self.cfg.CAM_HEIGHT

            if source_type == "rtsp":
                source = camera_config.get("source_url")
                if not source:
                    raise ValueError(f"RTSP URL is required: {camera_id}")
                if str(source).isdigit():
                    source = int(source)

                camera = RTSP_Threaded_Camera(
                    source,
                    width=width,
                    height=height,
                )
                input_mode = "stream"
            elif source_type == "basler":
                camera = Basler_Threaded_Camera(
                    serial_number=camera_config.get("serial_number"),
                    width=width,
                    height=height,
                )
                input_mode = "basler"
            else:
                raise ValueError(f"Unsupported camera source type: {source_type}")

            with self.camera_lock:
                self.cameras[camera_id] = camera
                self.input_mode[camera_id] = input_mode
                self.camera_status[camera_id] = {
                    "status": "connecting",
                    "error": None,
                }
        except Exception as exc:
            with self.camera_lock:
                self.camera_status[camera_id] = {
                    "status": "error",
                    "error": str(exc),
                }
            raise

    def disconnect_camera(self, camera_id: str, preserve_status: bool = False) -> None:
        """
        19082026 - KIET - Dừng đúng camera instance mà không ảnh hưởng camera khác.
        """

        camera_id = str(camera_id)
        with self.camera_lock:
            camera = self.cameras.pop(camera_id, None)
            self.input_mode.pop(camera_id, None)

        if camera is not None:
            camera.stop()

        if not preserve_status:
            with self.camera_lock:
                self.camera_status[camera_id] = {
                    "status": "disconnected",
                    "error": None,
                }

    def get_camera_status(self, camera_id: str) -> dict:
        """
        19082026 - KIET - Tổng hợp cấu hình và trạng thái frame thực tế của camera.
        """

        camera_id = str(camera_id)
        with self.camera_lock:
            camera_config = self.camera_configs.get(camera_id)
            camera = self.cameras.get(camera_id)
            runtime_status = dict(
                self.camera_status.get(
                    camera_id,
                    {"status": "disconnected", "error": None},
                )
            )

        if camera_config is None:
            raise KeyError(f"Camera config not found: {camera_id}")

        if camera is not None:
            has_frame = getattr(camera, "frame", None) is not None
            camera_error = getattr(camera, "last_error", None)
            if camera_error:
                runtime_status["status"] = "error"
            elif has_frame:
                runtime_status["status"] = "online"
            else:
                runtime_status["status"] = "connecting"
            runtime_status["error"] = camera_error
            runtime_status["last_frame_at"] = getattr(camera, "last_frame_at", None)
        else:
            runtime_status["last_frame_at"] = None

        return {
            **camera_config,
            **runtime_status,
        }

    def list_camera_statuses(self) -> list[dict]:
        """
        19082026 - KIET - Trả trạng thái tất cả camera cho Live Stream UI.
        """

        with self.camera_lock:
            camera_ids = list(self.camera_configs.keys())
        return [self.get_camera_status(camera_id) for camera_id in camera_ids]

    def set_camera_mode(self, cam_id: str, mode: str):
        """
        19082026 - KIET - Giữ API set-mode cũ và ưu tiên cấu hình camera đã lưu.
        """

        cam_id = str(cam_id)
        if self.input_mode.get(cam_id) == mode:
            return

        if mode == "folder":
            self.disconnect_camera(cam_id)
            self.input_mode[cam_id] = "folder"
            return

        with self.camera_lock:
            stored_config = self.camera_configs.get(cam_id)

        if stored_config is None:
            stored_config = {
                "camera_id": cam_id,
                "name": f"Camera {cam_id}",
                "source_type": "basler" if mode == "basler" else "rtsp",
                "source_url": str(self.cfg.RTSP_URL) if mode == "stream" else None,
                "serial_number": None,
                "assigned_task": None,
                "enabled": False,
                "width": self.cfg.CAM_WIDTH,
                "height": self.cfg.CAM_HEIGHT,
                "fps": None,
            }
            self.register_camera_config(stored_config)

        self.connect_camera(cam_id)

    def get_frame(self, cam_id: str = "default"):
        """
        19082026 - KIET - Lấy frame độc lập theo camera ID hoặc ảnh RAM ở folder mode.
        """

        cam_id = str(cam_id)
        with self.camera_lock:
            input_mode = self.input_mode.get(cam_id)
            camera = self.cameras.get(cam_id)

        if input_mode in ["stream", "basler"]:
            if camera is None:
                return None
            frame = camera.read()
            return frame

        # 22082026 - PHUC - Camera ID cụ thể đã tắt/chưa connect thì KHÔNG fallback sang ảnh RAM,
        # tránh feed và inference hiện nhầm ảnh upload ở Vision Inspection sau khi tắt camera.
        if cam_id != "default" and input_mode is None:
            return None

        # Lấy ảnh trực tiếp từ RAM (không đọc ổ cứng) - chỉ dành cho folder mode hoặc cam default
        if self.current_frame is None:
            return None
        return self.current_frame.copy()

# Singleton instance - shared across the API
camera_manager = CameraManager(cfg)
