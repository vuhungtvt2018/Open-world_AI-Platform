# apps/edge-agent/web_backend.py
from pathlib import Path
import sys
import os

# Resolve ROOT directory (2 levels up from edge-agent: edge)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import cv2
import time
import numpy as np
import shutil
import psutil
import threading
from collections import Counter
from typing import Literal
from fastapi import FastAPI, Response, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import logic (New DAG Architecture)
from packages.core.config import AppConfig
from packages.workflow.pipeline import Pipeline
from packages.camera import (
    RTSP_Threaded_Camera,
    Basler_Threaded_Camera,
    discover_basler_cameras,
)
from packages.utils.utils import (ts, union_box, pad_and_clip_box, 
                     arrow_angle, rotate_image, transform_points, need_clean, clean_data)
from packages.utils.visualize import concat_anomaly_crops
from services.database.session import init_db, SessionLocal
from services.database.crud import (
    create_inspection_record,
    get_camera_config,
    list_camera_configs,
    update_camera_config,
    upsert_camera_config,
)
from packages.utils.cleaner import run_cleaner_daemon
from services.database.models import InspectionRecord
from packages.ai.engines.pytorch import UltralyticsBackend
from packages.ai.tasks.detection import DetectionTask

app = FastAPI(title="Visual Inspection AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = AppConfig.from_yaml(os.path.join(str(ROOT), "config.yaml"))

UPLOAD_DIR = os.path.join(str(ROOT), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class CameraConfigRequest(BaseModel):
    """
    19082026 - KIET - Chuẩn hóa cấu hình RTSP/Basler gửi từ Live Stream UI.
    """

    camera_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    name: str = Field(min_length=1, max_length=150)
    source_type: Literal["rtsp", "basler"]
    source_url: str | None = None
    serial_number: str | None = None
    assigned_task: Literal["detection", "inspection"] | None = None
    enabled: bool = False
    width: int | None = None
    height: int | None = None
    fps: float | None = None


class CameraAssignmentRequest(BaseModel):
    """
    19082026 - KIET - Chuẩn hóa task được gán cho camera nhưng giữ API task riêng biệt.
    """

    assigned_task: Literal["detection", "inspection"] | None = None


class RtspTestRequest(BaseModel):
    """
    19082026 - KIET - Chuẩn hóa RTSP URL cần kiểm tra trước khi lưu camera.
    """

    source_url: str = Field(min_length=1, max_length=500)

def seed_models():
    from services.database.models import AIModel
    db = SessionLocal()
    try:
        if db.query(AIModel).count() == 0:
            models = [
                AIModel(
                    id="m-det-01", name=os.path.basename(cfg.MODEL_PATH),
                    type="Detection", format=cfg.MODEL_PATH.split('.')[-1].upper(),
                    version="1.0.0", map_acc=91.4, status="PRODUCTION", file_path=cfg.MODEL_PATH, speed_ms="42ms"
                ),
                AIModel(
                    id="m-anom-01", name=os.path.basename(cfg.ANOMALY_MODEL_PATH),
                    type="Anomaly", format=cfg.ANOMALY_MODEL_PATH.split('.')[-1].upper(),
                    version="1.0.0", map_acc=89.5, status="PRODUCTION", file_path=cfg.ANOMALY_MODEL_PATH, speed_ms="20ms"
                ),
                AIModel(
                    id="m-kpt-01", name=os.path.basename(cfg.KEYPOINTS_MODEL_PATH),
                    type="Keypoints", format=cfg.KEYPOINTS_MODEL_PATH.split('.')[-1].upper(),
                    version="1.0.0", map_acc=95.0, status="PRODUCTION", file_path=cfg.KEYPOINTS_MODEL_PATH, speed_ms="12ms"
                )
            ]
            db.add_all(models)
            db.commit()
    except Exception as e:
        print(f"Failed to seed models: {e}")
    finally:
        db.close()

@app.on_event("startup")
def start_background_tasks():
    init_db()
    seed_models()
    # 19082026 - KIET - Khôi phục camera đã enable sau khi database sẵn sàng.
    inference_engine.load_saved_camera_configs()
    run_cleaner_daemon(
        captures_dir=cfg.CAPTURE_DIR,
        days_to_keep=cfg.DISK_CLEANUP_DAYS,
        interval_hours=cfg.DISK_CLEANUP_INTERVAL_HOURS
    )
    
    # Đăng ký EventBus callback cho Database
    from packages.workflow.events import default_event_bus, EventBus
    def save_db_callback(record: dict):
        try:
            create_inspection_record(record)
        except Exception as e:
            print(f"[ERROR] Failed to insert DB record via EventBus: {e}")
            
    default_event_bus.subscribe(EventBus.EVENT_INFERENCE_DONE, save_db_callback)

app.mount("/captures", StaticFiles(directory=cfg.CAPTURE_DIR), name="captures")

SYNC_DEFECT_IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sync_defect_image"))
os.makedirs(SYNC_DEFECT_IMAGE_DIR, exist_ok=True)
app.mount("/sync_images", StaticFiles(directory=SYNC_DEFECT_IMAGE_DIR), name="sync_images")

class WebInference:
    def __init__(self, config):
        """
        11082026 - KIET - Khởi tạo inspection pipeline, Object Detection và camera inputs.
        """

        self.cfg = config
        self.pipeline = Pipeline(config)

        # 11082026 - KIET - Resolve checkpoint OD theo project root khi config dùng relative path.
        od_model_path = self.cfg.MODEL_PATH
        if not os.path.isabs(od_model_path):
            od_model_path = os.path.join(str(ROOT), od_model_path)

        # 11082026 - KIET - Khởi tạo và load Object Detection backend một lần.
        self.od_backend = UltralyticsBackend(
            checkpoint=od_model_path,
            device="cpu",
            confidence=0.6,
            image_size=640,
        )
        self.od_backend.load()

        # 11082026 - KIET - Nối Ultralytics backend với DetectionTask chuẩn hóa output.
        self.od_task = DetectionTask(backend=self.od_backend)

        # 11082026 - KIET - Bảo vệ model trước nhiều request inference đồng thời.
        self.od_lock = threading.Lock()

        # 19082026 - KIET - Bảo vệ Inspection pipeline khi nhiều camera gọi cùng task.
        self.inspection_lock = threading.Lock()
        
        # Khởi tạo Multi-Camera Stream
        self.cameras = {} # id -> Camera_Instance
        self.input_mode = {} # id -> "stream", "basler", "folder"
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

        if input_mode not in (None, "folder") and cam_id != "default":
            return None

        # Lấy ảnh trực tiếp từ RAM (không đọc ổ cứng)
        if self.current_frame is None:
            return None
        return self.current_frame.copy()

    def _count_predictions_by_class(self, predictions) -> dict[str, int]:
        """
        11082026 - KIET - Đếm số prediction theo từng class trong frame hiện tại.
        """

        return dict(
            Counter(
                prediction.class_name or "unknown"
                for prediction in predictions
            )
        )

    def _build_detection_objects(self, predictions) -> list[dict]:
        """
        11082026 - KIET - Chuyển Prediction thành payload tương thích database cũ.
        """

        objects = []

        for index, prediction in enumerate(predictions):
            bbox = prediction.bbox
            if bbox is None:
                continue

            objects.append({
                "index": index,
                "bbox": [
                    float(bbox.xmin),
                    float(bbox.ymin),
                    float(bbox.xmax),
                    float(bbox.ymax),
                ],
                "score": float(prediction.confidence),
                "class_id": int(prediction.class_id),
                "class_name": prediction.class_name,
                # 11082026 - KIET - Giữ các field inspection cũ để CRUD tương thích.
                "is_ng": False,
                "overlap_ratio": 0.0,
                "crop": "",
                "anomalies": [],
            })

        return objects

    def run_detection(self, cam_id: str = "default") -> dict:
        """
        11082026 - KIET - Chạy Object Detection và counting theo class trên một frame.
        """

        frame = self.get_frame(cam_id)
        if frame is None:
            return {"status": "error", "message": "No input frame available"}

        if self.cfg.FLIP_VERTICAL:
            frame = cv2.flip(frame, 0)

        timestamp = ts()

        # 11082026 - KIET - Chạy model tuần tự để tránh xung đột giữa các request.
        with self.od_lock:
            result = self.od_task.run(frame)

        objects = self._build_detection_objects(result.predictions)
        counts_by_class = self._count_predictions_by_class(result.predictions)
        total_objects = len(objects)

        # 11082026 - KIET - Tái sử dụng thư mục original của inspection session cũ.
        image_name = f"IMG_{timestamp}.jpg"
        image_path = os.path.join(self.pipeline.dir_original, image_name)
        cv2.imwrite(image_path, frame)

        record = {
            "timestamp": timestamp,
            "image": image_name,
            "task_type": "detection",
            "camera_id": cam_id,
            "latency_ms": round(result.processing_time_ms, 2),
            "total_objects": total_objects,
            "ng_detected": False,
            "objects": objects,
        }

        from packages.workflow.events import default_event_bus, EventBus

        # 11082026 - KIET - Tái sử dụng EventBus và callback lưu database hiện tại.
        default_event_bus.publish(EventBus.EVENT_INFERENCE_DONE, record=record)

        session_url = (
            f"/captures/{self.cfg.PRODUCT_NAME}/sessions/"
            f"{os.path.basename(self.pipeline.session_root)}"
        )

        return {
            "status": "success",
            "timestamp": timestamp,
            "task": "detection",
            "camera_id": cam_id,
            # 12082026 - KIET - Trả kích thước ảnh để frontend scale bounding box chính xác.
            "image_width": result.image_width,
            "image_height": result.image_height,
            "metrics": {
                "latency_ms": round(result.processing_time_ms, 2),
                "total_objects": total_objects,
                "counts_by_class": counts_by_class,
            },
            "original_image_url": f"{session_url}/original/{image_name}",
            "objects": objects,
        }

    def run_inspect(self, cam_id: str = "default"):
        frame = self.get_frame(cam_id)
        if frame is None:
            return {"status": "error", "message": "No input frame available"}
            
        if self.cfg.FLIP_VERTICAL:
            frame = cv2.flip(frame, 0)
            
        H, W = frame.shape[:2]
        name = ts()
        insp_time_str = self.pipeline._format_inspection_time(name)
        
        # Lưu ảnh gốc
        orig_path = os.path.join(self.pipeline.dir_original, f"IMG_{name}.jpg")
        cv2.imwrite(orig_path, frame)
        
        start_time = time.time()
        
        # 1. Keypoint Detection
        keypoint_objects = []
        result_keypoints = self.pipeline.keypoint_detection.predict(frame, self.cfg.KEYPOINT_DETECTION)
        for result in result_keypoints:
            raw_boxes = result.boxes.xywh.cpu().numpy().astype(int)
            raw_keypoints = result.keypoints.data.cpu().numpy()
            filtered_keypoints, filtered_box = [], []
            for box, group in zip(raw_boxes, raw_keypoints):
                pt1, pt2 = group[0], group[1]
                if (len(pt1) < 3 or len(pt2) < 3 or pt1[-1] < self.cfg.KEYPOINTS_SCORE_THRESHOLD or pt2[-1] < self.cfg.KEYPOINTS_SCORE_THRESHOLD):
                    continue
                filtered_keypoints.append([[int(pt1[0]), int(pt1[1]), pt1[-1]], [int(pt2[0]), int(pt2[1]), pt2[-1]]])
                x, y, w, h = box
                filtered_box.append([int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)])
            keypoint_objects.append({"name": result.names.get('bolt', 'unknown'), "keypoints": filtered_keypoints, "boxes": filtered_box})
        
        if need_clean(keypoint_objects):
            keypoint_objects = clean_data(keypoint_objects)

        # 2. Segmentation
        r = self.pipeline.detector.predict(frame)
        segment_objects = []
        seg_boxes = []
        if r is not None and hasattr(r, "boxes") and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            seg_boxes = xyxy.tolist()
            masks = (r.masks.data.cpu().numpy() > 0.5).astype(np.uint8)
            if masks.shape[1] != H or masks.shape[2] != W:
                up = np.zeros((masks.shape[0], H, W), dtype=np.uint8)
                for i in range(masks.shape[0]):
                    up[i] = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_NEAREST)
                masks = up
            for box, mask in zip(seg_boxes, masks):
                segment_objects.append({'box': box, 'mask': mask})
        else:
            masks = np.zeros((0, H, W), dtype=np.uint8)

        # 3. Mapping & Alignment
        mapping_object = self.pipeline.match_objects(keypoint_objects, segment_objects)
        if mapping_object:
            for m in mapping_object:
                m["union_box"] = temp = union_box(m["kp_box"], m["seg_box"])
                x1p, y1p, x2p, y2p = pad_and_clip_box(temp[0], temp[1], temp[2], temp[3], self.cfg.PAD_RATIO, W, H, square=True)
                m["union_box_pad"] = [x1p, y1p, x2p, y2p]

        # 4. Rotation & Anomaly Detection
        ng_any = False
        per_objects = []
        if mapping_object:
            for i, obj in enumerate(mapping_object):
                landmark_1 = obj['keypoints'][1][:2]
                landmark_2 = obj['keypoints'][0][:2]
                x1, y1, x2, y2 = obj['union_box_pad']
                mask = obj['mask']
                
                # Tính góc và xoay
                angle = arrow_angle(landmark_1, landmark_2)
                rotated_frame, M = rotate_image(frame, 90-angle)
                rotated_mask, _ = rotate_image(mask, 90-angle)

                # Xoay box và crop
                box_pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                rotated_box = transform_points(box_pts, M).astype(int)
                x_min, y_min = rotated_box[:,0].min(), rotated_box[:,1].min()
                x_max, y_max = rotated_box[:,0].max(), rotated_box[:,1].max()
                
                center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                max_val = max(abs(x_max - x_min), abs(y_max - y_min))
                
                cx1, cy1, cx2, cy2 = center_x - max_val // 2, center_y - max_val // 2, center_x + max_val // 2, center_y + max_val // 2
                crop = rotated_frame[max(0,cy1):cy2, max(0,cx1):cx2]
                crop_mask = rotated_mask[max(0,cy1):cy2, max(0,cx1):cx2]

                if crop.size == 0 or crop_mask.size == 0:
                    continue

                # Lưu crop
                crop_name = f"CROP_{name}_{i}.jpg"
                crop_path = os.path.join(self.pipeline.dir_crops, crop_name)
                cv2.imwrite(crop_path, crop)

                # Anomaly Detection
                out = self.pipeline.anomaly.run_on_crop(crop, crop_mask)
                is_ng = bool(out.get("is_ng", False))
                
                # Visualization tiles
                hm_tile = None
                if out.get("hm_disp") is not None:
                    hm_tile = out["hm_disp"].copy()
                    label_small = f"obj{i} {'NG' if is_ng else 'OK'}"
                    color = (0, 0, 255) if is_ng else (0, 200, 0)
                    cv2.putText(hm_tile, label_small, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                anomalies_full = []
                cls_labels_i = []
                
                if is_ng:
                    ng_any = True
                    # Lưu heatmap cho NG
                    cv2.imwrite(os.path.join(self.pipeline.dir_ng, f"NG_HEAT_{name}_{i}.jpg"), out["hm_disp"])
                    
                    for ak, a in enumerate(out.get("anomalies", [])):
                        cx1, cy1, cx2, cy2 = a["bbox_in_object_crop"]
                        anomalies_full.append({
                            "k": ak,
                            "bbox_full": [int(cx1), int(cy1), int(cx2), int(cy2)],
                            "bbox_in_object_crop": [int(cx1), int(cy1), int(cx2), int(cy2)],
                        })
                        anom_crop = crop[cy1:cy2, cx1:cx2]
                        if anom_crop.size != 0:
                            anom_name = f"ANOM_{name}_obj{i}_{ak}.jpg"
                            anom_path = os.path.join(self.pipeline.dir_anom_crops, anom_name)
                            cv2.imwrite(anom_path, anom_crop)
                            
                            # Gọi API phân loại của pipeline
                            cls_res = self.pipeline._classify_defect(anom_path)
                            if cls_res is not None:
                                label = cls_res.get("label", "NG")
                                sim = float(cls_res.get("similarity", 0.0))
                                anomalies_full[-1]["cls_label"] = label
                                anomalies_full[-1]["cls_similarity"] = sim
                                cls_labels_i.append(label)
                            else:
                                anomalies_full[-1]["cls_label"] = "NG"
                                anomalies_full[-1]["cls_similarity"] = None


                # Draw on crop for visualization
                crop_labeled = crop.copy()
                if is_ng and len(anomalies_full) > 0:
                    for idx_disp, a in enumerate(anomalies_full, start=1):
                        cx1, cy1, cx2, cy2 = a["bbox_in_object_crop"]
                        cv2.rectangle(crop_labeled, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
                        
                        label_text = f"NG{idx_disp}"
                        sim_val = a.get("cls_similarity", None)
                        if self.cfg.DEFECT_CLS_ENABLE and a.get("cls_label") and a["cls_label"] != "NG":
                            if sim_val is not None:
                                if float(sim_val) >= float(self.cfg.DEFECT_CLS_SIM_THRESHOLD):
                                    label_text = f"{a['cls_label']} ({float(sim_val):.2f})"
                                else:
                                    label_text = f"{a['cls_label']}? ({float(sim_val):.2f})"
                            else:
                                label_text = a['cls_label']
                                
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        tx1, ty1 = cx1, max(0, cy1 - th - 6)
                        tx2, ty2 = cx1 + tw + 8, cy1
                        cv2.rectangle(crop_labeled, (tx1, ty1), (tx2, ty2), (0, 0, 255), -1)
                        cv2.putText(crop_labeled, label_text, (cx1 + 3, cy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                per_objects.append({
                    "index": i,
                    "bbox": [0, 0, crop.shape[1], crop.shape[0]],
                    "score": round(float(out.get("score", 0.0)), 4),
                    "is_ng": is_ng,
                    "overlap_ratio": round(float(out.get("overlap_ratio", 0.0)), 4) if crop.size != 0 else 0.0,
                    "crop": crop_name,
                    "crop_url": f"/captures/{self.cfg.PRODUCT_NAME}/sessions/{os.path.basename(self.pipeline.session_root)}/crops/{crop_name}",
                    "hm_tile": hm_tile,
                    "crop_labeled": crop_labeled,
                    "anomalies": anomalies_full
                })

        # 1. Heatmap panel
        hm_tiles = [obj["hm_tile"] for obj in per_objects if obj["hm_tile"] is not None]
        hm_panel = concat_anomaly_crops(hm_tiles, target_h=400) if hm_tiles else np.zeros((400, 400, 3), dtype=np.uint8)
        hm_name = f"HM_{name}.jpg"
        hm_path = os.path.join(self.pipeline.session_root, hm_name)
        cv2.imwrite(hm_path, hm_panel)
        
        # 2. Labeled crops panel
        labeled_crops = [obj["crop_labeled"] for obj in per_objects]
        crops_panel = concat_anomaly_crops(labeled_crops, target_h=400) if labeled_crops else np.zeros((400, 400, 3), dtype=np.uint8)
        crops_name = f"CROPS_VIS_{name}.jpg"
        crops_vis_path = os.path.join(self.pipeline.session_root, crops_name)
        cv2.imwrite(crops_vis_path, crops_panel)

        # 3. Overall output
        overall_vis = frame.copy()
        for i, obj in enumerate(per_objects):
            pass 
        
        overall_name = f"OVERALL_{name}.jpg"
        overall_path = os.path.join(self.pipeline.session_root, overall_name)
        cv2.imwrite(overall_path, overall_vis)
        
        session_url = f"/captures/{self.cfg.PRODUCT_NAME}/sessions/{os.path.basename(self.pipeline.session_root)}"
        vis_urls = {
            "heatmap": f"{session_url}/{hm_name}",
            "crops": f"{session_url}/{crops_name}",
            "overall": f"{session_url}/{overall_name}"
        }

        clean_objects = []
        max_score = 0.0
        ng_count = 0
        for obj in per_objects:
            clean_obj = {k: v for k, v in obj.items() if k not in ["hm_tile", "crop_labeled"]}
            clean_objects.append(clean_obj)
            max_score = max(max_score, obj["score"])
            if obj["is_ng"]:
                ng_count += 1

        latency_ms = (time.time() - start_time) * 1000

        # 5. Summary Record
        record = {
            "timestamp": name,
            "image": os.path.basename(orig_path),
            "ng_detected": ng_any,
            "latency_ms": round(latency_ms, 2),
            "total_objects": len(per_objects),
            "ng_count": ng_count,
            "max_score": round(max_score, 4),
            "objects": clean_objects
        }
        
        # Publish event thay vì gọi DB trực tiếp (Event-driven Architecture)
        from packages.workflow.events import default_event_bus, EventBus
        default_event_bus.publish(EventBus.EVENT_INFERENCE_DONE, record=record)

        return {
            "status": "success",
            "timestamp": name,
            "ng_detected": ng_any,
            "metrics": {
                "latency_ms": round(latency_ms, 2),
                "total_objects": len(per_objects),
                "ng_count": ng_count,
                "max_score": round(max_score, 4)
            },
            "vis_urls": vis_urls,
            "original_image_url": f"/captures/{self.cfg.PRODUCT_NAME}/sessions/{os.path.basename(self.pipeline.session_root)}/original/IMG_{name}.jpg",
            "objects": clean_objects
        }

inference_engine = WebInference(cfg)


@app.on_event("shutdown")
def shutdown_inference_engine() -> None:
    """
    11082026 - KIET - Giải phóng OD backend và camera khi Web Backend dừng.
    """

    inference_engine.od_backend.close()

    for camera_id in list(inference_engine.cameras.keys()):
        inference_engine.disconnect_camera(camera_id)

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Đọc luồng byte trực tiếp vào RAM
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Lưu vào biến RAM, KHÔNG ghi xuống đĩa
    inference_engine.current_frame = frame
    
    # Dùng timestamp làm tên ảo để trình duyệt không bị cache ảnh cũ
    dynamic_name = f"in_memory_image_{int(time.time()*1000)}"
    inference_engine.current_image_name = dynamic_name
    
    # Trả về danh sách chứa ảnh trên RAM để UI hiển thị được thumbnail
    return {"filename": dynamic_name, "images": [dynamic_name]}

@app.get("/video-feed")
async def video_feed(cam: str = "default", fps: float = 15.0):
    """
    19082026 - KIET - Stream đúng camera ID và giới hạn FPS để multi-camera không chiếm CPU quá mức.
    """

    frame_interval = 1.0 / min(max(fps, 1.0), 30.0)

    def gen_frames():
        while True:
            frame = inference_engine.get_frame(cam)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(frame_interval)
            else:
                time.sleep(0.03)
                
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/images")
async def list_images():
    # Trả về ảnh duy nhất đang nằm trên RAM
    if inference_engine.current_frame is not None:
        return {"images": [inference_engine.current_image_name]}
    return {"images": []}

@app.get("/images/{filename}")
async def get_image(filename: str):
    if filename.startswith("in_memory_image") and inference_engine.current_frame is not None:
        ret, buffer = cv2.imencode('.jpg', inference_engine.current_frame)
        if ret:
            return Response(content=buffer.tobytes(), media_type="image/jpeg")
    return Response(status_code=404)

@app.post("/set-image/{filename}")
async def set_image(filename: str):
    return {"message": "In-memory mode does not support switching past uploads"}

@app.post("/set-mode/{mode}")
async def set_mode(mode: str, cam: str = "default"):
    try:
        inference_engine.set_camera_mode(cam, mode)
        return {"message": f"Input mode for {cam} set to {mode}"}
    except Exception as e:
        from fastapi import HTTPException
        print(f"Error setting mode: {e}")
        raise HTTPException(status_code=500, detail=f"Camera Error: {str(e)}")


@app.get("/api/cameras")
def list_cameras():
    """
    19082026 - KIET - Trả danh sách camera đã cấu hình kèm trạng thái runtime.
    """

    return {"cameras": inference_engine.list_camera_statuses()}


@app.get("/api/cameras/discover/basler")
def discover_available_basler_cameras():
    """
    19082026 - KIET - Discovery Basler camera đang kết nối với Edge node.
    """

    try:
        return {"cameras": discover_basler_cameras()}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Basler discovery failed: {exc}",
        ) from exc


@app.post("/api/cameras/test-rtsp")
def test_rtsp_camera(request: RtspTestRequest):
    """
    19082026 - KIET - Kiểm tra RTSP/local source và đọc thử một frame trước khi lưu.
    """

    source: str | int = request.source_url
    if request.source_url.isdigit():
        source = int(request.source_url)

    capture = cv2.VideoCapture()
    try:
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

        if not capture.open(source):
            raise HTTPException(status_code=400, detail="Cannot open RTSP source")

        ok, frame = capture.read()
        if not ok or frame is None:
            raise HTTPException(status_code=400, detail="RTSP opened but no frame received")

        height, width = frame.shape[:2]
        fps = capture.get(cv2.CAP_PROP_FPS)
        return {
            "available": True,
            "width": width,
            "height": height,
            "fps": round(float(fps), 2) if fps > 0 else None,
        }
    finally:
        capture.release()


@app.post("/api/cameras")
def save_camera_config(request: CameraConfigRequest):
    """
    19082026 - KIET - Lưu camera từ Live Stream UI và đăng ký vào runtime registry.
    """

    camera_data = request.model_dump()
    if request.source_type == "rtsp" and not request.source_url:
        raise HTTPException(status_code=400, detail="RTSP URL is required")
    if request.source_type == "basler" and not request.serial_number:
        raise HTTPException(status_code=400, detail="Basler serial number is required")

    try:
        saved_camera = upsert_camera_config(camera_data)
        inference_engine.register_camera_config(saved_camera)
        if saved_camera["enabled"]:
            inference_engine.connect_camera(saved_camera["camera_id"])
        else:
            # 19082026 - KIET - Dừng instance cũ nếu camera được cập nhật về trạng thái disable.
            inference_engine.disconnect_camera(saved_camera["camera_id"])
        return inference_engine.get_camera_status(saved_camera["camera_id"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/cameras/{camera_id}/connect")
def connect_configured_camera(camera_id: str):
    """
    19082026 - KIET - Connect một camera đã lưu mà không ảnh hưởng camera khác.
    """

    camera_config = get_camera_config(camera_id)
    if camera_config is None:
        raise HTTPException(status_code=404, detail="Camera config not found")

    try:
        inference_engine.register_camera_config(camera_config)
        inference_engine.connect_camera(camera_id)
        updated_camera = update_camera_config(camera_id, {"enabled": True})
        if updated_camera is not None:
            inference_engine.register_camera_config(updated_camera)
        return inference_engine.get_camera_status(camera_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/cameras/{camera_id}/disconnect")
def disconnect_configured_camera(camera_id: str):
    """
    19082026 - KIET - Disconnect một camera và giữ nguyên cấu hình để dùng lại.
    """

    if get_camera_config(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera config not found")

    inference_engine.disconnect_camera(camera_id)
    updated_camera = update_camera_config(camera_id, {"enabled": False})
    if updated_camera is not None:
        inference_engine.register_camera_config(updated_camera)
    return inference_engine.get_camera_status(camera_id)


@app.patch("/api/cameras/{camera_id}/assignment")
def assign_camera_task(camera_id: str, request: CameraAssignmentRequest):
    """
    19082026 - KIET - Gán Detection hoặc Inspection cho camera mà không đổi task API.
    """

    updated_camera = update_camera_config(
        camera_id,
        {"assigned_task": request.assigned_task},
    )
    if updated_camera is None:
        raise HTTPException(status_code=404, detail="Camera config not found")

    inference_engine.register_camera_config(updated_camera)
    return inference_engine.get_camera_status(camera_id)


@app.get("/api/cameras/{camera_id}/status")
def get_configured_camera_status(camera_id: str):
    """
    19082026 - KIET - Trả trạng thái online và lỗi gần nhất của một camera.
    """

    try:
        return inference_engine.get_camera_status(camera_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/inspect")
def trigger_inspect(cam: str = "default"):
    """
    19082026 - KIET - Chạy Inspection trong threadpool và khóa riêng theo task.
    """

    with inference_engine.inspection_lock:
        return inference_engine.run_inspect(cam)


@app.post("/detect")
def trigger_detection(cam: str = "default"):
    """
    11082026 - KIET - Chạy Object Detection và trả số lượng theo class.
    """

    return inference_engine.run_detection(cam)

@app.get("/history")
async def get_history():
    import ast
    import json
    db = SessionLocal()
    try:
        db.commit() # Force close any lingering transaction to get fresh data
        records = db.query(InspectionRecord).order_by(InspectionRecord.id.asc()).all()
        results = []
        for r in records:
            objs = []
            for o in r.objects:
                anoms = []
                for a in o.anomalies:
                    try:
                        bf = ast.literal_eval(a.bbox_full) if a.bbox_full else []
                    except: bf = []
                    try:
                        bi = ast.literal_eval(a.bbox_in_object_crop) if a.bbox_in_object_crop else []
                    except: bi = []
                    anoms.append({
                        "bbox_full": bf,
                        "bbox_in_object_crop": bi,
                        "cls_label": a.defect_class,
                        "cls_similarity": a.similarity
                    })
                try:
                    # 11082026 - KIET - Ưu tiên JSON mới và fallback dữ liệu str() cũ.
                    obbox = json.loads(o.bbox) if o.bbox else []
                except (json.JSONDecodeError, TypeError):
                    try:
                        obbox = ast.literal_eval(o.bbox) if o.bbox else []
                    except (ValueError, SyntaxError):
                        obbox = []
                objs.append({
                    "index": o.object_index,
                    "bbox": obbox,
                    "score": o.score,
                    # 11082026 - KIET - Trả metadata OD để frontend hiển thị class counting.
                    "class_id": o.class_id,
                    "class_name": o.class_name,
                    "is_ng": o.is_ng,
                    "overlap_ratio": o.overlap_ratio,
                    "crop": o.crop_image,
                    "anomalies": anoms,
                    "anomaly_count": len(anoms),
                    "anomaly_bboxes_sample": [a["bbox_full"] for a in anoms[:3]]
                })

            # 11082026 - KIET - Tính lại count theo class từ các object đã lưu.
            counts_by_class = {}
            if r.task_type == "detection":
                counts_by_class = dict(
                    Counter(
                        obj["class_name"] or "unknown"
                        for obj in objs
                    )
                )

            results.append({
                "timestamp": r.timestamp,
                "image": r.original_image,
                "ng_detected": r.ng_detected,
                "latency_ms": r.latency_ms,
                # 11082026 - KIET - Trả metadata OD nhưng vẫn giữ response inspection cũ.
                "task_type": r.task_type,
                "camera_id": r.camera_id,
                "total_objects": r.total_objects if r.task_type == "detection" else len(objs),
                "counts_by_class": counts_by_class,
                "ng_count": sum(1 for ob in objs if ob["is_ng"]),
                "max_score": max([ob["score"] for ob in objs]) if objs else 0.0,
                "objects": objs
            })
        return {"results": results}
    except Exception as e:
        print(f"Error fetching history: {e}")
        return {"results": []}
    finally:
        db.close()

import glob
from fastapi.responses import FileResponse, Response

@app.get("/api/image/{image_name}")
async def get_image(image_name: str):
    pattern = os.path.join(cfg.CAPTURE_DIR, cfg.PRODUCT_NAME, "sessions", "*", "original", image_name)
    matches = glob.glob(pattern)
    if matches:
        return FileResponse(matches[0])
    return Response(status_code=404)

@app.get("/api/result_image/{image_name}")
async def get_result_image(image_name: str):
    # Lấy tên file gốc (IMG_XXX.jpg) và chuyển thành OVERALL_XXX.jpg
    timestamp = image_name.replace("IMG_", "").replace(".jpg", "")
    overall_name = f"OVERALL_{timestamp}.jpg"
    pattern = os.path.join(cfg.CAPTURE_DIR, cfg.PRODUCT_NAME, "sessions", "*", overall_name)
    matches = glob.glob(pattern)
    if matches:
        return FileResponse(matches[0])
    return Response(status_code=404)


@app.get("/api/dataset-records/{record_id}")
async def get_dataset_record(record_id: int):
    """
    12082026 - KIET - Trả chi tiết record và object để frontend vẽ annotation từ database.
    """

    import ast
    import json

    db = SessionLocal()
    try:
        record = (
            db.query(InspectionRecord)
            .filter(InspectionRecord.id == record_id)
            .first()
        )

        if record is None:
            return Response(
                status_code=404,
                content="Dataset record not found",
            )

        image_name = (
            record.original_image.split("/")[-1].split("\\")[-1]
            if record.original_image
            else f"IMG_{record.timestamp}.jpg"
        )
        image_pattern = os.path.join(
            cfg.CAPTURE_DIR,
            cfg.PRODUCT_NAME,
            "sessions",
            "*",
            "original",
            image_name,
        )
        image_matches = glob.glob(image_pattern)
        image_width = 0
        image_height = 0

        if image_matches:
            image = cv2.imread(image_matches[0])
            if image is not None:
                image_height, image_width = image.shape[:2]

        objects = []
        for obj in record.objects:
            try:
                bbox = json.loads(obj.bbox) if obj.bbox else []
            except (json.JSONDecodeError, TypeError):
                try:
                    bbox = ast.literal_eval(obj.bbox) if obj.bbox else []
                except (ValueError, SyntaxError):
                    bbox = []

            objects.append({
                "index": obj.object_index,
                "bbox": bbox,
                "score": obj.score,
                "class_id": obj.class_id,
                "class_name": obj.class_name,
            })

        # 12082026 - KIET - Tính count từ object để tránh lưu dữ liệu tổng hợp trùng lặp.
        counts_by_class = {}
        if record.task_type == "detection":
            counts_by_class = dict(
                Counter(
                    obj["class_name"] or "unknown"
                    for obj in objects
                )
            )

        return {
            "id": record.id,
            "task_type": record.task_type,
            "image_url": f"/api/image/{image_name}",
            "image_width": image_width,
            "image_height": image_height,
            "total_objects": record.total_objects,
            "counts_by_class": counts_by_class,
            "objects": objects,
        }
    finally:
        db.close()


@app.get("/dataset-stats")
async def get_dataset_stats():
    """
    12082026 - KIET - Trả danh sách dataset có metadata riêng cho Inspection và Detection.
    """

    db = SessionLocal()
    try:
        db.commit() # Clear cached transaction
        records = db.query(InspectionRecord).order_by(InspectionRecord.id.desc()).all()
        total = len(records)
        
        images_list = []
        for r in records[:50]: # Lấy 50 ảnh gần nhất
            ts = r.timestamp
            date_str = ts
            if "_" in ts:
                parts = ts.split('_')
                if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
                    d, t = parts[0], parts[1]
                    date_str = f"{d[6:8]}/{d[4:6]}/{d[0:4]} {t[0:2]}:{t[2:4]}:{t[4:6]}"

            # 12082026 - KIET - Không gắn record Detection thành kết quả OK của Inspection.
            record_type = (
                "Detection"
                if r.task_type == "detection"
                else ("NG" if r.ng_detected else "OK")
            )
            counts_by_class = {}
            if r.task_type == "detection":
                counts_by_class = dict(
                    Counter(
                        obj.class_name or "unknown"
                        for obj in r.objects
                    )
                )
                
            images_list.append({
                "id": r.id,
                "name": r.original_image.split('/')[-1].split('\\')[-1] if r.original_image else f"IMG_{r.timestamp}.jpg",
                "status": "labeled",
                "type": record_type,
                "task_type": r.task_type,
                "total_objects": r.total_objects,
                "counts_by_class": counts_by_class,
                "date": date_str,
                "product": cfg.PRODUCT_NAME
            })
            
        return {
            "stats": [
                {"label": "Total Images", "value": str(total), "color": "#3b82f6"},
                {"label": "Labeled", "value": str(total), "color": "#10b981"},
                {"label": "Synthetic", "value": "0", "color": "#8b5cf6"},
                {"label": "Pending", "value": "0", "color": "#f59e0b"}
            ],
            "images": images_list
        }
    except Exception as e:
        print(f"Error fetching dataset stats: {e}")
        return {"stats": [], "images": []}
    finally:
        db.close()

@app.get("/model-registry")
async def get_model_registry():
    from services.database.models import AIModel
    db = SessionLocal()
    try:
        models = db.query(AIModel).all()
        model_list = []
        for m in models:
            model_list.append({
                "id": m.id,
                "name": m.name,
                "type": m.type,
                "status": "active" if m.status == "PRODUCTION" else m.status.lower(),
                "mAP": f"{m.map_acc}%" if m.map_acc else "N/A",
                "speed": m.speed_ms,
                "format": m.format,
                "version": m.version
            })
            
        
        # Real-time hardware metrics for resourceData
        import psutil
        import numpy as np
        cpu_load = psutil.cpu_percent(interval=0.1)
        vram_usage = min(100.0, cpu_load * 1.2 + (np.random.random() * 5.0)) # Simulated VRAM
        npu_load = min(100.0, 40.0 + (np.random.random() * 40.0))
        
        resourceData = [
            { "name": 'GPU Memory', "value": round(vram_usage, 1), "color": '#2563eb' },
            { "name": 'Free', "value": round(100 - vram_usage, 1), "color": '#e2e8f0' },
        ]
        
        # Breakdown of latency (simulate based on total 302ms)
        latencyData = [
            { "stage": 'Detection', "time": 45, "color": '#3b82f6' },
            { "stage": 'Alignment', "time": 32, "color": '#10b981' },
            { "stage": 'Segmentation', "time": 58, "color": '#6366f1' },
            { "stage": 'Anomaly', "time": 145, "color": '#f59e0b' },
            { "stage": 'Classification', "time": 22, "color": '#f43f5e' },
        ]

        return {
            "activeEngine": "v2.4.1 Stable",
            "pipelineLatency": 302,
            "mAP": 91.4,
            "models": model_list,
            "latencyData": latencyData,
            "resourceData": resourceData,
            "cpuThreads": round(cpu_load, 1),
            "npuLoad": round(npu_load, 1)
        }
    except Exception as e:
        print(f"Error fetching models: {e}")
        return {"models": []}
    finally:
        db.close()

@app.get("/system-health")
async def get_system_health():
    cpu_load = psutil.cpu_percent(interval=0.1) 
    memory = psutil.virtual_memory()
    ram_usage = memory.percent
    disk = psutil.disk_usage('/')
    disk_usage = disk.percent
    
    core_temp = 42.0 + (cpu_load * 0.35) + (np.random.random() * 2.0)
    gpu_load = min(100.0, cpu_load * 1.2 + (np.random.random() * 5.0))

    return {
        "cpu_load": round(cpu_load, 1),
        "ram_usage": round(ram_usage, 1),
        "disk_usage": round(disk_usage, 1),
        "temperature": round(core_temp, 1),
        "gpu_load": round(gpu_load, 1)
    }


@app.get("/api/analytics")
def get_analytics():
    from services.database.models import QCProductPhotoLibrary, InspectionRecord, BoltObject
    from sqlalchemy import func
    import dateutil.parser
    from datetime import datetime
    
    db = SessionLocal()
    try:
        # 1. Defect Pareto from QCProductPhotoLibrary
        counts = db.query(QCProductPhotoLibrary.ErrorDetail, func.count(QCProductPhotoLibrary.ID)).group_by(QCProductPhotoLibrary.ErrorDetail).all()
        
        pareto = []
        colors = ['#f43f5e', '#fb923c', '#facc15', '#38bdf8', '#94a3b8']
        
        for idx, row in enumerate(counts):
            err = row[0] or "Unknown"
            cnt = row[1]
            pareto.append({
                "name": err,
                "count": cnt,
                "color": colors[idx % len(colors)]
            })
            
        pareto = sorted(pareto, key=lambda x: x["count"], reverse=True)
        
        # 2. Production stats from InspectionRecord
        records = db.query(InspectionRecord).all()
        
        total_inspected = len(records)
        total_ng = sum(1 for r in records if r.ng_detected)

        # 11082026 - KIET - Tính tổng object của các record Object Detection.
        total_detected = (
            db.query(func.sum(InspectionRecord.total_objects))
            .filter(InspectionRecord.task_type == "detection")
            .scalar()
            or 0
        )

        # 11082026 - KIET - Thống kê tổng số object theo class đã lưu trong database.
        od_class_rows = (
            db.query(BoltObject.class_name, func.count(BoltObject.id))
            .join(InspectionRecord, BoltObject.record_id == InspectionRecord.id)
            .filter(InspectionRecord.task_type == "detection")
            .group_by(BoltObject.class_name)
            .all()
        )
        detection_counts_by_class = {
            class_name or "unknown": count
            for class_name, count in od_class_rows
        }
        
        hourly_stats = {}
        
        for r in records:
            try:
                ts_val = r.timestamp or str(r.created_at)
                dt = None
                if "_" in str(ts_val):
                    parts = str(ts_val).split('_')
                    if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) >= 6:
                        d, t = parts[0], parts[1][:6]
                        try:
                            dt = datetime.strptime(f"{d}_{t}", "%Y%m%d_%H%M%S")
                        except ValueError:
                            pass
                if not dt:
                    if isinstance(ts_val, (int, float)):
                        dt = datetime.fromtimestamp(ts_val)
                    else:
                        dt = dateutil.parser.parse(str(ts_val))
                    
                hour_str = f"{dt.hour:02d}h"
                if hour_str not in hourly_stats:
                    hourly_stats[hour_str] = {"ok": 0, "ng": 0}
                    
                if r.ng_detected:
                    hourly_stats[hour_str]["ng"] += 1
                else:
                    hourly_stats[hour_str]["ok"] += 1
                    
            except Exception:
                pass
                
        overall_yield = ((total_inspected - total_ng) / total_inspected * 100) if total_inspected > 0 else 100.0
                
        hourly_output_data = []
        yield_trend_data = []
        
        for hour in sorted(hourly_stats.keys()):
            ok_count = hourly_stats[hour]["ok"]
            ng_count = hourly_stats[hour]["ng"]
            total = ok_count + ng_count
            hourly_output_data.append({
                "hour": hour,
                "ok": ok_count,
                "ng": ng_count
            })
            rate = (ok_count / total * 100) if total > 0 else 100.0
            yield_trend_data.append({
                "time": hour.replace("h", ":00"),
                "rate": round(rate, 1)
            })
            
        if not hourly_output_data:
            hourly_output_data = [{"hour": "08h", "ok": 0, "ng": 0}]
            yield_trend_data = [{"time": "08:00", "rate": 100.0}]

        return {
            "totalInspected": total_inspected,
            # 11082026 - KIET - Bổ sung analytics OD mà không thay response inspection cũ.
            "totalDetected": int(total_detected),
            "detectionCountsByClass": detection_counts_by_class,
            "overallYield": round(overall_yield, 1),
            "totalNg": total_ng,
            "avgCycleTime": 2.45,
            "pareto": pareto,
            "hourlyOutputData": hourly_output_data,
            "yieldTrendData": yield_trend_data
        }
    except Exception as e:
        print(f"Error fetching analytics: {e}")
        return {
            "totalInspected": 0,
            "totalDetected": 0,
            "detectionCountsByClass": {},
            "overallYield": 100.0,
            "totalNg": 0,
            "avgCycleTime": 0.0,
            "pareto": [],
            "hourlyOutputData": [],
            "yieldTrendData": []
        }
    finally:
        db.close()


# ==========================================
# CLOUD-EDGE SYNCHRONIZATION APIs
# ==========================================
import requests
from fastapi import Response

CLOUD_API_BASE = "http://localhost:8031" # Default defect classification server port for testing

@app.post("/api/sync/dataset-up")
async def sync_dataset_up():
    from services.database.models import InspectionRecord, SyncState
    db = SessionLocal()
    try:
        # Get last synced id
        sync_state = db.query(SyncState).filter(SyncState.key == "last_synced_record_id").first()
        last_id = int(sync_state.value) if sync_state and sync_state.value else 0
        
        # Get new records
        new_records = db.query(InspectionRecord).filter(InspectionRecord.id > last_id).order_by(InspectionRecord.id.asc()).limit(100).all()
        
        if not new_records:
            return {"status": "success", "message": "Already up to date", "synced_count": 0}
            
        payload = {
            "records": [
                {
                    "edge_node_id": cfg.EDGE_CODE,
                    "timestamp": r.timestamp,
                    "ng_detected": r.ng_detected,
                    "total_objects": len(r.objects)
                } for r in new_records
            ]
        }
        
        # Send to Cloud
        res = requests.post(f"{CLOUD_API_BASE}/api/dataset/sync", json=payload, timeout=10)
        res.raise_for_status()
        
        # Update last synced id
        highest_id = new_records[-1].id
        if not sync_state:
            sync_state = SyncState(key="last_synced_record_id", value=str(highest_id))
            db.add(sync_state)
        else:
            sync_state.value = str(highest_id)
            
        db.commit()
        return {"status": "success", "synced_count": len(new_records)}
    except Exception as e:
        print(f"Sync Up Error: {e}")
        return Response(status_code=500, content=f"Sync Up Error: {e}")
    finally:
        db.close()

@app.post("/api/sync/models-up")
async def sync_models_up():
    from services.database.models import AIModel
    db = SessionLocal()
    try:
        models = db.query(AIModel).all()
        if not models:
            return {"status": "success", "synced_count": 0, "message": "No models to sync"}
            
        payload = {
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "type": m.type,
                    "format": m.format,
                    "version": m.version,
                    "map_acc": float(m.map_acc) if m.map_acc else 0.0,
                    "status": m.status,
                    "file_path": m.file_path,
                    "speed_ms": m.speed_ms or ""
                } for m in models
            ]
        }
        
        # Send to Cloud
        res = requests.post(f"{CLOUD_API_BASE}/api/models/sync-up", json=payload, timeout=10)
        res.raise_for_status()
        
        return {"status": "success", "synced_count": len(models)}
    except Exception as e:
        print(f"Sync Models Up Error: {e}")
        return Response(status_code=500, content=f"Sync Models Up Error: {e}")
    finally:
        db.close()

@app.post("/api/sync/models-down")
async def sync_models_down():
    from services.database.models import AIModel
    db = SessionLocal()
    try:
        res = requests.get(f"{CLOUD_API_BASE}/api/models/latest", timeout=10)
        res.raise_for_status()
        data = res.json()
        
        cloud_models = data.get("models", [])
        synced_count = 0
        
        for cm in cloud_models:
            local_m = db.query(AIModel).filter(AIModel.id == cm["id"]).first()
            if not local_m:
                # Add new model
                new_m = AIModel(
                    id=cm["id"],
                    name=cm["name"],
                    type=cm["type"],
                    format=cm["format"],
                    version=cm["version"],
                    map_acc=cm["map_acc"],
                    status="STANDBY", # Downloaded but not activated
                    file_path=f"cloud_downloads/{cm['name']}",
                    speed_ms=cm["speed_ms"]
                )
                db.add(new_m)
                synced_count += 1
            elif local_m.version != cm["version"]:
                # Update existing model metadata
                local_m.version = cm["version"]
                local_m.map_acc = cm["map_acc"]
                local_m.speed_ms = cm["speed_ms"]
                synced_count += 1
                
        db.commit()
        return {"status": "success", "synced_count": synced_count}
    except Exception as e:
        print(f"Sync Down Error: {e}")
        return Response(status_code=500, content=f"Sync Down Error: {e}")
    finally:
        db.close()

import io
from fastapi.responses import StreamingResponse
from datetime import datetime

@app.get("/api/reports/lookup/{item_code}")
async def report_lookup(item_code: str):
    db = SessionLocal()
    try:
        from sqlalchemy import text
        insp_count = db.execute(text(f"SELECT COUNT(*) FROM inspection_records WHERE \"ItemCode\" = '{item_code}'")).scalar()
        qc_count = db.execute(text(f"SELECT COUNT(*) FROM qc_product_photo_library WHERE \"ItemCode\" = '{item_code}'")).scalar()
        
        reports = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if insp_count and insp_count > 0:
            reports.append({
                "id": f"REP-INSP-{item_code}",
                "name": f"Inspection Records - {item_code}",
                "type": "System",
                "format": "Excel",
                "date": now_str,
                "size": f"{insp_count} rows",
                "downloads": 0,
                "lastDownloaded": "-"
            })
            
        if qc_count and qc_count > 0:
            reports.append({
                "id": f"REP-QC-{item_code}",
                "name": f"QC Photo Library - {item_code}",
                "type": "System",
                "format": "Excel",
                "date": now_str,
                "size": f"{qc_count} rows",
                "downloads": 0,
                "lastDownloaded": "-"
            })
            
        return {"status": "success", "reports": reports}
    finally:
        db.close()

@app.get("/api/reports/download/{table_name}/{item_code}")
async def report_download(table_name: str, item_code: str):
    db = SessionLocal()
    try:
        if table_name not in ["inspection_records", "qc_product_photo_library"]:
            return Response(status_code=400, content="Invalid table")
            
        try:
            import pandas as pd
            df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE \"ItemCode\" = '{item_code}'", db.bind)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name=item_code)
                
            output.seek(0)
            headers = {
                'Content-Disposition': f'attachment; filename="{table_name}_{item_code}.xlsx"',
                'Access-Control-Expose-Headers': 'Content-Disposition'
            }
            return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except ImportError:
            # Fallback to CSV if pandas/openpyxl is not installed
            import csv
            from sqlalchemy import text
            rows = db.execute(text(f"SELECT * FROM {table_name} WHERE \"ItemCode\" = '{item_code}'")).mappings().all()
            if not rows:
                return Response(status_code=404, content="No data found")
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
                
            headers = {
                'Content-Disposition': f'attachment; filename="{table_name}_{item_code}.csv"',
                'Access-Control-Expose-Headers': 'Content-Disposition'
            }
            return Response(content=output.getvalue(), media_type="text/csv", headers=headers)
            
    except Exception as e:
        print(f"Error generating report: {e}")
        return Response(status_code=500, content=f"Error generating report: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
