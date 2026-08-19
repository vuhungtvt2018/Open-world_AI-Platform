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
from fastapi import FastAPI, Response, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import logic (New DAG Architecture)
from packages.core.config import AppConfig
from packages.workflow.pipeline import Pipeline
from packages.camera import RTSP_Threaded_Camera, Basler_Threaded_Camera
from packages.utils.utils import (ts, union_box, pad_and_clip_box, 
                     arrow_angle, rotate_image, transform_points, need_clean, clean_data,
                     transform_bbox_to_original_coords)
from packages.utils.visualize import concat_anomaly_crops
from services.database.session import init_db, SessionLocal
from services.database.crud import create_inspection_record
from packages.utils.cleaner import run_cleaner_daemon
from services.database.models import InspectionRecord

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

"""
07082026 - KHAI - Refactor code to adhere to modified Pipeline
"""
class WebInference:
    def __init__(self, config):
        self.cfg = config
        self.pipeline = Pipeline(config)
        
        # Khởi tạo Multi-Camera Stream
        self.cameras = {} # id -> Camera_Instance
        self.input_mode = {} # id -> "stream", "basler", "folder"
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
            frame = self.cameras[cam_id].read()
            return frame
        else:
            # Lấy ảnh trực tiếp từ RAM (không đọc ổ cứng)
            if self.current_frame is None:
                return None
            return self.current_frame.copy()

    def run_inspect(self, cam_id: str = "default"):
        frame = self.get_frame(cam_id)
        if frame is None:
            return {"status": "error", "message": "No input frame available"}
            
        if self.cfg.FLIP_VERTICAL:
            frame = cv2.flip(frame, 0)
            
        H, W = frame.shape[:2]
        name = ts()
        insp_time_str = self.pipeline.format_inspection_time(name)
        
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
                
                crop_x1 = max(0, center_x - max_val // 2)
                crop_y1 = max(0, center_y - max_val // 2)
                crop_x2 = center_x + max_val // 2
                crop_y2 = center_y + max_val // 2
                crop = rotated_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_mask = rotated_mask[crop_y1:crop_y2, crop_x1:crop_x2]

                if crop.size == 0 or crop_mask.size == 0:
                    continue

                # Lưu crop
                crop_name = f"CROP_{name}_{i}.jpg"
                crop_path = os.path.join(self.pipeline.dir_crops, crop_name)
                cv2.imwrite(crop_path, crop)

                """
                09082026 - KHAI - Add codes to save mask
                """
                # Lưu mask
                mask_name = f"MASK_{name}_{i}.jpg"
                mask_path = os.path.join(self.pipeline.dir_masks, mask_name)
                cv2.imwrite(mask_path, crop_mask)

                # Anomaly Detection
                out = self.pipeline.anomaly.run(crop, crop_mask)
                is_ng = bool(getattr(out, "is_ng", False))
                
                # Visualization tiles
                hm_tile = None
                if getattr(out, "heatmap_display") is not None:
                    hm_tile = out.heatmap_display.copy()
                    """
                    19082026 - KHAI - Hide objects not in the main focus in heatmap tile for visualization
                    """
                    hm_h, hm_w = hm_tile.shape[:2]        
                    mask_resized = cv2.resize((crop_mask > 0).astype(np.uint8), (hm_w, hm_h), interpolation=cv2.INTER_NEAREST)
                    mask_3ch = mask_resized[:, :, None]
                    hm_tile = hm_tile * mask_3ch
                    label_small = f"obj{i} {'NG' if is_ng else 'OK'}"
                    color = (0, 0, 255) if is_ng else (0, 200, 0)
                    cv2.putText(hm_tile, label_small, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                anomalies_full = []
                cls_labels_i = []
                
                if is_ng:
                    ng_any = True
                    # Lưu heatmap cho NG
                    cv2.imwrite(os.path.join(self.pipeline.dir_ng, f"NG_HEAT_{name}_{i}.jpg"), out.heatmap_display)

                    for ak, a in enumerate(getattr(out, "boxes", [])):
                        """
                        08082026 - KHAI - Refactor code to adhere to modified Pipeline
                        """
                        cx1, cy1, cx2, cy2 = int(a.xmin), int(a.ymin), int(a.xmax), int(a.ymax)
                        """
                        16082026 - KHAI - Convert bbox in crop to bbox in original
                        """
                        bbox_in_crop = [int(cx1), int(cy1), int(cx2), int(cy2)]
                        crop_origin = (crop_x1, crop_y1)
                        bbox_in_original = transform_bbox_to_original_coords(
                            bbox_in_crop,
                            M,
                            crop_origin,
                            frame.shape,
                        )
                        anomalies_full.append({
                            "k": ak,
                            "bbox_full": bbox_in_original,
                            "bbox_in_object_crop": bbox_in_crop,
                            "bbox_in_original": bbox_in_original,
                        })
                        anom_crop = crop[cy1:cy2, cx1:cx2]
                        if anom_crop.size != 0:
                            anom_name = f"ANOM_{name}_obj{i}_{ak}.jpg"
                            anom_path = os.path.join(self.pipeline.dir_anom_crops, anom_name)
                            cv2.imwrite(anom_path, anom_crop)
                            
                            # Gọi API phân loại của pipeline
                            cls_res = self.pipeline.classify_defect(anom_path)
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
                """
                19082026 - KHAI - Add Vignette mask to crop to focus on main object
                """
                y_indices, x_indices = np.where(crop_mask > 0)

                # Blur surrounding of main object
                if len(y_indices) > 0 and len(x_indices) > 0:
                    tight_y1, tight_y2 = y_indices.min(), y_indices.max() + 1
                    tight_x1, tight_x2 = x_indices.min(), x_indices.max() + 1

                    h, w = crop.shape[:2]
                    cx, cy = (tight_x1 + tight_x2) // 2, (tight_y1 + tight_y2) // 2
                    X, Y = np.meshgrid(np.arange(w), np.arange(h))
                    max_radius = np.sqrt(w**2 + h**2) / 2.0
                    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)

                    # Vignette mask
                    vignette_mask = np.clip(1.0 - (dist_from_center / max_radius) * 0.90, 0.10, 1.0)
                    vignette_mask_3ch = np.dstack([vignette_mask] * 3)

                    # Dim background area
                    vignetted_crop = (crop.astype(np.float32) * vignette_mask_3ch * 0.5).astype(np.uint8)

                    # Segmentation mask blending
                    binary_mask = (crop_mask > 0).astype(np.float32)
                    feathered_mask = cv2.GaussianBlur(binary_mask, (15, 15), 0)[:, :, None]
                    crop_labeled = (crop.astype(np.float32) * feathered_mask + 
                                    vignetted_crop.astype(np.float32) * (1.0 - feathered_mask)).astype(np.uint8)
                
                if is_ng and len(anomalies_full) > 0:
                    for idx_disp, a in enumerate(anomalies_full, start=1):
                        """
                        19082026 - KHAI - Fix bbox thickness and label size for crop images
                        """
                        cx1, cy1, cx2, cy2 = a["bbox_in_object_crop"]
                        cv2.rectangle(crop_labeled, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)
                        
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
                                
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.25, 2)
                        tx1, ty1 = cx1, max(0, cy1 - th - 6)
                        tx2, ty2 = cx1 + tw + 8, cy1
                        cv2.rectangle(crop_labeled, (tx1, ty1), (tx2, ty2), (0, 0, 255), -1)
                        cv2.putText(crop_labeled, label_text, (cx1 + 3, cy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                per_objects.append({
                    "product_id": 1,
                    "index": i,
                    "bbox": [0, 0, crop.shape[1], crop.shape[0]],
                    "score": round(float(getattr(out, "classification_score", 0.0)), 4),
                    "is_ng": is_ng,
                    "overlap_ratio": round(float(getattr(out, "overlap_ratio", 0.0)), 4) if crop.size != 0 else 0.0,
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
            """
            10082026 - KHANH - Visualize OK and NG objects in overall output
            """
            object_index = obj["index"]

            if object_index >= len(mapping_object):
                continue

            """
            16082026 - KHAI - Convert bbox in crop to bbox in original
            """
            is_ng = obj["is_ng"]
            if is_ng and obj.get("anomalies"):
                for ak, anomaly in enumerate(obj["anomalies"], start=1):
                    x1, y1, x2, y2 = anomaly.get("bbox_in_original", [0, 0, 0, 0])
                    cv2.rectangle(
                        overall_vis,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        3,
                    )

                    label = anomaly.get("cls_label") or f"NG{ak}"
                    if anomaly.get("cls_similarity") is not None:
                        label = f"{label} ({float(anomaly['cls_similarity']):.2f})"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        2,
                    )
                    label_top = max(0, y1 - text_height - baseline - 8)
                    cv2.rectangle(
                        overall_vis,
                        (x1, label_top),
                        (x1 + text_width + 10, y1),
                        (0, 0, 255),
                        -1,
                    )
                    cv2.putText(
                        overall_vis,
                        label,
                        (x1 + 5, y1 - baseline - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
        
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
async def video_feed(cam: str = "default"):
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

@app.post("/inspect")
async def trigger_inspect(cam: str = "default"):
    return inference_engine.run_inspect(cam)

@app.get("/history")
async def get_history():
    import ast
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
                    obbox = ast.literal_eval(o.bbox) if o.bbox else []
                except: obbox = []
                objs.append({
                    "index": o.object_index,
                    "bbox": obbox,
                    "score": o.score,
                    "is_ng": o.is_ng,
                    "overlap_ratio": o.overlap_ratio,
                    "crop": o.crop_image,
                    "anomalies": anoms,
                    "anomaly_count": len(anoms),
                    "anomaly_bboxes_sample": [a["bbox_full"] for a in anoms[:3]]
                })
            results.append({
                "timestamp": r.timestamp,
                "image": r.original_image,
                "ng_detected": r.ng_detected,
                "latency_ms": r.latency_ms,
                "total_objects": len(objs),
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

@app.get("/dataset-stats")
async def get_dataset_stats():
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
                
            images_list.append({
                "id": r.id,
                "name": r.original_image.split('/')[-1].split('\\')[-1] if r.original_image else f"IMG_{r.timestamp}.jpg",
                "status": "labeled",
                "type": "NG" if r.ng_detected else "OK",
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
    from services.database.models import QCProductPhotoLibrary, InspectionRecord
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

CLOUD_API_BASE = "http://127.0.0.1:8031" # Default defect classification server port for testing

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
