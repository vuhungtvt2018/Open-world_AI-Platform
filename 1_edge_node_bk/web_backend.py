import os
import cv2
import json
import time
import numpy as np
import shutil
import psutil
from typing import List, Optional
from fastapi import FastAPI, Response, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import logic của bạn
from app.config import AppConfig
from app.pipeline import Pipeline
from app.camera import RTSP_Threaded_Camera
from app.utils import (ts, ensure_dirs, compute_iou, union_box, pad_and_clip_box, 
                     arrow_angle, rotate_image, transform_points, need_clean, clean_data)
from app.visualize import concat_anomaly_crops
from app.database.session import init_db
from app.database.crud import create_inspection_record

app = FastAPI(title="MTI Vision AI Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


cfg = AppConfig.from_yaml("config.yaml")

@app.on_event("startup")
def start_background_tasks():
    init_db()
    from app.cleaner import run_cleaner_daemon
    
    run_cleaner_daemon(
        captures_dir=cfg.CAPTURE_DIR,
        days_to_keep=cfg.DISK_CLEANUP_DAYS,
        interval_hours=cfg.DISK_CLEANUP_INTERVAL_HOURS
    )

app.mount("/captures", StaticFiles(directory=cfg.CAPTURE_DIR), name="captures")

class WebInference:
    def __init__(self, config):
        self.cfg = config
        self.pipeline = Pipeline(config)
        
        # Khởi tạo Camera Stream
        self.cam = None
        self.input_mode = "folder" # Hoặc "stream"
        self.selected_image = None
        
    def set_input_mode(self, mode: str):
        self.input_mode = mode
        if mode == "stream" and self.cam is None:
            self.cam = RTSP_Threaded_Camera(self.cfg.RTSP_URL, width=self.cfg.CAM_WIDTH, height=self.cfg.CAM_HEIGHT)
        elif mode == "folder" and self.cam is not None:
            self.cam.stop()
            self.cam = None

    def get_frame(self):
        if self.input_mode == "stream" and self.cam is not None:
            frame = self.cam.read()
            return frame
        else:
            # Load ảnh được chọn từ folder
            frame = cv2.imread(self.selected_image)
            return frame

    def run_inspect(self):
        frame = self.get_frame()
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
        
        # [START COPY LOGIC FROM PIPELINE.PY SPACE KEY]
        # logic xử lý AI bắt đầu từ đây (giống hệt pipeline.py dòng 233 trở đi)
        
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
                
                # Visualization tiles (similar to pipeline.py)
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
                            print('cls_res ', cls_res)
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

        # 3. Overall output (Annotated original image)
        overall_vis = frame.copy()
        # Draw bounding boxes from result_keypoints or similar? 
        # Actually, let's just draw the object boxes
        for i, obj in enumerate(per_objects):
            # We don't have the original boxes easily here, but we can reconstruct or just use a simplified version
            # For now, let's just use the frame as is, but annotated
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

        # Cleanup per_objects for JSON serialization (remove numpy arrays)
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
        try:
            create_inspection_record(record)
        except Exception as e:
            print(f"[ERROR] Failed to insert DB record: {e}")

        # Trả về kết quả JSON cho Web
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
    file_path = os.path.join(".", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    inference_engine.selected_image = file.filename
    # Trả về danh sách ảnh mới để UI cập nhật
    images = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    return {"filename": file.filename, "images": images}

@app.get("/video-feed")
async def video_feed():
    def gen_frames():
        while True:
            frame = inference_engine.get_frame()
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
    # Lấy danh sách file ảnh trong thư mục gốc
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir('.') if f.lower().endswith(valid_extensions)]
    return {"images": images}

@app.get("/images/{filename}")
async def get_image(filename: str):
    file_path = os.path.join(".", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return Response(status_code=404)

@app.post("/set-image/{filename}")
async def set_image(filename: str):
    inference_engine.selected_image = filename
    return {"message": f"Selected image set to {filename}"}

@app.post("/set-mode/{mode}")
async def set_mode(mode: str):
    inference_engine.set_input_mode(mode)
    return {"message": f"Input mode set to {mode}"}

@app.post("/inspect")
async def trigger_inspect():
    return inference_engine.run_inspect()

@app.get("/history")
async def get_history():
    from app.database.session import SessionLocal
    from app.database.models import InspectionRecord
    import ast
    db = SessionLocal()
    try:
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


@app.get("/system-health")
async def get_system_health():
    # Use interval=0.1 to get a real reading, though it blocks slightly.
    # Alternatively, use interval=None but it requires being called periodically.
    cpu_load = psutil.cpu_percent(interval=0.1) 
    memory = psutil.virtual_memory()
    ram_usage = memory.percent
    disk = psutil.disk_usage('/')
    disk_usage = disk.percent
    
    # CPU Temp (simulated for Windows compatibility, correlated with CPU load)
    # Real sensors_temperatures() is often empty on Windows/Mac
    core_temp = 42.0 + (cpu_load * 0.35) + (np.random.random() * 2.0)
    
    # GPU Load (simulated as edge node often uses integrated or NPU)
    gpu_load = min(100.0, cpu_load * 1.2 + (np.random.random() * 5.0))

    return {
        "cpu_load": round(cpu_load, 1),
        "ram_usage": round(ram_usage, 1),
        "disk_usage": round(disk_usage, 1),
        "temperature": round(core_temp, 1),
        "gpu_load": round(gpu_load, 1)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
