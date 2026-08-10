import ast
import glob
import os
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Response, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse

from services.database.models import InspectionRecord
from services.database.session import SessionLocal

from ..app_config import cfg
from ..inference import inference_engine

router = APIRouter()

@router.post("/upload")
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

@router.get("/video-feed")
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

@router.get("/images")
async def list_images():
    # Trả về ảnh duy nhất đang nằm trên RAM
    if inference_engine.current_frame is not None:
        return {"images": [inference_engine.current_image_name]}
    return {"images": []}

@router.get("/images/{filename}")
async def get_image(filename: str):
    if filename.startswith("in_memory_image") and inference_engine.current_frame is not None:
        ret, buffer = cv2.imencode('.jpg', inference_engine.current_frame)
        if ret:
            return Response(content=buffer.tobytes(), media_type="image/jpeg")
    return Response(status_code=404)

@router.post("/set-image/{filename}")
async def set_image(filename: str):
    return {"message": "In-memory mode does not support switching past uploads"}

@router.post("/set-mode/{mode}")
async def set_mode(mode: str, cam: str = "default"):
    try:
        inference_engine.set_camera_mode(cam, mode)
        return {"message": f"Input mode for {cam} set to {mode}"}
    except Exception as e:
        from fastapi import HTTPException
        print(f"Error setting mode: {e}")
        raise HTTPException(status_code=500, detail=f"Camera Error: {str(e)}")

@router.post("/inspect")
async def trigger_inspect(cam: str = "default"):
    return inference_engine.run_inspect(cam)

@router.get("/history")
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

@router.get("/api/image/{image_name}")
async def get_image(image_name: str):
    pattern = os.path.join(cfg.CAPTURE_DIR, cfg.PRODUCT_NAME, "sessions", "*", "original", image_name)
    matches = glob.glob(pattern)
    if matches:
        return FileResponse(matches[0])
    return Response(status_code=404)

@router.get("/api/result_image/{image_name}")
async def get_result_image(image_name: str):
    # Lấy tên file gốc (IMG_XXX.jpg) và chuyển thành OVERALL_XXX.jpg
    timestamp = image_name.replace("IMG_", "").replace(".jpg", "")
    overall_name = f"OVERALL_{timestamp}.jpg"
    pattern = os.path.join(cfg.CAPTURE_DIR, cfg.PRODUCT_NAME, "sessions", "*", overall_name)
    matches = glob.glob(pattern)
    if matches:
        return FileResponse(matches[0])
    return Response(status_code=404)
