import cv2
import time
import numpy as np
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import APIRouter, UploadFile, File, Response, HTTPException
from fastapi.responses import StreamingResponse
from packages.camera import discover_basler_cameras
from services.camera_service.core import camera_manager
from services.database.crud import (
    get_camera_config,
    update_camera_config,
    upsert_camera_config,
)

router = APIRouter(tags=["Camera"])


"""
23082026 - KHAI - Add request models for camera-related APIs
"""
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


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    camera_manager.current_frame = frame

    dynamic_name = f"in_memory_image_{int(time.time()*1000)}"
    camera_manager.current_image_name = dynamic_name

    return {"filename": dynamic_name, "images": [dynamic_name]}


@router.get("/video-feed")
async def video_feed(cam: str = "default", fps: float = 15.0):
    """
    19082026 - KIET - Stream đúng camera ID và giới hạn FPS để multi-camera không chiếm CPU quá mức.
    """

    frame_interval = 1.0 / min(max(fps, 1.0), 30.0)

    def gen_frames():
        while True:
            frame = camera_manager.get_frame(cam)
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


@router.get("/camera-stats")
async def camera_stats():
    stats = {}
    for cam_id, cam in camera_manager.cameras.items():
        cam_type = camera_manager.input_mode.get(cam_id)
        if cam_type == "basler" and cam is not None:
            try:
                if hasattr(cam, "camera") and cam.camera.IsOpen():
                    w = cam.camera.Width.GetValue()
                    h = cam.camera.Height.GetValue()
                    fps = round(cam.camera.ResultingFrameRate.GetValue(), 1)
                    stats[cam_id] = {
                        "resolution": f"{w}x{h}", "fps": fps,
                        "codec": "RAW / Mono8",
                        "bitrate": f"{round((w * h * fps * 8) / 1_000_000, 1)} Mbps",
                        "latency": "2ms"
                    }
            except Exception as e:
                print(f"Stats Error Basler: {e}")
        elif cam_type == "stream" and cam is not None:
            try:
                if hasattr(cam, "cap") and cam.cap.isOpened():
                    w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cam.cap.get(cv2.CAP_PROP_FPS))
                    if fps <= 0:
                        fps = 30
                    stats[cam_id] = {
                        "resolution": f"{w}x{h}", "fps": fps,
                        "codec": "MJPEG / H.264", "bitrate": "VBR", "latency": "40ms"
                    }
            except Exception as e:
                print(f"Stats Error RTSP: {e}")
    return {"stats": stats}


@router.get("/images")
async def list_images():
    if camera_manager.current_frame is not None:
        return {"images": [camera_manager.current_image_name]}
    return {"images": []}


@router.get("/images/{filename}")
async def get_image_ram(filename: str):
    if filename.startswith("in_memory_image") and camera_manager.current_frame is not None:
        ret, buffer = cv2.imencode('.jpg', camera_manager.current_frame)
        if ret:
            return Response(content=buffer.tobytes(), media_type="image/jpeg")
    return Response(status_code=404)


@router.post("/set-image/{filename}")
async def set_image(filename: str):
    return {"message": "In-memory mode does not support switching past uploads"}


@router.post("/set-mode/{mode}")
async def set_mode(mode: str, cam: str = "default"):
    try:
        camera_manager.set_camera_mode(cam, mode)
        return {"message": f"Input mode for {cam} set to {mode}"}
    except Exception as e:
        print(f"Error setting mode: {e}")
        raise HTTPException(status_code=500, detail=f"Camera Error: {str(e)}")


"""
23082026 - KHAI - Add new endpoints related to camera service
"""
@router.get("/api/cameras")
def list_cameras():
    """
    19082026 - KIET - Trả danh sách camera đã cấu hình kèm trạng thái runtime.
    """

    return {"cameras": camera_manager.list_camera_statuses()}


@router.get("/api/cameras/discover/basler")
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


@router.post("/api/cameras/test-rtsp")
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


@router.post("/api/cameras")
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
        camera_manager.register_camera_config(saved_camera)
        if saved_camera["enabled"]:
            camera_manager.connect_camera(saved_camera["camera_id"])
        else:
            # 19082026 - KIET - Dừng instance cũ nếu camera được cập nhật về trạng thái disable.
            camera_manager.disconnect_camera(saved_camera["camera_id"])
        return camera_manager.get_camera_status(saved_camera["camera_id"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/cameras/{camera_id}/connect")
def connect_configured_camera(camera_id: str):
    """
    19082026 - KIET - Connect một camera đã lưu mà không ảnh hưởng camera khác.
    """

    camera_config = get_camera_config(camera_id)
    if camera_config is None:
        raise HTTPException(status_code=404, detail="Camera config not found")

    try:
        camera_manager.register_camera_config(camera_config)
        camera_manager.connect_camera(camera_id)
        updated_camera = update_camera_config(camera_id, {"enabled": True})
        if updated_camera is not None:
            camera_manager.register_camera_config(updated_camera)
        return camera_manager.get_camera_status(camera_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/cameras/{camera_id}/disconnect")
def disconnect_configured_camera(camera_id: str):
    """
    19082026 - KIET - Disconnect một camera và giữ nguyên cấu hình để dùng lại.
    """

    if get_camera_config(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera config not found")

    camera_manager.disconnect_camera(camera_id)
    updated_camera = update_camera_config(camera_id, {"enabled": False})
    if updated_camera is not None:
        camera_manager.register_camera_config(updated_camera)
    return camera_manager.get_camera_status(camera_id)


@router.patch("/api/cameras/{camera_id}/assignment")
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

    camera_manager.register_camera_config(updated_camera)
    return camera_manager.get_camera_status(camera_id)


@router.get("/api/cameras/{camera_id}/status")
def get_configured_camera_status(camera_id: str):
    """
    19082026 - KIET - Trả trạng thái online và lỗi gần nhất của một camera.
    """

    try:
        return camera_manager.get_camera_status(camera_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc