import cv2
import time
import numpy as np
from fastapi import APIRouter, UploadFile, File, Response, HTTPException
from fastapi.responses import StreamingResponse
from services.camera_service.core import camera_manager

router = APIRouter(tags=["Camera"])


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
async def video_feed(cam: str = "default"):
    def gen_frames():
        while True:
            frame = camera_manager.get_frame(cam)
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
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
