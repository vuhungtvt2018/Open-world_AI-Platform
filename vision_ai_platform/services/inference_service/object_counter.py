import tempfile
import uuid
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Body, HTTPException, Path, UploadFile
from fastapi.responses import StreamingResponse

from vision_ai_platform.packages.core import WorkflowConfig
from vision_ai_platform.packages.utils import LOGGER, colorstr
from vision_ai_platform.packages.workflow import (
    ImageObjectCounterWorkflow,
    VideoObjectCounterWorkflow,
    Workflow,
)

# Global in-memory storage for active workflow instances
WORKFLOWS: Dict[str, Dict[str, Workflow]] = {
    "image": {},
    "video": {},
}

router = APIRouter(
    prefix="/obj_counter",
    tags=["obj_counter"],
)


@router.get("/")
def root():
    """Root endpoint listing available Object Counter services."""
    LOGGER.info("Root endpoint accessed")
    return {
        "service": "Object Counter service",
        "router_name": "obj_counter",
        "types": ["image", "video"],
    }


@router.get("/health")
def health():
    """Health check endpoint showing currently active workflow instances."""
    return {
        "status": "ok",
        "num_img": len(WORKFLOWS["image"]),
        "num_video": len(WORKFLOWS["video"]),
        "service": "Object Counter service",
    }


@router.get("/image/")
def list_img_workflows():
    """List all initialized Image Object Counter workflows."""
    img_wf_dict = WORKFLOWS["image"]
    img_wf_list = []
    for wf_key, workflow in img_wf_dict.items():
        img_wf_list.append({
            "key": wf_key,
            "config": workflow.cfg.model_dump(),
            "counted_ids_count": len(workflow.counted_ids),
            "margin": workflow.margin,
        })
    return img_wf_list


@router.get("/video/")
def list_video_workflows():
    """List all initialized Video Object Counter workflows."""
    video_wf_dict = WORKFLOWS["video"]
    video_wf_list = []
    for wf_key, workflow in video_wf_dict.items():
        video_wf_list.append({
            "key": wf_key,
            "config": workflow.cfg.model_dump(),
            "in_count": workflow.in_count,
            "out_count": workflow.out_count,
            "init_region": workflow.region_initialized,
            "margin": workflow.margin,
        })
    return video_wf_list


@router.get("/image/{wf_key}")
def get_info_wf_image(wf_key: str = Path(..., description="The ID of the image workflow")):
    """Get detailed information about a specific Image Object Counter workflow."""
    if wf_key not in WORKFLOWS["image"]:
        raise HTTPException(404, detail=f"Image object counter ID '{wf_key}' not found")

    workflow: ImageObjectCounterWorkflow = WORKFLOWS["image"][wf_key]
    return {
        "key": wf_key,
        "config": workflow.cfg.model_dump(),
        "classwise_count": dict(workflow.classwise_count),
        "counted_ids_count": len(workflow.counted_ids),
        "margin": workflow.margin,
    }


@router.get("/video/{wf_key}")
def get_info_wf_video(wf_key: str = Path(..., description="The ID of the video workflow")):
    """Get detailed information about a specific Video Object Counter workflow."""
    if wf_key not in WORKFLOWS["video"]:
        raise HTTPException(404, detail=f"Video object counter ID '{wf_key}' not found")

    workflow: VideoObjectCounterWorkflow = WORKFLOWS["video"][wf_key]
    return {
        "key": wf_key,
        "config": workflow.cfg.model_dump(),
        "in_count": workflow.in_count,
        "out_count": workflow.out_count,
        "classwise_count": dict(workflow.classwise_count),
        "init_region": workflow.region_initialized,
        "margin": workflow.margin,
    }


@router.post("/image/setup")
async def setup_wf_image(config_override: Optional[Dict[str, Any]] = Body(None)):
    """Set up a new Image Object Counter workflow."""
    wf_key = str(uuid.uuid4())
    try:
        wf_config = WorkflowConfig()
        if config_override:
            wf_config.update(**config_override)

        workflow = ImageObjectCounterWorkflow(wf_config)
        WORKFLOWS["image"][wf_key] = workflow

        LOGGER.info(f"Image object counter workflow ID {colorstr(wf_key)} successfully set up")
        return {
            "success": True,
            "key": wf_key,
            "task": "object_counter",
            "type": "image",
            "config": wf_config.model_dump(),
        }
    except Exception as e:
        LOGGER.error(f"Error initializing image workflow: {str(e)}")
        raise HTTPException(500, detail=f"Error while setting up workflow for image: {str(e)}")


@router.post("/video/setup")
async def setup_wf_video(config_override: Optional[Dict[str, Any]] = Body(None)):
    """Set up a new Video Object Counter workflow."""
    wf_key = str(uuid.uuid4())
    try:
        wf_config = WorkflowConfig()
        if config_override:
            wf_config.update(**config_override)

        workflow = VideoObjectCounterWorkflow(wf_config)
        WORKFLOWS["video"][wf_key] = workflow

        LOGGER.info(f"Video object counter workflow ID {colorstr(wf_key)} successfully set up")
        return {
            "success": True,
            "key": wf_key,
            "task": "object_counter",
            "type": "video",
            "config": wf_config.model_dump(),
        }
    except Exception as e:
        LOGGER.error(f"Error initializing video workflow: {str(e)}")
        raise HTTPException(500, detail=f"Error while setting up workflow for video: {str(e)}")


@router.put("/image/{wf_key}")
def change_wf_image(
    wf_key: str = Path(...),
    config_updates: Dict[str, Any] = Body(..., description="Key-value pairs to update in WorkflowConfig"),
):
    """Update configuration parameters for an active image workflow."""
    if wf_key not in WORKFLOWS["image"]:
        raise HTTPException(404, detail=f"Image object counter ID {wf_key} not found")

    workflow = WORKFLOWS["image"][wf_key]
    try:
        workflow.cfg.update(**config_updates)
        LOGGER.info(f"Image object counter ID {colorstr(wf_key)}")
        return {
            "success": True,
            "key": wf_key,
            "config": workflow.cfg.model_dump(),
        }
    except ValueError as ve:
        raise HTTPException(400, detail=str(ve))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to update config: {str(e)}")


@router.put("/video/{wf_key}")
def change_wf_video(
    wf_key: str = Path(...),
    config_updates: Dict[str, Any] = Body(..., description="Key-value pairs to update in WorkflowConfig"),
):
    """Update configuration parameters for an active video workflow."""
    if wf_key not in WORKFLOWS["video"]:
        raise HTTPException(404, detail=f"Video object counter ID '{wf_key}' not found")

    workflow = WORKFLOWS["video"][wf_key]
    try:
        workflow.cfg.update(**config_updates)
        # Reset region initialization to force recalculation if region updated
        if "region" in config_updates:
            workflow.region = workflow.cfg.region
            workflow.initialize_region()
            workflow.region_initialized = True

        return {
            "success": True,
            "key": wf_key,
            "updated_config": workflow.cfg.model_dump(),
        }
    except ValueError as ve:
        raise HTTPException(400, detail=str(ve))
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to update config: {str(e)}")


@router.delete("/image/{wf_key}")
def remove_wf_image(wf_key: str = Path(...)):
    """Remove an image workflow instance from memory."""
    if wf_key not in WORKFLOWS["image"]:
        raise HTTPException(404, detail=f"Image object counter ID '{wf_key}' not found")

    del WORKFLOWS["image"][wf_key]
    LOGGER.info(f"Removed Image workflow {wf_key}")
    return {"success": True, "message": f"Image workflow '{wf_key}' successfully removed"}


@router.delete("/video/{wf_key}")
def remove_wf_video(wf_key: str = Path(...)):
    """Remove a video workflow instance from memory."""
    if wf_key not in WORKFLOWS["video"]:
        raise HTTPException(404, detail=f"Video object counter ID '{wf_key}' not found")

    del WORKFLOWS["video"][wf_key]
    LOGGER.info(f"Removed Video workflow {wf_key}")
    return {"success": True, "message": f"Video workflow '{wf_key}' successfully removed"}


@router.post("/image/{wf_key}")
async def count_image(
    wf_key: str = Path(...),
    file: UploadFile = UploadFile(...),
):
    """Process an image file through the workflow and return the annotated output image."""
    if wf_key not in WORKFLOWS["image"]:
        raise HTTPException(404, detail=f"Image object counter ID '{wf_key}' not found")

    workflow = WORKFLOWS["image"][wf_key]

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if im0 is None:
            raise HTTPException(400, detail="Invalid image file format")

        # Execute object counting pipeline
        results = workflow.process(im0)

        # Encode resulting plot frame to JPEG byte response
        success, encoded_img = cv2.imencode(".jpg", results.plot_im)
        if not success:
            raise HTTPException(500, detail="Failed to encode resulting image")

        return StreamingResponse(
            iter([encoded_img.tobytes()]),
            media_type="image/jpeg",
            headers={"X-Classwise-Count": str(results.classwise_count)},
        )
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Error processing image: {str(e)}")
        raise HTTPException(500, detail=f"Error processing image: {str(e)}")


@router.post("/video/{wf_key}")
async def count_video(
    wf_key: str = Path(...),
    file: UploadFile = UploadFile(...),
):
    """Stream MJPEG annotated counting results from an uploaded video file."""
    if wf_key not in WORKFLOWS["video"]:
        raise HTTPException(404, detail=f"Video object counter ID '{wf_key}' not found")

    workflow = WORKFLOWS["video"][wf_key]

    # Save uploaded bytes to a temporary file for OpenCV VideoCapture compatibility
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        contents = await file.read()
        temp_file.write(contents)
        temp_file.close()

        def frame_generator():
            cap = cv2.VideoCapture(temp_file.name)
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = workflow.process(frame)

                    success, encoded_frame = cv2.imencode(".jpg", results.plot_im)
                    if not success:
                        continue

                    # Construct standard MJPEG boundary frame payload
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + encoded_frame.tobytes()
                        + b"\r\n"
                    )
            finally:
                cap.release()

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        LOGGER.error(f"Error processing video stream: {str(e)}")
        raise HTTPException(500, detail=f"Error streaming video processing: {str(e)}")