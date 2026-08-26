from fastapi import APIRouter
from services.inference_service.core import inference_engine

router = APIRouter(tags=["Inference"])


@router.post("/inspect")
async def trigger_inspect(cam: str = "default"):
    return inference_engine.run_inspect(cam)


@router.post("/detect")
async def trigger_detection(cam: str = "default"):
    return inference_engine.run_detection(cam)
