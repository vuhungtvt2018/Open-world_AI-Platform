from fastapi import APIRouter
from services.inference_service.core import inference_engine

router = APIRouter(tags=["Inspection"])


@router.post("/inspect")
async def trigger_inspect(cam: str = "default"):
    return inference_engine.run_inspect(cam)
