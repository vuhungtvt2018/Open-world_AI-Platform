from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

if __package__ in (None, ""):
    package_parent = Path(__file__).resolve().parent.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    from edge_agent.app_config import SYNC_DEFECT_IMAGE_DIR, cfg
    from edge_agent.lifecycle import register_startup_event
    from edge_agent.routers import (
        inspection,
        monitoring,
        reports,
        synchronization,
    )
else:
    from .app_config import SYNC_DEFECT_IMAGE_DIR, cfg
    from .lifecycle import register_startup_event
    from .routers import inspection, monitoring, reports, synchronization

app = FastAPI(title="Visual Inspection AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_startup_event(app)

app.mount(
    "/captures",
    StaticFiles(directory=cfg.CAPTURE_DIR),
    name="captures",
)
app.mount(
    "/sync_images",
    StaticFiles(directory=SYNC_DEFECT_IMAGE_DIR),
    name="sync_images",
)

app.include_router(inspection.router)
app.include_router(monitoring.router)
app.include_router(synchronization.router)
app.include_router(reports.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
