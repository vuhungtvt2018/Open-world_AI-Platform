# apps/edge-agent/web_backend.py  –  Entry point only
from pathlib import Path
import sys
import os

# Resolve ROOT (1_edge_node/) – 2 levels up from apps/edge-agent/
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from packages.core.config import AppConfig
from services.database.session import init_db, SessionLocal
from services.database.crud import create_inspection_record
from services.camera_service.core import camera_manager
from packages.utils.cleaner import run_cleaner_daemon

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
            counting_model = getattr(cfg, "MODEL_COUNTING_PATH", None) or cfg.MODEL_PATH
            models = [
                AIModel(id="m-det-01",  name=os.path.basename(counting_model),
                        type="Detection", format=counting_model.split('.')[-1].upper(),
                        version="1.0.0", map_acc=91.4, status="PRODUCTION",
                        file_path=counting_model, speed_ms="42ms"),
                AIModel(id="m-anom-01", name=os.path.basename(cfg.ANOMALY_MODEL_PATH),
                        type="Anomaly",    format=cfg.ANOMALY_MODEL_PATH.split('.')[-1].upper(),
                        version="1.0.0", map_acc=89.5, status="PRODUCTION",
                        file_path=cfg.ANOMALY_MODEL_PATH, speed_ms="20ms"),
                AIModel(id="m-kpt-01",  name=os.path.basename(cfg.KEYPOINTS_MODEL_PATH),
                        type="Keypoints",  format=cfg.KEYPOINTS_MODEL_PATH.split('.')[-1].upper(),
                        version="1.0.0", map_acc=95.0, status="PRODUCTION",
                        file_path=cfg.KEYPOINTS_MODEL_PATH, speed_ms="12ms"),
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
    """
    23082026 - KHAI - Restore enabled camera
    """
    camera_manager.load_saved_camera_configs()
    seed_models()
    run_cleaner_daemon(
        captures_dir=cfg.CAPTURE_DIR,
        days_to_keep=cfg.DISK_CLEANUP_DAYS,
        interval_hours=cfg.DISK_CLEANUP_INTERVAL_HOURS,
    )
    from packages.workflow.events import default_event_bus, EventBus
    def save_db_callback(record: dict):
        try:
            create_inspection_record(record)
        except Exception as e:
            print(f"[ERROR] Failed to insert DB record via EventBus: {e}")
    default_event_bus.subscribe(EventBus.EVENT_INFERENCE_DONE, save_db_callback)


"""
23082026 - KHAI - Free camera when shutting down web backend.
"""
@app.on_event("shutdown")
def shutdown_web_backend() -> None:
    for camera_id in list(camera_manager.cameras.keys()):
        camera_manager.disconnect_camera(camera_id)


app.mount("/captures", StaticFiles(directory=cfg.CAPTURE_DIR), name="captures")

SYNC_DEFECT_IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sync_defect_image"))
os.makedirs(SYNC_DEFECT_IMAGE_DIR, exist_ok=True)
app.mount("/sync_images", StaticFiles(directory=SYNC_DEFECT_IMAGE_DIR), name="sync_images")

# ── Include routers from services/ ──────────────────────────────────────────
# pyrefly: ignore [missing-import]
from services.camera_service import api as camera_service_api
from services.inference_service import api as inference_service_api
from services.analytics_service import api as analytics_service_api
from services.deployment_service import api as deployment_service_api

app.include_router(camera_service_api.router)
app.include_router(inference_service_api.router)
app.include_router(analytics_service_api.router)
app.include_router(deployment_service_api.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
