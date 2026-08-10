from fastapi import FastAPI

from packages.utils.cleaner import run_cleaner_daemon
from services.database.crud import create_inspection_record
from services.database.session import SessionLocal, init_db

from .app_config import cfg

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


def register_startup_event(app: FastAPI):
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

        default_event_bus.subscribe(
            EventBus.EVENT_INFERENCE_DONE,
            save_db_callback,
        )
