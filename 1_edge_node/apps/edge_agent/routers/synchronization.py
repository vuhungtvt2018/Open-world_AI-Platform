import requests
from fastapi import APIRouter, Response

from services.database.session import SessionLocal

from ..app_config import cfg

router = APIRouter()

CLOUD_API_BASE = "http://127.0.0.1:8031" # Default defect classification server port for testing

@router.post("/api/sync/dataset-up")
async def sync_dataset_up():
    from services.database.models import InspectionRecord, SyncState
    db = SessionLocal()
    try:
        # Get last synced id
        sync_state = db.query(SyncState).filter(SyncState.key == "last_synced_record_id").first()
        last_id = int(sync_state.value) if sync_state and sync_state.value else 0
        
        # Get new records
        new_records = db.query(InspectionRecord).filter(InspectionRecord.id > last_id).order_by(InspectionRecord.id.asc()).limit(100).all()
        
        if not new_records:
            return {"status": "success", "message": "Already up to date", "synced_count": 0}
            
        payload = {
            "records": [
                {
                    "edge_node_id": cfg.EDGE_CODE,
                    "timestamp": r.timestamp,
                    "ng_detected": r.ng_detected,
                    "total_objects": len(r.objects)
                } for r in new_records
            ]
        }
        
        # Send to Cloud
        res = requests.post(f"{CLOUD_API_BASE}/api/dataset/sync", json=payload, timeout=10)
        res.raise_for_status()
        
        # Update last synced id
        highest_id = new_records[-1].id
        if not sync_state:
            sync_state = SyncState(key="last_synced_record_id", value=str(highest_id))
            db.add(sync_state)
        else:
            sync_state.value = str(highest_id)
            
        db.commit()
        return {"status": "success", "synced_count": len(new_records)}
    except Exception as e:
        print(f"Sync Up Error: {e}")
        return Response(status_code=500, content=f"Sync Up Error: {e}")
    finally:
        db.close()

@router.post("/api/sync/models-up")
async def sync_models_up():
    from services.database.models import AIModel
    db = SessionLocal()
    try:
        models = db.query(AIModel).all()
        if not models:
            return {"status": "success", "synced_count": 0, "message": "No models to sync"}
            
        payload = {
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "type": m.type,
                    "format": m.format,
                    "version": m.version,
                    "map_acc": float(m.map_acc) if m.map_acc else 0.0,
                    "status": m.status,
                    "file_path": m.file_path,
                    "speed_ms": m.speed_ms or ""
                } for m in models
            ]
        }
        
        # Send to Cloud
        res = requests.post(f"{CLOUD_API_BASE}/api/models/sync-up", json=payload, timeout=10)
        res.raise_for_status()
        
        return {"status": "success", "synced_count": len(models)}
    except Exception as e:
        print(f"Sync Models Up Error: {e}")
        return Response(status_code=500, content=f"Sync Models Up Error: {e}")
    finally:
        db.close()

@router.post("/api/sync/models-down")
async def sync_models_down():
    from services.database.models import AIModel
    db = SessionLocal()
    try:
        res = requests.get(f"{CLOUD_API_BASE}/api/models/latest", timeout=10)
        res.raise_for_status()
        data = res.json()
        
        cloud_models = data.get("models", [])
        synced_count = 0
        
        for cm in cloud_models:
            local_m = db.query(AIModel).filter(AIModel.id == cm["id"]).first()
            if not local_m:
                # Add new model
                new_m = AIModel(
                    id=cm["id"],
                    name=cm["name"],
                    type=cm["type"],
                    format=cm["format"],
                    version=cm["version"],
                    map_acc=cm["map_acc"],
                    status="STANDBY", # Downloaded but not activated
                    file_path=f"cloud_downloads/{cm['name']}",
                    speed_ms=cm["speed_ms"]
                )
                db.add(new_m)
                synced_count += 1
            elif local_m.version != cm["version"]:
                # Update existing model metadata
                local_m.version = cm["version"]
                local_m.map_acc = cm["map_acc"]
                local_m.speed_ms = cm["speed_ms"]
                synced_count += 1
                
        db.commit()
        return {"status": "success", "synced_count": synced_count}
    except Exception as e:
        print(f"Sync Down Error: {e}")
        return Response(status_code=500, content=f"Sync Down Error: {e}")
    finally:
        db.close()

import io
from fastapi.responses import StreamingResponse
from datetime import datetime
