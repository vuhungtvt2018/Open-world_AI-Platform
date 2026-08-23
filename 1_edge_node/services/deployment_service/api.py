import io
import os
import re
import sys
import requests
import psutil
from collections import Counter
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import StreamingResponse
from services.database.session import SessionLocal
from services.database.models import AIModel, InspectionRecord, SyncState
from packages.core.config import AppConfig
from services.database.crud import (
    acknowledge_camera_configs,
    list_camera_configs,
    apply_cloud_camera_configs,
)
from services.camera_service.core import camera_manager

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
cfg = AppConfig.from_yaml(os.path.join(str(ROOT), "config.yaml"))

router = APIRouter(tags=["System & Deployment"])
CLOUD_API_BASE = "http://127.0.0.1:8031"


"""
23082026 - KHAI - Add functions related to camera registry
"""
# ── Camera Registry ─────────────────────────────────────────────────────────────

# 23082026-KIET-Chuyển camera config Edge thành payload Camera Hub dùng chung
def _camera_sync_payload(camera: dict) -> dict:
    return {
        "camera_id": camera["camera_id"],
        "name": camera["name"],
        "source_type": camera["source_type"],
        "source_url": camera.get("source_url"),
        "serial_number": camera.get("serial_number"),
        "assigned_task": camera.get("assigned_task"),
        "enabled": camera.get("enabled", False),
        "width": camera.get("width"),
        "height": camera.get("height"),
        "fps": camera.get("fps"),
        "created_at": camera.get("created_at"),
        "updated_at": camera.get("updated_at"),
        "base_revision": int(camera.get("cloud_revision", 0)),
    }


# 23082026-KIET-Push các camera config mới hoặc vừa thay đổi từ Edge lên Camera Hub
def _push_dirty_camera_configs() -> dict:
    dirty_cameras = list_camera_configs(dirty_only=True)
    if not dirty_cameras:
        return {"pushed_count": 0, "acknowledged_count": 0}

    edge_node_id = cfg.EDGE_CODE or "EDGE_001"
    response = requests.post(
        f"{CLOUD_API_BASE}/api/cameras/sync-up",
        json={
            "edge_node_id": edge_node_id,
            "cameras": [_camera_sync_payload(camera) for camera in dirty_cameras],
        },
        timeout=10,
    )
    response.raise_for_status()
    response_data = response.json()
    acknowledged_count = acknowledge_camera_configs(
        response_data.get("cameras", [])
    )
    return {
        "pushed_count": response_data.get("synced_count", 0),
        "acknowledged_count": acknowledged_count,
    }


# 23082026-KIET-Pull camera config theo Edge ID và cập nhật camera runtime khi cần
def _pull_camera_configs_from_hub() -> dict:
    edge_node_id = cfg.EDGE_CODE or "EDGE_001"
    response = requests.get(
        f"{CLOUD_API_BASE}/api/cameras/sync-down/{edge_node_id}",
        timeout=10,
    )
    response.raise_for_status()
    response_data = response.json()
    applied_cameras = apply_cloud_camera_configs(
        response_data.get("cameras", [])
    )

    runtime_errors = []
    for camera in applied_cameras:
        try:
            camera_manager.register_camera_config(camera)
            if camera.get("enabled", False):
                camera_manager.connect_camera(camera["camera_id"])
            else:
                camera_manager.disconnect_camera(camera["camera_id"])
        except Exception as exc:
            runtime_errors.append({
                "camera_id": camera["camera_id"],
                "message": str(exc),
            })

    return {
        "pulled_count": len(applied_cameras),
        "runtime_errors": runtime_errors,
    }


@router.post("/api/sync/cameras")
async def sync_camera_configs_with_hub():
    """23082026-KIET-Push camera mới lên Hub trước rồi pull config đúng Edge về local."""

    try:
        push_result = _push_dirty_camera_configs()
        pull_result = _pull_camera_configs_from_hub()
        return {
            "status": "success",
            "edge_node_id": cfg.EDGE_CODE or "EDGE_001",
            **push_result,
            **pull_result,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Camera Hub sync failed: {exc}",
        ) from exc


# ── Dataset Stats ─────────────────────────────────────────────────────────────

"""
23082026 - KHAI - Modify output of endpoint /dataset-stats
"""
@router.get("/dataset-stats")
async def get_dataset_stats():
    """
    12082026 - KIET - Trả danh sách dataset có metadata riêng cho Inspection và Detection.
    """

    db = SessionLocal()
    try:
        db.commit() # Clear cached transaction
        records = db.query(InspectionRecord).order_by(InspectionRecord.id.desc()).all()
        total = len(records)
        
        images_list = []
        for r in records[:50]: # Lấy 50 ảnh gần nhất
            ts = r.timestamp
            date_str = ts
            if "_" in ts:
                parts = ts.split('_')
                if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
                    d, t = parts[0], parts[1]
                    date_str = f"{d[6:8]}/{d[4:6]}/{d[0:4]} {t[0:2]}:{t[2:4]}:{t[4:6]}"

            # 12082026 - KIET - Không gắn record Detection thành kết quả OK của Inspection.
            record_type = (
                "Detection"
                if r.task_type == "detection"
                else ("NG" if r.ng_detected else "OK")
            )
            counts_by_class = {}
            if r.task_type == "detection":
                counts_by_class = dict(
                    Counter(
                        obj.class_name or "unknown"
                        for obj in r.objects
                    )
                )
                
            images_list.append({
                "id": r.id,
                "name": r.original_image.split('/')[-1].split('\\')[-1] if r.original_image else f"IMG_{r.timestamp}.jpg",
                "status": "labeled",
                "type": record_type,
                "task_type": r.task_type,
                "total_objects": r.total_objects,
                "counts_by_class": counts_by_class,
                "date": date_str,
                "product": cfg.PRODUCT_NAME
            })
            
        return {
            "stats": [
                {"label": "Total Images", "value": str(total), "color": "#3b82f6"},
                {"label": "Labeled", "value": str(total), "color": "#10b981"},
                {"label": "Synthetic", "value": "0", "color": "#8b5cf6"},
                {"label": "Pending", "value": "0", "color": "#f59e0b"}
            ],
            "images": images_list
        }
    except Exception as e:
        print(f"Error fetching dataset stats: {e}")
        return {"stats": [], "images": []}
    finally:
        db.close()


# ── Model Registry ────────────────────────────────────────────────────────────

"""
23082026 - Modify implementation of get_model_registry
"""
def _parse_model_speed_ms(speed_value) -> float:
    """
    22082026 - KIET - Chuyển metadata speed_ms của từng model thành số millisecond.
    """

    if speed_value is None:
        return 0.0

    match = re.search(r"-?\d+(?:\.\d+)?", str(speed_value))
    if match is None:
        return 0.0

    return max(0.0, float(match.group(0)))


@router.get("/model-registry")
async def get_model_registry():
    """
    22082026 - KIET - Trả model registry kèm latency và resource metrics theo schema ổn định.
    """
    db = SessionLocal()
    try:
        models = db.query(AIModel).all()
        model_list = []
        for m in models:
            model_list.append({
                "id": m.id,
                "name": m.name,
                "type": m.type,
                "status": "active" if m.status == "PRODUCTION" else m.status.lower(),
                "mAP": f"{m.map_acc}%" if m.map_acc else "N/A",
                "speed": m.speed_ms,
                "format": m.format,
                "version": m.version
            })            
        
        # Real-time hardware metrics for resourceData
        cpu_load = psutil.cpu_percent(interval=0.1)
        vram_usage = min(100.0, cpu_load * 1.2 + (np.random.random() * 5.0)) # Simulated VRAM
        npu_load = min(100.0, 40.0 + (np.random.random() * 40.0))
        
        resourceData = [
            { "name": 'GPU Memory', "value": round(vram_usage, 1), "color": '#2563eb' },
            { "name": 'Free', "value": round(100 - vram_usage, 1), "color": '#e2e8f0' },
        ]
        
        # 22082026 - KIET - Dựng latency chart từ speed_ms của model hiện có sau khi sync.
        latency_colors = [
            "#3b82f6",
            "#10b981",
            "#6366f1",
            "#f59e0b",
            "#f43f5e",
            "#8b5cf6",
        ]
        latencyData = [
            {
                "stage": f"{model.type} ({model.id})",
                "time": _parse_model_speed_ms(model.speed_ms),
                "color": latency_colors[index % len(latency_colors)],
                "modelId": model.id,
                "modelName": model.name,
                "modelType": model.type,
            }
            for index, model in enumerate(models)
        ]
        pipeline_latency = round(
            sum(item["time"] for item in latencyData),
            2,
        )

        valid_map_values = [
            float(model.map_acc)
            for model in models
            if model.map_acc is not None
        ]
        registry_map = round(
            sum(valid_map_values) / len(valid_map_values),
            2,
        ) if valid_map_values else 0.0

        return {
            "activeEngine": "v2.4.1 Stable",
            "pipelineLatency": pipeline_latency,
            "mAP": registry_map,
            "models": model_list,
            "latencyData": latencyData,
            "resourceData": resourceData,
            "cpuThreads": round(cpu_load, 1),
            "npuLoad": round(npu_load, 1)
        }
    except Exception as e:
        print(f"Error fetching models: {e}")
        # 22082026 - KIET - Giữ đủ field response khi database hoặc hardware metrics gặp lỗi.
        return {
            "activeEngine": "Unavailable",
            "pipelineLatency": 0,
            "mAP": 0,
            "models": [],
            "latencyData": [],
            "resourceData": [
                {"name": "GPU Memory", "value": 0, "color": "#2563eb"},
                {"name": "Free", "value": 100, "color": "#e2e8f0"},
            ],
            "cpuThreads": 0,
            "npuLoad": 0,
        }
    finally:
        db.close()


# ── System Health ─────────────────────────────────────────────────────────────

@router.get("/system-health")
async def get_system_health():
    cpu_load = psutil.cpu_percent(interval=0.1)
    ram_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent
    core_temp = 42.0 + (cpu_load * 0.35) + (np.random.random() * 2.0)
    gpu_load = min(100.0, cpu_load * 1.2 + (np.random.random() * 5.0))
    return {
        "cpu_load":   round(cpu_load, 1),
        "ram_usage":  round(ram_usage, 1),
        "disk_usage": round(disk_usage, 1),
        "temperature": round(core_temp, 1),
        "gpu_load":   round(gpu_load, 1)
    }


# ── Cloud-Edge Sync ───────────────────────────────────────────────────────────

@router.post("/api/sync/dataset-up")
async def sync_dataset_up():
    db = SessionLocal()
    try:
        sync_state = db.query(SyncState).filter(SyncState.key == "last_synced_record_id").first()
        last_id = int(sync_state.value) if sync_state and sync_state.value else 0
        new_records = (db.query(InspectionRecord)
                         .filter(InspectionRecord.id > last_id)
                         .order_by(InspectionRecord.id.asc()).limit(100).all())
        if not new_records:
            return {"status": "success", "message": "Already up to date", "synced_count": 0}
        payload = {"records": [{"edge_node_id": cfg.EDGE_CODE, "timestamp": r.timestamp,
                                 "ng_detected": r.ng_detected, "total_objects": len(r.objects)}
                                for r in new_records]}
        res = requests.post(f"{CLOUD_API_BASE}/api/dataset/sync", json=payload, timeout=10)
        res.raise_for_status()
        highest_id = new_records[-1].id
        if not sync_state:
            db.add(SyncState(key="last_synced_record_id", value=str(highest_id)))
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
    db = SessionLocal()
    try:
        models = db.query(AIModel).all()
        if not models:
            return {"status": "success", "synced_count": 0, "message": "No models to sync"}
        payload = {"models": [{"id": m.id, "name": m.name, "type": m.type, "format": m.format,
                                "version": m.version, "map_acc": float(m.map_acc) if m.map_acc else 0.0,
                                "status": m.status, "file_path": m.file_path, "speed_ms": m.speed_ms or ""}
                               for m in models]}
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
    db = SessionLocal()
    try:
        res = requests.get(f"{CLOUD_API_BASE}/api/models/latest", timeout=10)
        res.raise_for_status()
        cloud_models = res.json().get("models", [])
        synced_count = 0
        for cm in cloud_models:
            local_m = db.query(AIModel).filter(AIModel.id == cm["id"]).first()
            if not local_m:
                db.add(AIModel(id=cm["id"], name=cm["name"], type=cm["type"], format=cm["format"],
                               version=cm["version"], map_acc=cm["map_acc"], status="STANDBY",
                               file_path=f"cloud_downloads/{cm['name']}", speed_ms=cm["speed_ms"]))
                synced_count += 1
            elif local_m.version != cm["version"]:
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


# ── Reports ───────────────────────────────────────────────────────────────────

@router.get("/api/reports/lookup/{item_code}")
async def report_lookup(item_code: str):
    db = SessionLocal()
    try:
        from sqlalchemy import text
        insp_count = db.execute(text(f"SELECT COUNT(*) FROM inspection_records WHERE \"ItemCode\" = '{item_code}'")).scalar()
        qc_count   = db.execute(text(f"SELECT COUNT(*) FROM qc_product_photo_library WHERE \"ItemCode\" = '{item_code}'")).scalar()
        reports = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        if insp_count and insp_count > 0:
            reports.append({"id": f"REP-INSP-{item_code}", "name": f"Inspection Records - {item_code}",
                             "type": "System", "format": "Excel", "date": now_str,
                             "size": f"{insp_count} rows", "downloads": 0, "lastDownloaded": "-"})
        if qc_count and qc_count > 0:
            reports.append({"id": f"REP-QC-{item_code}", "name": f"QC Photo Library - {item_code}",
                             "type": "System", "format": "Excel", "date": now_str,
                             "size": f"{qc_count} rows", "downloads": 0, "lastDownloaded": "-"})
        return {"status": "success", "reports": reports}
    finally:
        db.close()


@router.get("/api/reports/download/{table_name}/{item_code}")
async def report_download(table_name: str, item_code: str):
    db = SessionLocal()
    try:
        if table_name not in ["inspection_records", "qc_product_photo_library"]:
            return Response(status_code=400, content="Invalid table")
        try:
            import pandas as pd
            df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE \"ItemCode\" = '{item_code}'", db.bind)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name=item_code)
            output.seek(0)
            headers = {'Content-Disposition': f'attachment; filename="{table_name}_{item_code}.xlsx"',
                       'Access-Control-Expose-Headers': 'Content-Disposition'}
            return StreamingResponse(output, headers=headers,
                                     media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except ImportError:
            import csv
            from sqlalchemy import text
            rows = db.execute(text(f"SELECT * FROM {table_name} WHERE \"ItemCode\" = '{item_code}'")).mappings().all()
            if not rows:
                return Response(status_code=404, content="No data found")
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
            headers = {'Content-Disposition': f'attachment; filename="{table_name}_{item_code}.csv"',
                       'Access-Control-Expose-Headers': 'Content-Disposition'}
            return Response(content=output.getvalue(), media_type="text/csv", headers=headers)
    except Exception as e:
        print(f"Error generating report: {e}")
        return Response(status_code=500, content=f"Error generating report: {e}")
    finally:
        db.close()
