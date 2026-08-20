import io
import os
import re
import sys
import requests
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from services.database.session import SessionLocal
from services.database.models import AIModel, InspectionRecord, SyncState
from packages.core.config import AppConfig

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
cfg = AppConfig.from_yaml(os.path.join(str(ROOT), "config.yaml"))

router = APIRouter(tags=["System & Deployment"])
CLOUD_API_BASE = "http://127.0.0.1:8031"


# ── Dataset Stats ─────────────────────────────────────────────────────────────

@router.get("/dataset-stats")
async def get_dataset_stats():
    db = SessionLocal()
    try:
        db.commit()
        records = db.query(InspectionRecord).order_by(InspectionRecord.id.desc()).all()
        total = len(records)
        images_list = []
        for r in records[:50]:
            ts_val = r.timestamp
            date_str = ts_val
            if "_" in ts_val:
                parts = ts_val.split('_')
                if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
                    d, t = parts[0], parts[1]
                    date_str = f"{d[6:8]}/{d[4:6]}/{d[0:4]} {t[0:2]}:{t[2:4]}:{t[4:6]}"
            images_list.append({
                "id": r.id,
                "name": r.original_image.split('/')[-1].split('\\')[-1] if r.original_image else f"IMG_{r.timestamp}.jpg",
                "status": "labeled",
                "type": "NG" if r.ng_detected else "OK",
                "date": date_str,
                "product": cfg.PRODUCT_NAME
            })
        return {
            "stats": [
                {"label": "Total Images", "value": str(total), "color": "#3b82f6"},
                {"label": "Labeled",      "value": str(total), "color": "#10b981"},
                {"label": "Synthetic",    "value": "0",        "color": "#8b5cf6"},
                {"label": "Pending",      "value": "0",        "color": "#f59e0b"},
            ],
            "images": images_list
        }
    except Exception as e:
        print(f"Error fetching dataset stats: {e}")
        return {"stats": [], "images": []}
    finally:
        db.close()


# ── Model Registry ────────────────────────────────────────────────────────────

@router.get("/model-registry")
async def get_model_registry():
    db = SessionLocal()
    try:
        models = db.query(AIModel).all()
        model_list = [{
            "id": m.id, "name": m.name, "type": m.type,
            "status": "active" if m.status == "PRODUCTION" else m.status.lower(),
            "mAP": f"{m.map_acc}%" if m.map_acc is not None else "N/A",
            "speed": m.speed_ms, "format": m.format, "version": m.version
        } for m in models]

        active_model = (db.query(AIModel)
                        .filter(AIModel.status == "PRODUCTION")
                        .order_by(AIModel.version.desc())
                        .first()) or (models[0] if models else None)

        active_engine_label, active_map = "N/A", 0.0
        if active_model:
            status_label = "Stable" if active_model.status == "PRODUCTION" else active_model.status.capitalize()
            active_engine_label = f"{active_model.version} {status_label}"
            active_map = float(active_model.map_acc or 0.0)

        recent_latencies = (db.query(InspectionRecord.latency_ms)
                              .order_by(InspectionRecord.id.desc()).limit(20).all())
        pipeline_latency = (round(sum(r[0] for r in recent_latencies) / len(recent_latencies), 2)
                            if recent_latencies else 0.0)

        def parse_speed_ms(value: str) -> float:
            if not value:
                return 0.0
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
            return float(m.group(1)) if m else 0.0

        type_colors = {
            "Detection": '#3b82f6', "Alignment": '#10b981', "Segmentation": '#6366f1',
            "Anomaly": '#f59e0b', "Classification": '#f43f5e', "Keypoints": '#8b5cf6'
        }

        latencyData = [{"stage": m.type, "time": parse_speed_ms(m.speed_ms),
                        "color": type_colors.get(m.type, '#94a3b8')}
                       for m in models if parse_speed_ms(m.speed_ms) > 0]
        if not latencyData:
            latencyData = [
                {"stage": "Detection",      "time": 0, "color": '#3b82f6'},
                {"stage": "Alignment",      "time": 0, "color": '#10b981'},
                {"stage": "Segmentation",   "time": 0, "color": '#6366f1'},
                {"stage": "Anomaly",        "time": 0, "color": '#f59e0b'},
                {"stage": "Classification", "time": 0, "color": '#f43f5e'},
            ]

        total_from_models = round(sum(item['time'] for item in latencyData), 2)
        if pipeline_latency == 0.0 and total_from_models > 0:
            pipeline_latency = total_from_models

        cpu_load = psutil.cpu_percent(interval=0.1)
        vram_usage = min(100.0, cpu_load * 1.2 + (np.random.random() * 5.0))
        npu_load = min(100.0, 40.0 + (np.random.random() * 40.0))

        return {
            "activeEngine": active_engine_label,
            "pipelineLatency": pipeline_latency,
            "mAP": round(active_map, 1),
            "models": model_list,
            "latencyData": latencyData,
            "resourceData": [
                {"name": "GPU Memory", "value": round(vram_usage, 1),       "color": '#2563eb'},
                {"name": "Free",       "value": round(100 - vram_usage, 1), "color": '#e2e8f0'},
            ],
            "cpuThreads": round(cpu_load, 1),
            "npuLoad": round(npu_load, 1)
        }
    except Exception as e:
        print(f"Error fetching models: {e}")
        return {"models": []}
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
