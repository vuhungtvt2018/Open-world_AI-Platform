import os
import sys
import ast
import glob
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import APIRouter
from fastapi.responses import FileResponse, Response
from services.database.session import SessionLocal
from services.database.models import InspectionRecord, BoltObject, AnomalyDetail
from packages.core.config import AppConfig

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
cfg = AppConfig.from_yaml(os.path.join(str(ROOT), "config.yaml"))

router = APIRouter(tags=["Analytics"])


@router.get("/analytics")
async def get_analytics(date: str = None):
    """date: YYYY-MM-DD string in local time (UTC+7). Defaults to today."""
    db = SessionLocal()
    try:
        db.commit()

        if date:
            try:
                target_local = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                target_local = datetime.utcnow() + timedelta(hours=7)
                target_local = target_local.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            target_local = datetime.utcnow() + timedelta(hours=7)
            target_local = target_local.replace(hour=0, minute=0, second=0, microsecond=0)

        day_start_utc = target_local - timedelta(hours=7)
        day_end_utc = day_start_utc + timedelta(days=1)

        records = db.query(InspectionRecord).filter(
            InspectionRecord.created_at >= day_start_utc,
            InspectionRecord.created_at < day_end_utc
        ).all()

        hourly_data = {f"{i:02d}h": {"ok": 0, "ng": 0} for i in range(24)}
        for r in records:
            if not r.created_at:
                continue
            local_hour = (r.created_at.hour + 7) % 24  # UTC -> UTC+7
            hour_str = f"{local_hour:02d}h"
            for obj in r.objects:
                if obj.is_ng:
                    hourly_data[hour_str]["ng"] += 1
                else:
                    hourly_data[hour_str]["ok"] += 1

        hourlyOutputData = []
        yieldTrendData = []
        for i in range(24):
            hour_str = f"{i:02d}h"
            time_str = f"{i:02d}:00"
            ok_count = hourly_data[hour_str]["ok"]
            ng_count = hourly_data[hour_str]["ng"]
            total = ok_count + ng_count
            if total > 0 or (8 <= i <= 17):
                hourlyOutputData.append({"hour": hour_str, "ok": ok_count, "ng": ng_count})
                rate = round((ok_count / total * 100) if total > 0 else 0, 1)
                yieldTrendData.append({"time": time_str, "rate": rate})

        total_ok = sum(h["ok"] for h in hourly_data.values())
        total_ng = sum(h["ng"] for h in hourly_data.values())
        total_inspected = total_ok + total_ng
        overall_yield = round((total_ok / total_inspected * 100) if total_inspected > 0 else 100.0, 1)

        defect_counts = {}
        for r in records:
            for obj in r.objects:
                for a in obj.anomalies:
                    cls_name = a.defect_class or "Unknown"
                    defect_counts[cls_name] = defect_counts.get(cls_name, 0) + 1

        pareto = []
        colors = ['#f43f5e', '#fb923c', '#facc15', '#38bdf8', '#94a3b8']
        for i, (name, count) in enumerate(sorted(defect_counts.items(), key=lambda x: x[1], reverse=True)):
            pareto.append({"name": name, "count": count, "color": colors[i % len(colors)]})

        if not pareto:
            pareto = [{"name": "No Defects", "count": 1, "color": "#10b981"}]

        return {
            "totalInspected": total_inspected,
            "overallYield": overall_yield,
            "totalNg": total_ng,
            "avgCycleTime": 1.2,
            "hourlyOutputData": hourlyOutputData,
            "yieldTrendData": yieldTrendData,
            "pareto": pareto
        }
    except Exception as e:
        print(f"Error fetching analytics: {e}")
        return {
            "totalInspected": 0, "overallYield": 100.0,
            "totalNg": 0, "avgCycleTime": 0.0,
            "pareto": [], "hourlyOutputData": [], "yieldTrendData": []
        }
    finally:
        db.close()


@router.get("/history")
async def get_history():
    db = SessionLocal()
    try:
        db.commit()
        records = db.query(InspectionRecord).order_by(InspectionRecord.id.asc()).all()
        results = []
        for r in records:
            objs = []
            for o in r.objects:
                anoms = []
                for a in o.anomalies:
                    try:
                        bf = ast.literal_eval(a.bbox_full) if a.bbox_full else []
                    except Exception:
                        bf = []
                    try:
                        bi = ast.literal_eval(a.bbox_in_object_crop) if a.bbox_in_object_crop else []
                    except Exception:
                        bi = []
                    anoms.append({
                        "bbox_full": bf,
                        "bbox_in_object_crop": bi,
                        "cls_label": a.defect_class,
                        "cls_similarity": a.similarity
                    })
                try:
                    obbox = ast.literal_eval(o.bbox) if o.bbox else []
                except Exception:
                    obbox = []
                objs.append({
                    "index": o.object_index,
                    "bbox": obbox,
                    "score": o.score,
                    "is_ng": o.is_ng,
                    "overlap_ratio": o.overlap_ratio,
                    "crop": o.crop_image,
                    "anomalies": anoms,
                    "anomaly_count": len(anoms),
                    "anomaly_bboxes_sample": [a["bbox_full"] for a in anoms[:3]]
                })
            results.append({
                "timestamp": r.timestamp,
                "image": r.original_image,
                "ng_detected": r.ng_detected,
                "latency_ms": r.latency_ms,
                "total_objects": len(objs),
                "ng_count": sum(1 for ob in objs if ob["is_ng"]),
                "max_score": max([ob["score"] for ob in objs]) if objs else 0.0,
                "objects": objs
            })
        return {"results": results}
    except Exception as e:
        print(f"Error fetching history: {e}")
        return {"results": []}
    finally:
        db.close()


@router.get("/api/image/{image_name}")
async def get_inspection_image(image_name: str):
    pattern = os.path.join(cfg.CAPTURE_DIR, cfg.PRODUCT_NAME, "sessions", "*", "original", image_name)
    matches = glob.glob(pattern)
    if matches:
        return FileResponse(matches[0])
    return Response(status_code=404)


@router.get("/api/result_image/{image_name}")
async def get_result_image(image_name: str):
    timestamp = image_name.replace("IMG_", "").replace(".jpg", "")
    overall_name = f"OVERALL_{timestamp}.jpg"
    pattern = os.path.join(cfg.CAPTURE_DIR, cfg.PRODUCT_NAME, "sessions", "*", overall_name)
    matches = glob.glob(pattern)
    if matches:
        return FileResponse(matches[0])
    return Response(status_code=404)
