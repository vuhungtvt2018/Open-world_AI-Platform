from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import select, text, bindparam, func
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from config import settings
from typing import Optional, List
from uuid import uuid4
from datetime import datetime
import os, io, json
import dateutil.parser

from db import Base, engine, SessionLocal, ensure_pgvector
from models import QCProductPhotoLibrary, CloudAIModel, InspectionRecord, BoltObject, AnomalyDetail
from schemas import (
    PhotoCreate, PhotoRead, PhotoUpdateErrorDetail,
    SearchRequestByImage, SearchResultItem
)
from embedding import embed_pil
from PIL import Image

app = FastAPI(title="AI Visual Inspection - Photo Library & Vector DB")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB init ---

# 17082026 - KIET - Chỉ tạo bảng model khi test sync bằng SQLite
if engine.dialect.name == "sqlite":
    CloudAIModel.__table__.create(
        bind=engine,
        checkfirst=True
    )

# 17082026 - KIET - Giữ cơ chế tạo đầy đủ database khi dùng PostgreSQL
else:
    ensure_pgvector()
    Base.metadata.create_all(bind=engine)

# --- IVFFLAT index tạo như cũ (cosine) ---
if settings.ENABLE_IVFFLAT_INDEX and  engine.dialect.name == "postgresql":
    dist = settings.IVFFLAT_DISTANCE.lower()
    metric = {
        "l2": "vector_l2_ops",
        "cosine": "vector_cosine_ops",
        "ip": "vector_ip_ops",
    }.get(dist, "vector_cosine_ops")
    with engine.begin() as conn:
        conn.execute(text(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE schemaname = 'public'
                AND indexname = 'idx_{settings.TABLE_NAME}_embedding_ivfflat'
            ) THEN
                CREATE INDEX idx_{settings.TABLE_NAME}_embedding_ivfflat
                ON {settings.TABLE_NAME} USING ivfflat (embedding {metric})
                WITH (lists = {settings.IVFFLAT_LISTS});
            END IF;
        END $$;
        """))

# --- Ensure upload dir exists ---
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _metric_sql(metric: str) -> str:
    m = metric.lower()
    if m == "cosine":
        return "<=>"
    if m == "ip":
        return "<#>"
    return "<->"

def _gen_filename(ext: str = ".jpg") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid4().hex[:16]}{ext}"

@app.post("/photos", response_model=PhotoRead)
async def create_photo(
    SM_ID: Optional[str] = Form(None),
    ItemCode: Optional[str] = Form(None),
    ImageType: Optional[int] = Form(None),
    ErrorDetail: Optional[str] = Form(None),
    Insert_PIC: Optional[str] = Form(None),
    Update_PIC: Optional[str] = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if SM_ID:
        exists = db.execute(
            select(QCProductPhotoLibrary.ID).where(QCProductPhotoLibrary.SM_ID == SM_ID)
        ).scalar_one_or_none()
        if exists is not None:
            raise HTTPException(status_code=409, detail="SM_ID đã tồn tại")

    content = await file.read()
    try:
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File ảnh không hợp lệ")

    base, ext = os.path.splitext(file.filename or "")
    if not ext or len(ext) > 5:
        ext = ".jpg"
    fname = _gen_filename(ext)
    save_dir = os.path.join(settings.UPLOAD_DIR, ItemCode) if ItemCode else settings.UPLOAD_DIR
    os.makedirs(save_dir, exist_ok=True)
    pil.save(os.path.join(save_dir, fname))

    vec = embed_pil(pil)

    rec = QCProductPhotoLibrary(
        SM_ID=SM_ID,
        ItemCode=ItemCode,
        ImageName=fname,
        ImageType=ImageType,
        ErrorDetail=ErrorDetail,
        Insert_PIC=Insert_PIC,
        Update_PIC=Update_PIC,
        embedding=vec,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return PhotoRead.model_validate(rec)

@app.get("/photos/{photo_id}", response_model=PhotoRead)
def get_photo(photo_id: int, db: Session = Depends(get_db)):
    rec = db.get(QCProductPhotoLibrary, photo_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy")
    return PhotoRead.model_validate(rec)

@app.get("/photos/by-smid/{sm_id}", response_model=PhotoRead)
def get_photo_by_smid(sm_id: str, db: Session = Depends(get_db)):
    rec = db.execute(
        select(QCProductPhotoLibrary).where(QCProductPhotoLibrary.SM_ID == sm_id)
    ).scalar_one_or_none()
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy")
    return PhotoRead.model_validate(rec)

@app.get("/photos", response_model=List[PhotoRead])
def list_photos(
    item_code: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    stmt = select(QCProductPhotoLibrary).offset(skip).limit(limit)
    if item_code:
        stmt = select(QCProductPhotoLibrary).where(
            QCProductPhotoLibrary.ItemCode == item_code
        ).offset(skip).limit(limit)

    rows = db.execute(stmt).scalars().all()
    return [PhotoRead.model_validate(r) for r in rows]

@app.put("/photos/{photo_id}/error-detail", response_model=PhotoRead)
def update_error_detail(
    photo_id: int,
    payload: PhotoUpdateErrorDetail,
    db: Session = Depends(get_db),
):
    rec = db.get(QCProductPhotoLibrary, photo_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy")

    rec.ErrorDetail = payload.ErrorDetail
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return PhotoRead.model_validate(rec)

@app.put("/photos/by-smid/{sm_id}/error-detail", response_model=PhotoRead)
def update_error_detail_by_smid(
    sm_id: str,
    payload: PhotoUpdateErrorDetail,
    db: Session = Depends(get_db),
):
    rec = db.execute(
        select(QCProductPhotoLibrary).where(QCProductPhotoLibrary.SM_ID == sm_id)
    ).scalar_one_or_none()
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy")

    rec.ErrorDetail = payload.ErrorDetail
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return PhotoRead.model_validate(rec)

@app.delete("/photos/{photo_id}")
def delete_photo(photo_id: int, db: Session = Depends(get_db)):
    rec = db.get(QCProductPhotoLibrary, photo_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy")
    db.delete(rec)
    db.commit()
    return JSONResponse({"deleted": photo_id})

@app.delete("/photos/by-smid/{sm_id}")
def delete_photo_by_smid(sm_id: str, db: Session = Depends(get_db)):
    rec = db.execute(
        select(QCProductPhotoLibrary).where(QCProductPhotoLibrary.SM_ID == sm_id)
    ).scalar_one_or_none()
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy")
    db.delete(rec)
    db.commit()
    return JSONResponse({"deleted_smid": sm_id})

@app.get("/api/analytics")
def get_analytics(db: Session = Depends(get_db)):
    # 1. Defect Pareto from QCProductPhotoLibrary
    counts = db.execute(
        select(QCProductPhotoLibrary.ErrorDetail, func.count(QCProductPhotoLibrary.ID))
        .group_by(QCProductPhotoLibrary.ErrorDetail)
    ).all()
    
    pareto = []
    colors = ['#f43f5e', '#fb923c', '#facc15', '#38bdf8', '#94a3b8']
    
    for idx, row in enumerate(counts):
        err = row[0] or "Unknown"
        cnt = row[1]
        pareto.append({
            "name": err,
            "count": cnt,
            "color": colors[idx % len(colors)]
        })
        
    pareto = sorted(pareto, key=lambda x: x["count"], reverse=True)
    
    # 2. Production stats from InspectionRecord
    records = db.execute(select(InspectionRecord)).scalars().all()
    
    total_inspected = len(records)
    total_ng = sum(1 for r in records if r.ng_detected)
    
    hourly_stats = {}
    
    for r in records:
        try:
            ts_val = r.timestamp or str(r.created_at)
            dt = None
            if isinstance(ts_val, (int, float)):
                dt = datetime.fromtimestamp(ts_val)
            else:
                dt = dateutil.parser.parse(str(ts_val))
                
            hour_str = f"{dt.hour:02d}h"
            if hour_str not in hourly_stats:
                hourly_stats[hour_str] = {"ok": 0, "ng": 0}
                
            if r.ng_detected:
                hourly_stats[hour_str]["ng"] += 1
            else:
                hourly_stats[hour_str]["ok"] += 1
                
        except Exception:
            pass
            
    overall_yield = ((total_inspected - total_ng) / total_inspected * 100) if total_inspected > 0 else 100.0
            
    hourly_output_data = []
    yield_trend_data = []
    
    for hour in sorted(hourly_stats.keys()):
        ok_count = hourly_stats[hour]["ok"]
        ng_count = hourly_stats[hour]["ng"]
        total = ok_count + ng_count
        hourly_output_data.append({
            "hour": hour,
            "ok": ok_count,
            "ng": ng_count
        })
        rate = (ok_count / total * 100) if total > 0 else 100.0
        yield_trend_data.append({
            "time": hour.replace("h", ":00"),
            "rate": round(rate, 1)
        })
        
    if not hourly_output_data:
        hourly_output_data = [{"hour": "08h", "ok": 0, "ng": 0}]
        yield_trend_data = [{"time": "08:00", "rate": 100.0}]

    return {
        "totalInspected": total_inspected,
        "overallYield": round(overall_yield, 1),
        "totalNg": total_ng,
        "avgCycleTime": 2.45,
        "pareto": pareto,
        "hourlyOutputData": hourly_output_data,
        "yieldTrendData": yield_trend_data
    }

@app.post("/search/by-image", response_model=List[SearchResultItem])
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    metric: str = Form("cosine"),
    ItemCode: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    content = await file.read()
    try:
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File ảnh không hợp lệ")

    qvec = embed_pil(pil)
    op = _metric_sql(metric)

    where = 'WHERE embedding IS NOT NULL'
    params = {"qvec": qvec, "k": top_k}
    if ItemCode:
        where += ' AND "ItemCode" = :item_code'
        params["item_code"] = ItemCode

    sql = f"""
        SELECT "ID", "SM_ID", "ItemCode", "ImageName",
            (embedding {op} :qvec) AS distance,
            "ErrorDetail"
        FROM {settings.TABLE_NAME}
        {where}
        ORDER BY embedding {op} :qvec
        LIMIT :k
    """

    stmt = text(sql).bindparams(
        bindparam("qvec", type_=Vector(settings.EMBEDDING_DIM)),
        bindparam("k"),
        *( [bindparam("item_code")] if "item_code" in where else [] )
    )

    rows = db.execute(stmt, {"qvec": qvec, "k": top_k, **({"item_code": ItemCode} if ItemCode else {})}).mappings().all()

    total_in_lib = db.execute(text(f'SELECT COUNT(*) FROM {settings.TABLE_NAME}')).scalar() or 0
    if len(rows) == 0 and ItemCode and total_in_lib > 0:
        sql_fallback = f"""
            SELECT "ID", "SM_ID", "ItemCode", "ImageName",
                (embedding {op} :qvec) AS distance,
                "ErrorDetail"
            FROM {settings.TABLE_NAME}
            WHERE embedding IS NOT NULL
            ORDER BY embedding {op} :qvec
            LIMIT :k
        """
        stmt_fallback = text(sql_fallback).bindparams(
            bindparam("qvec", type_=Vector(settings.EMBEDDING_DIM)),
            bindparam("k")
        )
        rows = db.execute(stmt_fallback, {"qvec": qvec, "k": top_k}).mappings().all()

    return [
        SearchResultItem(
            ID=r["ID"],
            SM_ID=r["SM_ID"],
            distance=float(r["distance"]),
            ItemCode=r["ItemCode"],
            ImageName=r["ImageName"],
            ErrorDetail=r["ErrorDetail"],
        )
        for r in rows
    ]

# ==========================================
# EDGE-TO-CLOUD SYNC ENDPOINTS
# ==========================================

@app.post("/api/sync/up")
async def sync_up(
    payload: str = Form(...),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    try:
        data = json.loads(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    print(f"\n[SERVER SYNC UP] Nhan duoc du lieu tu Edge Node:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    file_path = ""
    if file:
        sync_dir = os.path.join(settings.UPLOAD_DIR, "synced_images")
        os.makedirs(sync_dir, exist_ok=True)
        file_path = os.path.join(sync_dir, file.filename or "unknown.jpg")
        with open(file_path, "wb") as f:
            f.write(await file.read())

    try:
        # Create InspectionRecord
        new_record = InspectionRecord(
            edge_node_id=data.get("edge_node_id", "unknown"),
            edge_record_id=data.get("record_id"),
            item_code=data.get("item_code"),
            timestamp=data.get("timestamp"),
            original_image=file_path,
            ng_detected=data.get("ng_detected", False),
            latency_ms=data.get("latency_ms", 0.0),
            created_at=data.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Parse objects
        for obj_data in data.get("objects", []):
            bolt = BoltObject(
                edge_bolt_id=obj_data.get("id"),
                object_index=obj_data.get("object_index", 0),
                bbox=obj_data.get("bbox", ""),
                score=obj_data.get("score", 0.0),
                is_ng=obj_data.get("is_ng", False),
                overlap_ratio=obj_data.get("overlap_ratio", 0.0),
                crop_image=obj_data.get("crop_image", "")
            )
            new_record.objects.append(bolt)
            
            # Parse anomalies
            for ano_data in obj_data.get("anomalies", []):
                anomaly = AnomalyDetail(
                    edge_anomaly_id=ano_data.get("id"),
                    bbox_full=ano_data.get("bbox_full", ""),
                    bbox_in_object_crop=ano_data.get("bbox_in_object_crop", ""),
                    defect_class=ano_data.get("defect_class", ""),
                    similarity=ano_data.get("similarity", 0.0)
                )
                bolt.anomalies.append(anomaly)
                
        db.add(new_record)
        db.commit()
        print(f"[SERVER SYNC UP] Da luu thanh cong InspectionRecord tu Edge vao Server DB.")
    except Exception as e:
        print(f"[SERVER SYNC UP] Loi khi luu xuong DB: {e}")
        db.rollback()

    return {"status": "success", "message": "Synced successfully"}


@app.get("/api/sync/down")
async def sync_down():
    config_file = os.path.join(os.path.dirname(__file__), "edge_config_distribution.json")
    
    if not os.path.exists(config_file):
        default_config = {
            "command": "update_config",
            "data": {
                "DEFECT_CLS_SIM_THRESHOLD": 0.1,
                "DISK_CLEANUP_DAYS": 15.0
            }
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
            
    with open(config_file, "r", encoding="utf-8") as f:
        config_update = json.load(f)
        
    return config_update

@app.get("/api/sync/library/down")
def sync_library_down(last_update: Optional[str] = Query(None), db: Session = Depends(get_db)):
    stmt = select(QCProductPhotoLibrary)
    if last_update:
        stmt = stmt.where(QCProductPhotoLibrary.Update_Date > last_update)
    
    rows = db.execute(stmt).scalars().all()
    
    import base64
    results = []
    for r in rows:
        image_base64 = None
        if r.ImageName:
            img_path = os.path.join(settings.UPLOAD_DIR, r.ItemCode, r.ImageName) if r.ItemCode else os.path.join(settings.UPLOAD_DIR, r.ImageName)
            if not os.path.exists(img_path):
                img_path = os.path.join(settings.UPLOAD_DIR, r.ImageName)
            if os.path.exists(img_path):
                try:
                    with open(img_path, "rb") as img_file:
                        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                except Exception:
                    pass

        results.append({
            "ID": r.ID,
            "SM_ID": r.SM_ID,
            "ItemCode": r.ItemCode,
            "ImageName": r.ImageName,
            "ImageType": r.ImageType,
            "ErrorDetail": r.ErrorDetail,
            "Insert_PIC": r.Insert_PIC,
            "Insert_Date": str(r.Insert_Date) if r.Insert_Date else None,
            "Update_PIC": r.Update_PIC,
            "Update_Date": str(r.Update_Date) if r.Update_Date else None,
            "image_base64": image_base64,
        })
        
    return {"status": "success", "data": results}


# --- AI Models APIs ---
from pydantic import BaseModel

@app.get("/api/models/latest")
def get_latest_models(db: Session = Depends(get_db)):
    models = db.query(CloudAIModel).filter(CloudAIModel.status == "PRODUCTION").all()
    return {
        "status": "success",
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "type": m.type,
                "format": m.format,
                "version": m.version,
                "map_acc": m.map_acc,
                "speed_ms": m.speed_ms
            } for m in models
        ]
    }

class SyncModelRequest(BaseModel):
    id: str
    name: str
    type: str
    format: str
    version: str
    map_acc: float = None
    status: str
    file_path: str
    speed_ms: str = None

class SyncModelsPayload(BaseModel):
    models: List[SyncModelRequest]

@app.post("/api/models/sync-up")
def sync_models_up(req: SyncModelsPayload, db: Session = Depends(get_db)):
    try:
        updated_count = 0
        for m in req.models:
            existing = db.query(CloudAIModel).filter(CloudAIModel.id == m.id).first()
            if not existing:
                new_model = CloudAIModel(
                    id=m.id, name=m.name, type=m.type, format=m.format, 
                    version=m.version, map_acc=m.map_acc, status=m.status,
                    file_path=m.file_path, speed_ms=m.speed_ms
                )
                db.add(new_model)
                updated_count += 1
            else:
                if existing.version != m.version or existing.status != m.status:
                    existing.version = m.version
                    existing.map_acc = m.map_acc
                    existing.status = m.status
                    existing.speed_ms = m.speed_ms
                    updated_count += 1
        db.commit()
        return {"status": "success", "synced_count": updated_count}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
