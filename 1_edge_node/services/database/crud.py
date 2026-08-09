from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
from .models import InspectionRecord, InspectionObject, AnomalyDetail
from .session import SessionLocal

def create_inspection_record(record_data: Dict[str, Any]):
    """
    Tạo bản ghi InspectionRecord kèm theo các InspectionObject và AnomalyDetail.
    record_data có dạng:
    {
        "timestamp": "...",
        "image": "...",
        "ng_detected": bool,
        "latency_ms": 0.0,
        "objects": [
            {
                "product_id": int,
                "index": int,
                "bbox": [x1, y1, x2, y2],
                "score": float,
                "is_ng": bool,
                "overlap_ratio": float,
                "crop": "...",
                "anomalies": [
                    {
                        "bbox_full": [...],
                        "bbox_in_object_crop": [...],
                        "cls_label": "...",
                        "cls_similarity": float
                    }
                ]
            }
        ]
    }
    """
    db: Session = SessionLocal()
    try:
        db_record = InspectionRecord(
            timestamp=record_data["timestamp"],
            original_image=record_data["image"],
            ng_detected=record_data.get("ng_detected", False),
            latency_ms=record_data.get("latency_ms", 0.0)
        )
        
        for obj_data in record_data.get("objects", []):
            db_obj = InspectionObject(
                product_id=obj_data["product_id"],
                object_index=obj_data.get("index", 0),
                bbox=str(obj_data.get("bbox", [])),
                score=obj_data.get("score", 0.0),
                is_ng=obj_data.get("is_ng", False),
                overlap_ratio=obj_data.get("overlap_ratio", 0.0),
                crop_image=obj_data.get("crop", "")
            )
            
            for anom_data in obj_data.get("anomalies", []):
                db_anom = AnomalyDetail(
                    bbox_full=str(anom_data.get("bbox_full", [])),
                    bbox_in_object_crop=str(anom_data.get("bbox_in_object_crop", [])),
                    defect_class=anom_data.get("cls_label"),
                    similarity=anom_data.get("cls_similarity")
                )
                db_obj.anomalies.append(db_anom)
                
            db_record.objects.append(db_obj)
            
        db.add(db_record)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] Failed to save inspection record: {e}")
    finally:
        db.close()
