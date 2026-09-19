import json

from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
from .models import InspectionRecord, InspectionObject, AnomalyDetail, CameraConfig
from .session import SessionLocal

def create_inspection_record(record_data: Dict[str, Any]):
    """
    11082026 - KIET - Lưu record inspection hoặc Object Detection vào database cũ.

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
            original_image=record_data.get("image", ""),
            ng_detected=record_data.get("ng_detected", False),
            latency_ms=record_data.get("latency_ms", 0.0),
            # 11082026 - KIET - Lưu metadata Object Detection trên record cũ.
            task_type=record_data.get("task_type", "inspection"),
            camera_id=record_data.get("camera_id"),
            total_objects=record_data.get(
                "total_objects",
                len(record_data.get("objects", [])),
            ),
        )
        
        for obj_data in record_data.get("objects", []):
            db_obj = InspectionObject(
                product_id=obj_data["product_id"],
                object_index=obj_data.get("index", 0),
                # 11082026 - KIET - Lưu bbox bằng JSON và tương thích dữ liệu cũ khi đọc.
                bbox=json.dumps(obj_data.get("bbox", [])),
                score=obj_data.get("score", 0.0),
                is_ng=obj_data.get("is_ng", False),
                overlap_ratio=obj_data.get("overlap_ratio", 0.0),
                crop_image=obj_data.get("crop", ""),
                # 11082026 - KIET - Lưu class để phục vụ counting theo class.
                class_id=obj_data.get("class_id"),
                class_name=obj_data.get("class_name"),
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


def camera_config_to_dict(camera: CameraConfig) -> Dict[str, Any]:
    """
    19082026 - KIET - Chuyển CameraConfig ORM thành payload dùng cho API camera.
    """

    return {
        "camera_id": camera.camera_id,
        "name": camera.name,
        "source_type": camera.source_type,
        "source_url": camera.source_url,
        "serial_number": camera.serial_number,
        "assigned_task": camera.assigned_task,
        "enabled": camera.enabled,
        "width": camera.width,
        "height": camera.height,
        "fps": camera.fps,
        "sync_dirty": camera.sync_dirty,
        "cloud_revision": camera.cloud_revision,
        "last_synced_at": str(camera.last_synced_at) if camera.last_synced_at else None,
        "created_at": str(camera.created_at) if camera.created_at else None,
        "updated_at": str(camera.updated_at) if camera.updated_at else None,
    }


def list_camera_configs(dirty_only: bool = False) -> List[Dict[str, Any]]:
    """
    19082026 - KIET - Lấy toàn bộ cấu hình camera đã lưu trên Edge database.
    """

    db: Session = SessionLocal()
    try:
        query = db.query(CameraConfig)
        if dirty_only:
            # 23082026-KIET-Chỉ lấy camera config có thay đổi cục bộ chưa được Hub xác nhận
            query = query.filter(CameraConfig.sync_dirty.is_(True))
        cameras = query.order_by(CameraConfig.camera_id.asc()).all()
        return [camera_config_to_dict(camera) for camera in cameras]
    finally:
        db.close()


def get_camera_config(camera_id: str) -> Dict[str, Any] | None:
    """
    19082026 - KIET - Lấy cấu hình của một camera theo camera ID.
    """

    db: Session = SessionLocal()
    try:
        camera = db.get(CameraConfig, camera_id)
        return camera_config_to_dict(camera) if camera is not None else None
    finally:
        db.close()


def upsert_camera_config(
    camera_data: Dict[str, Any],
    mark_dirty: bool = True,
) -> Dict[str, Any]:
    """
    19082026 - KIET - Tạo mới hoặc cập nhật cấu hình RTSP/Basler từ Live Stream UI.
    """

    db: Session = SessionLocal()
    try:
        camera_id = str(camera_data["camera_id"])
        camera = db.get(CameraConfig, camera_id)

        if camera is None:
            camera = CameraConfig(camera_id=camera_id, name=camera_data["name"], source_type=camera_data["source_type"])
            db.add(camera)

        editable_fields = (
            "name",
            "source_type",
            "source_url",
            "serial_number",
            "assigned_task",
            "enabled",
            "width",
            "height",
            "fps",
        )
        for field_name in editable_fields:
            if field_name in camera_data:
                setattr(camera, field_name, camera_data[field_name])

        # 23082026-KIET-Phân biệt thay đổi từ UI Edge và cấu hình vừa pull từ Camera Hub
        camera.sync_dirty = mark_dirty
        if not mark_dirty:
            camera.cloud_revision = int(camera_data.get("revision", camera.cloud_revision or 0))
            camera.last_synced_at = datetime.now()

        db.commit()
        db.refresh(camera)
        return camera_config_to_dict(camera)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_camera_config(camera_id: str, changes: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    19082026 - KIET - Cập nhật assignment, trạng thái hoặc metadata của camera đã lưu.
    """

    db: Session = SessionLocal()
    try:
        camera = db.get(CameraConfig, camera_id)
        if camera is None:
            return None

        editable_fields = (
            "name",
            "source_type",
            "source_url",
            "serial_number",
            "assigned_task",
            "enabled",
            "width",
            "height",
            "fps",
        )
        for field_name in editable_fields:
            if field_name in changes:
                setattr(camera, field_name, changes[field_name])

        # 23082026-KIET-Đánh dấu thay đổi camera tại Edge để worker hoặc nút Sync đẩy lên Hub
        camera.sync_dirty = True

        db.commit()
        db.refresh(camera)
        return camera_config_to_dict(camera)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# 23082026-KIET-Xác nhận các camera config đã được Camera Hub lưu thành công
def acknowledge_camera_configs(camera_revisions: List[Dict[str, Any]]) -> int:
    db: Session = SessionLocal()
    try:
        acknowledged_count = 0
        for item in camera_revisions:
            camera = db.get(CameraConfig, str(item["camera_id"]))
            if camera is None:
                continue
            client_updated_at = item.get("client_updated_at")
            if client_updated_at and str(camera.updated_at) != str(client_updated_at):
                # 23082026-KIET-Không xóa dirty nếu camera lại thay đổi trong lúc request đang chạy
                continue
            camera.cloud_revision = int(item.get("revision", camera.cloud_revision or 0))
            camera.sync_dirty = False
            camera.last_synced_at = datetime.now()
            acknowledged_count += 1
        db.commit()
        return acknowledged_count
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# 23082026-KIET-Upsert camera config từ Hub nhưng không tạo vòng lặp push ngược lại
def apply_cloud_camera_configs(camera_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    applied_configs: List[Dict[str, Any]] = []
    for camera_data in camera_configs:
        local_camera = get_camera_config(str(camera_data["camera_id"]))
        remote_revision = int(camera_data.get("revision", 0))
        local_revision = int(local_camera.get("cloud_revision", 0)) if local_camera else -1
        if local_camera is not None and local_camera.get("sync_dirty", False):
            # 23082026-KIET-Không ghi đè thay đổi local phát sinh giữa bước push và pull
            continue
        if local_camera is not None and remote_revision <= local_revision:
            continue
        applied_configs.append(
            upsert_camera_config(camera_data, mark_dirty=False)
        )
    return applied_configs
