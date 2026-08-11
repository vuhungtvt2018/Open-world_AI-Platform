from sqlalchemy import Integer, String, Float, Boolean, DateTime, ForeignKey, Text, SmallInteger
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func, text as sqltext
from typing import List
import json
from .session import Base

class InspectionRecord(Base):
    """
    11082026 - KIET - Đại diện cho một lần inspection hoặc Object Detection.
    """
    __tablename__ = "inspection_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(50), index=True)
    original_image: Mapped[str] = mapped_column(String(255))
    ng_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())

    # 11082026 - KIET - Phân biệt record inspection cũ với Object Detection.
    task_type: Mapped[str] = mapped_column(String(30), default="inspection", index=True)

    # 11082026 - KIET - Lưu camera tạo ra kết quả inference.
    camera_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # 11082026 - KIET - Lưu tổng số object được phát hiện trong frame.
    total_objects: Mapped[int] = mapped_column(Integer, default=0)
    
    # Quan hệ 1-N với BoltObject
    objects: Mapped[List["BoltObject"]] = relationship(
        "BoltObject", back_populates="record", cascade="all, delete-orphan"
    )

class BoltObject(Base):
    """
    11082026 - KIET - Đại diện cho một object inspection hoặc Object Detection.
    """
    __tablename__ = "bolt_objects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[int] = mapped_column(ForeignKey("inspection_records.id"), index=True)
    object_index: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[str] = mapped_column(String(255))  # Chứa chuỗi JSON [x1, y1, x2, y2]
    score: Mapped[float] = mapped_column(Float, default=0.0)
    is_ng: Mapped[bool] = mapped_column(Boolean, default=False)
    overlap_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    crop_image: Mapped[str] = mapped_column(String(255))

    # 11082026 - KIET - Lưu class ID của object từ model Detection.
    class_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # 11082026 - KIET - Lưu class name để thống kê số lượng theo class.
    class_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    
    record: Mapped["InspectionRecord"] = relationship("InspectionRecord", back_populates="objects")
    
    # Quan hệ 1-N với AnomalyDetail
    anomalies: Mapped[List["AnomalyDetail"]] = relationship(
        "AnomalyDetail", back_populates="bolt", cascade="all, delete-orphan"
    )

class AnomalyDetail(Base):
    """Bảng lưu chi tiết từng vết lỗi trên con bu-lông"""
    __tablename__ = "anomaly_details"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bolt_id: Mapped[int] = mapped_column(ForeignKey("bolt_objects.id"), index=True)
    bbox_full: Mapped[str] = mapped_column(String(255))           # [x1, y1, x2, y2] toàn cảnh
    bbox_in_object_crop: Mapped[str] = mapped_column(String(255)) # [cx1, cy1, cx2, cy2] trong crop
    defect_class: Mapped[str] = mapped_column(String(100), nullable=True) # Nhãn phân loại lỗi
    similarity: Mapped[float] = mapped_column(Float, nullable=True)       # Độ tin cậy (Cosine similarity)
    
    bolt: Mapped["BoltObject"] = relationship("BoltObject", back_populates="anomalies")

class SyncState(Base):
    """Bảng quản lý trạng thái đồng bộ dữ liệu (Edge -> Cloud)"""
    __tablename__ = "sync_states"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(String(255))
    updated_at: Mapped[str] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

class AIModel(Base):
    """Bảng quản lý Model (Model Registry)"""
    __tablename__ = "ai_models"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    type: Mapped[str] = mapped_column(String(50))     # Detection, Anomaly, Alignment, Classification
    format: Mapped[str] = mapped_column(String(20))   # PT, ONNX, IR...
    version: Mapped[str] = mapped_column(String(20))
    map_acc: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20))   # PRODUCTION, STAGING, INACTIVE
    file_path: Mapped[str] = mapped_column(String(255))
    speed_ms: Mapped[str] = mapped_column(String(20), nullable=True)

class QCProductPhotoLibrary(Base):
    """Metadata cho CSDL mẫu của thư viện Vector Search (Edge ko lưu embedding)"""
    __tablename__ = "qc_product_photo_library"

    ID: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    SM_ID: Mapped[str | None] = mapped_column(String(100))
    ItemCode: Mapped[str | None] = mapped_column(String(50))
    ImageName: Mapped[str | None] = mapped_column(Text)
    ImageType: Mapped[int | None] = mapped_column(SmallInteger)
    ErrorDetail: Mapped[str | None] = mapped_column(Text)
    Insert_PIC: Mapped[str | None] = mapped_column(String(50))
    Insert_Date: Mapped[str | None] = mapped_column(DateTime(timezone=False), server_default=func.now())
    Update_PIC: Mapped[str | None] = mapped_column(String(50))
    Update_Date: Mapped[str | None] = mapped_column(DateTime(timezone=False), server_default=func.now(), server_onupdate=sqltext("NOW()"))
