from sqlalchemy import Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from typing import List
import json
from .session import Base

class InspectionRecord(Base):
    """Bảng đại diện cho một lần chụp ảnh toàn cảnh (Capture)"""
    __tablename__ = "inspection_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(50), index=True)
    original_image: Mapped[str] = mapped_column(String(255))
    ng_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())
    
    # Quan hệ 1-N với BoltObject
    objects: Mapped[List["BoltObject"]] = relationship(
        "BoltObject", back_populates="record", cascade="all, delete-orphan"
    )

class BoltObject(Base):
    """Bảng đại diện cho một con bu-lông trong ảnh"""
    __tablename__ = "bolt_objects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[int] = mapped_column(ForeignKey("inspection_records.id"), index=True)
    object_index: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[str] = mapped_column(String(255))  # Chứa chuỗi JSON [x1, y1, x2, y2]
    score: Mapped[float] = mapped_column(Float, default=0.0)
    is_ng: Mapped[bool] = mapped_column(Boolean, default=False)
    overlap_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    crop_image: Mapped[str] = mapped_column(String(255))
    
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
