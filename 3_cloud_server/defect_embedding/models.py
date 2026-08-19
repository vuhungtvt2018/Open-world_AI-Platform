from sqlalchemy import Integer, String, Float, Boolean, Text, DateTime, SmallInteger, func, text as sqltext, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from db import Base
from config import settings
from typing import List

class QCProductPhotoLibrary(Base):
    __tablename__ = settings.TABLE_NAME

    ID: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    SM_ID: Mapped[str | None] = mapped_column(String(100))
    ItemCode: Mapped[str | None] = mapped_column(String(50))
    ImageName: Mapped[str | None] = mapped_column(Text)
    ImageType: Mapped[int | None] = mapped_column(SmallInteger)
    ErrorDetail: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.EMBEDDING_DIM))
    Insert_PIC: Mapped[str | None] = mapped_column(String(50))
    Insert_Date: Mapped[str | None] = mapped_column(DateTime(timezone=False), server_default=func.now())
    Update_PIC: Mapped[str | None] = mapped_column(String(50))
    Update_Date: Mapped[str | None] = mapped_column(DateTime(timezone=False), server_default=func.now(), server_onupdate=sqltext("NOW()"))


class CloudAIModel(Base):
    __tablename__ = "cloud_ai_models"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    type: Mapped[str] = mapped_column(String(50))     
    format: Mapped[str] = mapped_column(String(20))   
    version: Mapped[str] = mapped_column(String(20))
    map_acc: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20))   
    file_path: Mapped[str] = mapped_column(String(255))
    speed_ms: Mapped[str] = mapped_column(String(20), nullable=True)
    updated_at: Mapped[str] = mapped_column(DateTime, server_default=func.now(), server_onupdate=sqltext("NOW()"))


class InspectionRecord(Base):
    __tablename__ = "inspection_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    edge_node_id: Mapped[str] = mapped_column(String(50), index=True) # ID của máy Edge gửi lên
    edge_record_id: Mapped[int] = mapped_column(Integer, index=True) # ID gốc của record dưới Edge
    item_code: Mapped[str] = mapped_column(String(255), nullable=True)
    timestamp: Mapped[str] = mapped_column(String(50), index=True)
    original_image: Mapped[str] = mapped_column(String(500))
    ng_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())
    
    # Quan hệ 1-N với BoltObject
    objects: Mapped[List["BoltObject"]] = relationship(
        "BoltObject", back_populates="record", cascade="all, delete-orphan"
    )

class BoltObject(Base):
    __tablename__ = "bolt_objects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[int] = mapped_column(ForeignKey("inspection_records.id"), index=True)
    edge_bolt_id: Mapped[int] = mapped_column(Integer, nullable=True) # ID gốc của bolt dưới Edge
    object_index: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[str] = mapped_column(String(255))
    score: Mapped[float] = mapped_column(Float, default=0.0)
    is_ng: Mapped[bool] = mapped_column(Boolean, default=False)
    overlap_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    crop_image: Mapped[str] = mapped_column(String(500), nullable=True)
    
    record: Mapped["InspectionRecord"] = relationship("InspectionRecord", back_populates="objects")
    
    # Quan hệ 1-N với AnomalyDetail
    anomalies: Mapped[List["AnomalyDetail"]] = relationship(
        "AnomalyDetail", back_populates="bolt", cascade="all, delete-orphan"
    )

class AnomalyDetail(Base):
    __tablename__ = "anomaly_details"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bolt_id: Mapped[int] = mapped_column(ForeignKey("bolt_objects.id"), index=True)
    edge_anomaly_id: Mapped[int] = mapped_column(Integer, nullable=True) # ID gốc của anomaly dưới Edge
    bbox_full: Mapped[str] = mapped_column(String(255))
    bbox_in_object_crop: Mapped[str] = mapped_column(String(255))
    defect_class: Mapped[str] = mapped_column(String(100), nullable=True)
    similarity: Mapped[float] = mapped_column(Float, nullable=True)
    
    bolt: Mapped["BoltObject"] = relationship("BoltObject", back_populates="anomalies")
