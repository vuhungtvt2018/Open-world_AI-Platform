from sqlalchemy import Integer, String, Float, Boolean, DateTime, ForeignKey, Text, SmallInteger
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func, text as sqltext
from typing import List
from datetime import datetime
import json
from .session import Base

class Product(Base):
    __tablename__ = "products"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    item_code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    product_type_id: Mapped[int] = mapped_column(ForeignKey("product_types.id"), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    product_type: Mapped["ProductType"] = relationship("ProductType", back_populates="products")    

class ProductType(Base):
    __tablename__ = "product_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    products: Mapped[List["Product"]] = relationship("Product", back_populates="product_type")

class InspectionRecord(Base):
    """Bảng đại diện cho một lần chụp ảnh toàn cảnh (Capture)"""
    __tablename__ = "inspection_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(String(50), index=True)
    original_image: Mapped[str] = mapped_column(String(255))
    ng_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Quan hệ 1-N với InspectionObject
    objects: Mapped[List["InspectionObject"]] = relationship(
        "InspectionObject", back_populates="record", cascade="all, delete-orphan"
    )

class InspectionObject(Base):
    """Một đối tượng được phát hiện trong ảnh inspection."""
    __tablename__ = "inspection_objects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[int] = mapped_column(ForeignKey("inspection_records.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    object_index: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[str] = mapped_column(String(255))  # Chứa chuỗi JSON [x1, y1, x2, y2]
    score: Mapped[float] = mapped_column(Float, default=0.0)
    is_ng: Mapped[bool] = mapped_column(Boolean, default=False)
    overlap_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    crop_image: Mapped[str] = mapped_column(String(255))
    
    record: Mapped["InspectionRecord"] = relationship("InspectionRecord", back_populates="objects")
    
    product: Mapped["Product"] = relationship("Product")
    
    # Quan hệ 1-N với AnomalyDetail
    anomalies: Mapped[List["AnomalyDetail"]] = relationship(
        "AnomalyDetail", back_populates="inspection_object", cascade="all, delete-orphan"
    )

class AnomalyDetail(Base):
    """Bảng lưu chi tiết từng vết lỗi trên con bu-lông"""
    __tablename__ = "anomaly_details"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    object_id: Mapped[int] = mapped_column(ForeignKey("inspection_objects.id"), index=True)
    bbox_full: Mapped[str] = mapped_column(String(255))           # [x1, y1, x2, y2] toàn cảnh
    bbox_in_object_crop: Mapped[str] = mapped_column(String(255)) # [cx1, cy1, cx2, cy2] trong crop
    defect_class: Mapped[str | None] = mapped_column(String(100), nullable=True) # Nhãn phân loại lỗi
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)       # Độ tin cậy (Cosine similarity)
    
    inspection_object: Mapped["InspectionObject"] = relationship("InspectionObject", back_populates="anomalies")

class SyncState(Base):
    """Bảng quản lý trạng thái đồng bộ dữ liệu (Edge -> Cloud)"""
    __tablename__ = "sync_states"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(String(255))
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

class AIModel(Base):
    """Bảng quản lý Model (Model Registry)"""
    __tablename__ = "ai_models"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    type: Mapped[str] = mapped_column(String(50))     # Detection, Anomaly, Alignment, Classification
    format: Mapped[str] = mapped_column(String(20))   # PT, ONNX, IR...
    version: Mapped[str] = mapped_column(String(20))
    map_acc: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20))   # PRODUCTION, STAGING, INACTIVE
    file_path: Mapped[str] = mapped_column(String(255))
    speed_ms: Mapped[str | None] = mapped_column(String(20), nullable=True)

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
    Insert_Date: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), server_default=func.now())
    Update_PIC: Mapped[str | None] = mapped_column(String(50))
    Update_Date: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), server_default=func.now(), server_onupdate=sqltext("NOW()"))
