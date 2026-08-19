from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class PhotoBase(BaseModel):
    SM_ID: Optional[str] = None
    ItemCode: Optional[str] = None
    ImageType: Optional[int] = None
    ErrorDetail: Optional[str] = None
    Insert_PIC: Optional[str] = None
    Update_PIC: Optional[str] = None

class PhotoCreate(PhotoBase):
    pass

class PhotoRead(BaseModel):
    ID: int
    SM_ID: Optional[str] = None
    ItemCode: Optional[str] = None
    ImageName: Optional[str] = None
    ImageType: Optional[int] = None
    ErrorDetail: Optional[str] = None
    Insert_PIC: Optional[str] = None
    Insert_Date: Optional[datetime] = None
    Update_PIC: Optional[str] = None
    Update_Date: Optional[datetime] = None
    model_config = {"from_attributes": True}


class PhotoUpdateErrorDetail(BaseModel):
    ErrorDetail: str

class SearchRequestByImage(BaseModel):
    top_k: int = Field(default=10, ge=1, le=200)
    metric: str = Field(default="cosine")
    ItemCode: Optional[str] = None

class SearchResultItem(BaseModel):
    ID: int
    distance: float
    SM_ID: Optional[str] = None
    ItemCode: Optional[str] = None
    ImageName: Optional[str] = None
    ErrorDetail: Optional[str] = None
