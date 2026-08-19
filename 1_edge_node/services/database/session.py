from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import os

# Đường dẫn DB cục bộ (SQLite) đặt tại thư mục captures để không mất khi update code
DB_DIR = "captures"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "visual_inspection_edge.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

# Kết nối CSDL SQLite (cho phép multithreading qua check_same_thread=False)
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}, echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def init_db():
    from . import models
    Base.metadata.create_all(bind=engine)
