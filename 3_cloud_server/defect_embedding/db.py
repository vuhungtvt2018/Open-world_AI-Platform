from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config import settings

engine = create_engine(settings.DATABASE_URL, future=True, echo=False)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class Base(DeclarativeBase):
    pass

def ensure_pgvector():
    with engine.begin() as conn:
        # Bật extension pgvector
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # Chọn khoảng cách cho ivfflat
        dist = settings.IVFFLAT_DISTANCE.lower()
        if dist not in ("l2", "cosine", "ip"):
            dist = "l2"

        if settings.ENABLE_IVFFLAT_INDEX:
            # Tạo index nếu chưa tồn tại (đặt sau khi tạo bảng)
            # Ta chỉ chuẩn bị lệnh, sẽ chạy sau create_all trong main.py
            pass
