from sqlalchemy import create_engine, inspect, text
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
    """
    11082026 - KIET - Khai báo base class dùng chung cho SQLAlchemy models.
    """

    pass


def migrate_od_columns() -> None:
    """
    11082026 - KIET - Bổ sung các column OD vào database SQLite cũ nếu còn thiếu.
    """

    required_columns = {
        "inspection_records": {
            "task_type": (
                "ALTER TABLE inspection_records "
                "ADD COLUMN task_type VARCHAR(30) DEFAULT 'inspection'"
            ),
            "camera_id": (
                "ALTER TABLE inspection_records "
                "ADD COLUMN camera_id VARCHAR(100)"
            ),
            "total_objects": (
                "ALTER TABLE inspection_records "
                "ADD COLUMN total_objects INTEGER DEFAULT 0"
            ),
        },
        "bolt_objects": {
            "class_id": (
                "ALTER TABLE bolt_objects "
                "ADD COLUMN class_id INTEGER"
            ),
            "class_name": (
                "ALTER TABLE bolt_objects "
                "ADD COLUMN class_name VARCHAR(100)"
            ),
        },
        "camera_configs": {
            # 23082026-KIET-Bổ sung trạng thái dirty cho database Edge đã tồn tại
            "sync_dirty": (
                "ALTER TABLE camera_configs "
                "ADD COLUMN sync_dirty BOOLEAN DEFAULT 1"
            ),
            # 23082026-KIET-Bổ sung revision Camera Hub cho database Edge đã tồn tại
            "cloud_revision": (
                "ALTER TABLE camera_configs "
                "ADD COLUMN cloud_revision INTEGER DEFAULT 0"
            ),
            # 23082026-KIET-Bổ sung thời điểm sync camera gần nhất cho database Edge đã tồn tại
            "last_synced_at": (
                "ALTER TABLE camera_configs "
                "ADD COLUMN last_synced_at DATETIME"
            ),
        },
    }

    with engine.begin() as connection:
        for table_name, column_definitions in required_columns.items():
            existing_columns = {
                column["name"]
                for column in inspect(connection).get_columns(table_name)
            }

            for column_name, alter_statement in column_definitions.items():
                if column_name in existing_columns:
                    continue

                connection.execute(text(alter_statement))


def init_db() -> None:
    """
    11082026 - KIET - Khởi tạo tables và migrate schema OD trên database cũ.
    """

    from . import models

    Base.metadata.create_all(bind=engine)
    migrate_od_columns()
