import os
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # PostgreSQL DSN
    DATABASE_URL: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/visual_inspection"
    )

    # Lưu file ảnh (server-side)
    UPLOAD_DIR: str = Field(default="uploads")

    # Tên model dinov2
    MODEL_NAME: str = Field(default="vit_small_patch14_dinov2.lvd142m")

    # Thiết bị: "cuda" | "cpu" | "mps"
    DEVICE: str = Field(default="cuda")

    # Kích thước embedding của ViT-S/14 (DINOv2): 384
    EMBEDDING_DIM: int = Field(default=384)

    # Bảng & index config
    TABLE_NAME: str = Field(default="qc_productphotolibrary")
    ENABLE_IVFFLAT_INDEX: bool = Field(default=True)
    IVFFLAT_LISTS: int = Field(default=100)
    IVFFLAT_DISTANCE: str = Field(default="cosine")

    # FastAPI Config (Sakura style)
    PROJECT_PATH: str = Field(default="")
    IP: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8031)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @classmethod
    def load_settings(cls) -> "Settings":
        yaml_path = os.path.join(os.path.dirname(__file__), "main_server_config.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                # Map keys cleanly
                mapped = {}
                for field_name in cls.model_fields.keys():
                    if field_name in raw:
                        mapped[field_name] = raw[field_name]
                # Return Settings constructed from yaml keys
                return cls(**mapped)
            except Exception as e:
                print(f"[WARN] Failed to load main_server_config.yaml: {e}")
        return cls()

settings = Settings.load_settings()
