from pathlib import Path
import os
import sys


FILE = Path(__file__).resolve()

# app_config.py -> edge_agent -> apps -> 1_edge_node
ROOT = FILE.parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.core.config import AppConfig


cfg = AppConfig.from_yaml(
    os.path.join(str(ROOT), "config.yaml")
)

UPLOAD_DIR = os.path.join(str(ROOT), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SYNC_DEFECT_IMAGE_DIR = os.path.join(
    str(ROOT),
    "sync_defect_image",
)
os.makedirs(SYNC_DEFECT_IMAGE_DIR, exist_ok=True)