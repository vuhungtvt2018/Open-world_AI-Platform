__version__ = "1.0.0"

import importlib
import os
from typing import TYPE_CHECKING

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from vision_ai_platform.packages.utils import ASSETS, SETTINGS
from vision_ai_platform.packages.utils.check import check_yolo as checks
from vision_ai_platform.packages.utils.downloads import download

settings = SETTINGS

MODELS = ("YOLO", "YOLOWorld", "YOLOE", "RTDETR")

__all__ = (
    "__version__",
    "ASSETS",
    *MODELS,
    "checks",
    "download",
    "settings",
)

if TYPE_CHECKING:
    # Enable hints for type checkers
    from vision_ai_platform.packages.ai.models import YOLO, YOLOWorld, YOLOE, RTDETR  # noqa


def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("vision_ai_platform.packages.ai.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS))


if __name__ == "__main__":
    print(__version__)
