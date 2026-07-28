from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from .config import EvaluatorConfig

class BaseEvaluator:
    """A base class for creating evaluator

    This class provides the foundation for validation processes, including model evaluation, metric computation, and
    result visualization.
    """
    def __init__(self,
                 cfg: EvaluatorConfig,
                 *args,
                 **kwargs):
        pass