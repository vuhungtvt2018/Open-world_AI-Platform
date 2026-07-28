from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import TrainerConfig

class BaseTrainer:
    def __init__(
        self,
        cfg: TrainerConfig,
    ):
        pass