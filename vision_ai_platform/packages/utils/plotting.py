from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

def _gaussian_filter1d(y, sigma: int = 3, truncate: float = 4.0) -> np.ndarray:
    """Smooth a 1D array with a Gaussian kernel (NumPy replacement for scipy.ndimage.gaussian_filter1d).

    Args:
        y (np.ndarray): Input 1D array to smooth.
        sigma (int): Standard deviation of the Gaussian kernel.
        truncate (float): Truncate the kernel at this many standard deviations.

    Returns:
        (np.ndarray): Smoothed 1D array with the same length as the input.
    """
    y = np.asarray(y, dtype=float)
    radius = int(truncate * sigma + 0.5)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    # scipy 'reflect' boundary mode is equivalent to NumPy 'symmetric'
    return np.convolve(np.pad(y, radius, mode="symmetric"), kernel, mode="valid")

