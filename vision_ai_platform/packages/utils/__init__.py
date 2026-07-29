from __future__ import annotations

import contextlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import re
import socket
import sys
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from urllib.parse import unquote

import cv2
import numpy as np
import torch


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable, accepting common truthy strings.

    Accepts "1", "true", "yes", "on", "y", "t" (case-insensitive, whitespace-trimmed) as True; any other set value is
    False. The default is returned only when the variable is unset, not when it is set to an empty string.

    Args:
        name (str): Environment variable name.
        default (bool): Value returned when the variable is unset.

    Returns:
        (bool): Parsed boolean value.

    Examples:
        >>> env_bool("YOLO_UNSET_EXAMPLE_VAR", True)  # returns the default when the variable is unset
        True
    """
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", "-1"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "-1"))  # https://pytorch.org/docs/stable/elastic/run.html

# Other Constants
ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
ASSETS = ROOT / "assets"  # default images
ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"  # assets GitHub URL
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLO multiprocessing threads
VERBOSE = env_bool("GLOB_VERBOSE", True)  # global verbose mode
SAFE_LOAD = env_bool("MODEL_SAFE_LOAD")  # opt-in weights_only model loading
LOGGING_NAME = "vision_ai_platform"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
MACOS_VERSION = platform.mac_ver()[0] if MACOS else None
NOT_MACOS14 = not (MACOS and MACOS_VERSION.startswith("14."))
ARM64 = platform.machine() in {"arm64", "aarch64"}  # ARM64 booleans
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = str(torch.__version__)  # Normalize torch.__version__ (PyTorch>1.9 returns TorchVersion objects)
TORCHVISION_VERSION = importlib.metadata.version("torchvision")  # faster than importing torchvision
IS_VSCODE = os.environ.get("TERM_PROGRAM") == "vscode"
RKNN_CHIPS = frozenset(
    {
        "rk3588",
        "rk3576",
        "rk3566",
        "rk3568",
        "rk3562",
        "rv1103",
        "rv1106",
        "rv1103b",
        "rv1106b",
        "rk2118",
        "rv1126b",
    }
)  # Rockchip processors available for export
QNN_HTP_ARCHS = frozenset(
    {
        "68",  # Snapdragon 888
        "69",  # Snapdragon 8 Gen 1
        "73",  # Snapdragon 8 Gen 2 / X Elite
        "75",  # Snapdragon 8 Gen 3
        "79",  # Snapdragon 8 Elite
        "81",  # Snapdragon 8 Elite Gen 5
    }
)  # Qualcomm Hexagon HTP architecture versions available for QNN export

# Settings and Environment Variables
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress verbose TF compiler warnings in Colab
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # suppress "NNPACK.cpp could not initialize NNPACK" warnings
os.environ["KINETO_LOG_LEVEL"] = "5"  # suppress verbose PyTorch profiler output when computing FLOPs

# Centralized warning suppression
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")  # PyTorch deprecation
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")  # matplotlib>=3.7.2
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")  # mobileclip timm.layers deprecation
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # ONNX/TorchScript export tracer warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*prim::Constant.*")  # ONNX shape warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="coremltools")  # CoreML np.bool deprecation
logging.getLogger("coremltools").setLevel(logging.ERROR)  # Suppress native binary load failures on non-macOS

# Precompiled type tuples for faster isinstance() checks
FLOAT_OR_INT = (float, int)
STR_OR_PATH = (str, Path)


def threaded(func):
    """Multi-thread a target function by default and return the thread or function result.

    This decorator provides flexible execution of the target function, either in a separate thread or synchronously. By
    default, the function runs in a thread, but this can be controlled via the 'threaded=False' keyword argument which
    is removed from kwargs before calling the function.

    Args:
        func (callable): The function to be potentially executed in a separate thread.

    Returns:
        (callable): A wrapper function that either returns a thread or the direct function result. The thread is
            non-daemon so interpreter shutdown joins it before module teardown, avoiding teardown races.

    Examples:
        >>> @threaded
        ... def process_data(data):
        ...     return data
        >>>
        >>> thread = process_data(my_data)  # Runs in background thread
        >>> result = process_data(my_data, threaded=False)  # Runs synchronously, returns function result
    """

    def wrapper(*args, **kwargs):
        """Multi-thread a given function based on 'threaded' kwarg and return the thread or function result."""
        if kwargs.pop("threaded", True):  # run in thread
            # Non-daemon so interpreter shutdown joins the thread before module teardown; a daemon thread still
            # running at exit can be killed mid-C-call, aborting the process (e.g. 'FATAL: exception not rethrown').
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=False)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper


def read_device_model() -> str:
    """Read the device model information from the system.

    Returns:
        (str): Platform release string in lowercase, used to identify device models like Jetson or Raspberry Pi.
    """
    return platform.release().lower()


def is_ubuntu() -> bool:
    """Check if the OS is Ubuntu.

    Returns:
        (bool): True if OS is Ubuntu, False otherwise.
    """
    try:
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    except FileNotFoundError:
        return False


def is_debian(codenames: list[str] | str | None = None) -> list[bool] | bool:
    """Check if the OS is Debian.

    Args:
        codenames (list[str] | None | str): Specific Debian codename to check for (e.g., 'buster', 'bullseye'). If None,
            only checks for Debian.

    Returns:
        (list[bool] | bool): List of booleans indicating if OS matches each Debian codename, or a single boolean if no
            codenames provided.
    """
    try:
        with open("/etc/os-release") as f:
            content = f.read()
            if codenames is None:
                return "ID=debian" in content
            if isinstance(codenames, str):
                codenames = [codenames]
            return [
                f"VERSION_CODENAME={codename}" in content if codename else "ID=debian" in content
                for codename in codenames
            ]
    except FileNotFoundError:
        return [False] * len(codenames) if codenames else False


def is_colab():
    """Check if the current script is running inside a Google Colab notebook.

    Returns:
        (bool): True if running inside a Colab notebook, False otherwise.
    """
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ


def is_kaggle():
    """Check if the current script is running inside a Kaggle kernel.

    Returns:
        (bool): True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_jupyter():
    """Check if the current script is running inside a Jupyter Notebook.

    Returns:
        (bool): True if running inside a Jupyter Notebook, False otherwise.

    Notes:
        - Only works on Colab and Kaggle, other environments like Jupyterlab and Paperspace are not reliably detectable.
        - "get_ipython" in globals() method suffers false positives when IPython package installed manually.
    """
    return IS_COLAB or IS_KAGGLE


def is_runpod():
    """Check if the current script is running inside a RunPod container.

    Returns:
        (bool): True if running in RunPod, False otherwise.
    """
    return "RUNPOD_POD_ID" in os.environ


def is_docker() -> bool:
    """Determine if the script is running inside a Docker container.

    Returns:
        (bool): True if the script is running inside a Docker container, False otherwise.
    """
    try:
        return os.path.exists("/.dockerenv")
    except Exception:
        return False


@lru_cache(maxsize=3)
def is_jetson(jetpack=None) -> bool:
    """Determine if the Python environment is running on an NVIDIA Jetson device.

    Args:
        jetpack (int | None): If specified, check for specific JetPack version (4, 5, 6).

    Returns:
        (bool): True if running on an NVIDIA Jetson device, False otherwise.
    """
    jetson = "tegra" in DEVICE_MODEL
    if jetson and jetpack:
        try:
            content = Path("/etc/nv_tegra_release").read_text()
            version_map = {4: "R32", 5: "R35", 6: "R36", 7: "R38"}  # JetPack to L4T major version mapping
            return jetpack in version_map and version_map[jetpack] in content
        except Exception:
            return False
    return jetson


DEVICE_MODEL = read_device_model()
IS_COLAB = is_colab()
IS_KAGGLE = is_kaggle()
IS_DOCKER = is_docker()
IS_JUPYTER = is_jupyter()
IS_JETSON = is_jetson()