from __future__ import annotations

import contextlib
from contextlib import contextmanager
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

def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string

def set_logging(name="LOGGING_NAME", verbose=True):
    """Set up logging with UTF-8 encoding and configurable verbosity.

    This function configures logging for the Ultralytics library, setting the appropriate logging level and formatter
    based on the verbosity flag and the current process rank. It handles special cases for Windows environments where
    UTF-8 encoding might not be the default.

    Args:
        name (str): Name of the logger.
        verbose (bool): Flag to set logging level to INFO if True, ERROR otherwise.

    Returns:
        (logging.Logger): Configured logger object.

    Examples:
        >>> set_logging(name="ultralytics", verbose=True)
        >>> logger = logging.getLogger("ultralytics")
        >>> logger.info("This is an info message")

    Notes:
        - On Windows, this function attempts to reconfigure stdout to use UTF-8 encoding if possible.
        - If reconfiguration is not possible, it falls back to a custom formatter that handles non-UTF-8 environments.
        - The function sets up a StreamHandler with the appropriate formatter and level.
        - The logger's propagate flag is set to False to prevent duplicate logging in parent loggers.
    """
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings

    class PrefixFormatter(logging.Formatter):
        def format(self, record):
            """Format log records with prefixes based on level."""
            # Apply prefixes based on log level
            if record.levelno == logging.WARNING:
                prefix = "WARNING" if WINDOWS else "WARNING ⚠️"
                record.msg = f"{prefix} {record.msg}"
            elif record.levelno == logging.ERROR:
                prefix = "ERROR" if WINDOWS else "ERROR ❌"
                record.msg = f"{prefix} {record.msg}"

            # Handle emojis in message based on platform
            formatted_message = super().format(record)
            return emojis(formatted_message)

    formatter = PrefixFormatter("%(message)s")

    # Handle Windows UTF-8 encoding issues
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        with contextlib.suppress(Exception):
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)  # define globally (used in train.py, val.py, predict.py, etc.)
logging.getLogger("sentry_sdk").setLevel(logging.CRITICAL + 1)

class TryExcept(contextlib.ContextDecorator):
    """Ultralytics TryExcept class for handling exceptions gracefully.

    This class can be used as a decorator or context manager to catch exceptions and optionally print warning messages.
    It allows code to continue execution even when exceptions occur, which is useful for non-critical operations.

    Attributes:
        msg (str): Optional message to display when an exception occurs.
        verbose (bool): Whether to print the exception message.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        ... def func():
        ...     # Function logic here
        ...     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        ...     # Code block here
        ...     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Execute when entering TryExcept context, initialize instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Define behavior when exiting a 'with' block, print error message if necessary."""
        if self.verbose and value:
            LOGGER.warning(f"{self.msg}{': ' if self.msg else ''}{value}")
        return True


class Retry(contextlib.ContextDecorator):
    """Retry class for function execution with exponential backoff.

    This decorator can be used to retry a function on exceptions, up to a specified number of times with an
    exponentially increasing delay between retries. It's useful for handling transient failures in network operations or
    other unreliable processes.

    Attributes:
        times (int): Maximum number of retry attempts.
        delay (int): Initial delay between retries in seconds.

    Examples:
        Example usage as a decorator:
        >>> @Retry(times=3, delay=2)
        ... def test_func():
        ...     # Replace with function logic that may raise exceptions
        ...     return True
    """

    def __init__(self, times=3, delay=2):
        """Initialize Retry class with specified number of retries and delay."""
        self.times = times
        self.delay = delay
        self._attempts = 0

    def __call__(self, func):
        """Decorator implementation for Retry with exponential backoff."""

        def wrapped_func(*args, **kwargs):
            """Apply retries to the decorated function or method."""
            self._attempts = 0
            while self._attempts < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attempts += 1
                    LOGGER.warning(f"Retry {self._attempts}/{self.times} failed: {e}")
                    if self._attempts >= self.times:
                        raise e
                    time.sleep(self.delay * (2**self._attempts))  # exponential backoff delay

        return wrapped_func

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

class ThreadingLocked:
    """A decorator class for ensuring thread-safe execution of a function or method.

    This class can be used as a decorator to make sure that if the decorated function is called from multiple threads,
    only one thread at a time will be able to execute the function.

    Attributes:
        lock (threading.Lock): A lock object used to manage access to the decorated function.

    Examples:
        >>> from ultralytics.utils import ThreadingLocked
        >>> @ThreadingLocked()
        ... def my_function():
        ...    # Your code here
    """

    def __init__(self):
        """Initialize the decorator class with a threading lock."""
        self.lock = threading.Lock()

    def __call__(self, f):
        """Run thread-safe execution of function or method."""
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            """Apply thread-safety to the decorated function or method."""
            with self.lock:
                return f(*args, **kwargs)

        return decorated

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

def is_dir_writeable(dir_path: str | Path) -> bool:
    """Check if a directory is writable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)

def get_user_config_dir(sub_dir="Ultralytics"):
    """Return a writable config dir, preferring YOLO_CONFIG_DIR and being OS-aware.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    if env_dir := os.getenv("YOLO_CONFIG_DIR"):
        p = Path(env_dir).expanduser() / sub_dir
    elif LINUX:
        p = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / sub_dir
    elif WINDOWS:
        p = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:
        p = Path.home() / "Library" / "Application Support" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    if p.exists():  # already created → trust it
        return p
    if is_dir_writeable(p.parent):  # create if possible
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Fallbacks for Docker, GCP/AWS functions where only /tmp is writable
    for alt in [Path("/tmp") / sub_dir, Path.cwd() / sub_dir]:
        if alt.exists():
            return alt
        if is_dir_writeable(alt.parent):
            alt.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(
                f"user config directory '{p}' is not writable, using '{alt}'. Set YOLO_CONFIG_DIR to override."
            )
            return alt

    # Last fallback → CWD
    p = Path.cwd() / sub_dir
    p.mkdir(parents=True, exist_ok=True)
    return p

DEVICE_MODEL = read_device_model()
IS_COLAB = is_colab()
IS_KAGGLE = is_kaggle()
IS_DOCKER = is_docker()
IS_JUPYTER = is_jupyter()
IS_JETSON = is_jetson()
USER_CONFIG_DIR = get_user_config_dir()  # Ultralytics settings dir
SETTINGS_FILE = USER_CONFIG_DIR / "settings.json"

def colorstr(*input):
    r"""Color a string based on the provided color and style arguments using ANSI escape codes.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str | Path): A sequence of strings where the first n-1 strings are color and style arguments, and the
            last string is the one to be colored.

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        "\033[34m\033[1mhello world\033[0m"

    Notes:
        Supported Colors and Styles:
        - Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        - Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        - Misc: 'end', 'bold', 'underline'

    References:
        https://en.wikipedia.org/wiki/ANSI_escape_code
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]

def clean_url(url):
    """Strip auth from URL, i.e. `https://example.com/path/file.txt?auth` -> `https://example.com/path/file.txt`."""
    url = Path(url).as_posix().replace(":/", "://")  # Pathlib turns :// -> :/, as_posix() for Windows
    return unquote(url).split("?", 1)[0]  # '%2F' to '/', split authentication query strings

def url2file(url):
    """Convert URL to filename, i.e. `https://example.com/path/file.txt?auth` -> `file.txt`."""
    return Path(clean_url(url)).name or "download"

def is_online() -> bool:
    """Fast online check using DNS (v4/v6) resolution (Cloudflare + Google).

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    if env_bool("YOLO_OFFLINE"):
        return False

    for host in ("one.one.one.one", "dns.google"):
        try:
            socket.getaddrinfo(host, 0, socket.AF_UNSPEC, 0, 0, socket.AI_ADDRCONFIG)
            return True
        except OSError:
            continue
    return False