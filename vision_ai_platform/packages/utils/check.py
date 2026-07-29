# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import functools
import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from . import (
    ARM64,
    ASSETS_URL,
    IS_COLAB,
    IS_DOCKER,
    IS_KAGGLE,
    IS_JETSON,
    LINUX,
    LOGGER,
    MACOS,
    ONLINE,
    PYTHON_VERSION,
    RKNN_CHIPS,
    ROOT,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    USER_CONFIG_DIR,
    WINDOWS,
    Retry,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    downloads,
    env_bool,
)

REMOTE_FILE_PREFIXES = ("https://", "http://", "rtsp://", "rtmp://", "tcp://", "ul://", "gs://")


def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):
    """Parse a requirements.txt file, ignoring lines that start with '#' and any text after '#'.

    Args:
        file_path (Path): Path to the requirements.txt file.
        package (str, optional): Python package to use instead of requirements.txt file.

    Returns:
        requirements (list[SimpleNamespace]): List of parsed requirements as SimpleNamespace objects with `name` and
            `specifier` attributes.

    Examples:
        >>> from ultralytics.utils.checks import parse_requirements
        >>> parse_requirements(package="ultralytics")
    """
    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.partition("#")[0].strip()  # ignore inline comments
            if match := re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line):
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))

    return requirements


def get_distribution_name(import_name: str) -> str:
    """Get the pip distribution name for a given import name (e.g., 'cv2' -> 'opencv-python-headless')."""
    for dist in metadata.distributions():
        top_level = (dist.read_text("top_level.txt") or "").split()
        if import_name in top_level:
            return dist.metadata["Name"]
    return import_name


@functools.lru_cache
def parse_version(version="0.0.0") -> tuple:
    """Convert a version string to a tuple of integers from its release segments, ignoring prefixes and suffixes.

    Not PEP 440: pre-release/dev/post/local suffixes are dropped, so '1.0rc1', '1.0.post1', and '1.0+cu118' all compare
    equal to '1.0'. Use the `packaging` library where exact pre-release ordering matters.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu', '4.13.0.92', or 'v2.1'

    Returns:
        (tuple): Tuple of integers representing the release segments, at least 3 long, i.e. (2, 0, 1)
    """
    try:
        nums = [int(x) for x in re.search(r"\d+(?:\.\d+)*", version).group(0).split(".")]
        return tuple(nums + [0] * (3 - len(nums)))  # keep all release segments, ignore 'v' prefix and '+cu118'/'rc1'
    except Exception as e:
        LOGGER.warning(f"failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0


def is_ascii(s) -> bool:
    """Check if a string is composed of only ASCII characters.

    Args:
        s (str | list | tuple | dict): Input to be checked (all are converted to string for checking).

    Returns:
        (bool): True if the string is composed only of ASCII characters, False otherwise.
    """
    return all(ord(c) < 128 for c in str(s))


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int | list[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        max_dim (int): Maximum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (list[int] | int): Updated image size.
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):  # i.e. '640' or '[640,640]'
        try:
            imgsz = [int(imgsz)] if imgsz.isnumeric() else ast.literal_eval(imgsz)
        except (ValueError, SyntaxError):
            raise ValueError(
                f"'imgsz={imgsz}' is not a valid image size. "
                f"Valid imgsz values are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
            ) from None
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )

    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        LOGGER.warning(f"imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


@functools.lru_cache
def check_uv():
    """Check if uv package manager is installed and can run successfully."""
    try:
        return subprocess.run(["uv", "-V"], capture_output=True, check=False).returncode == 0
    except FileNotFoundError:
        return False


@functools.lru_cache
def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str): Name to be used in warning message.
        hard (bool): If True, raise a ModuleNotFoundError if the requirement is not met.
        verbose (bool): If True, print warning message if requirement is not met.
        msg (str): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Examples:
        Check if current version is exactly 22.04
        >>> check_version(current="22.04", required="==22.04")

        Check if current version is greater than or equal to 22.04
        >>> check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        Check if current version is less than or equal to 22.04
        >>> check_version(current="22.04", required="<=22.04")

        Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        >>> check_version(current="21.10", required=">20.04,<22.04")
    """
    if not current:  # if current is '' or None
        LOGGER.warning(f"invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if re.fullmatch(
                r"v\d+(\.\d+)*([-_.]?(a|b|c|rc|alpha|beta|pre|preview)[-_.]?\d*)?"
                r"([-_.]?(post|rev|r)[-_.]?\d*)?([-_.]?dev[-_.]?\d*)?(\+[\w.-]+)?",
                current,
                re.IGNORECASE,
            ):
                pass
            elif hard:
                raise ModuleNotFoundError(f"{current} package is required but not installed") from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    if "sys_platform" in required and (  # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        if not op:
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        n = max(len(c), len(v))  # pad to equal length so 4-segment pins like '!=4.13.0.90' compare exactly
        cn, vn = c + (0,) * (n - len(c)), v + (0,) * (n - len(v))
        if (
            (op == "==" and cn != vn)
            or (op == "!=" and cn == vn)
            or (op == ">=" and not (cn >= vn))
            or (op == "<=" and not (cn <= vn))
            or (op == ">" and not (cn > vn))
            or (op == "<" and not (cn < vn))
        ):
            result = False
    if not result:
        warning = f"{name}{required} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(warning)  # assert version requirements met
        if verbose:
            LOGGER.warning(warning)
    return result


def check_latest_pypi_version(package_name="ultralytics"):
    """Return the latest version of a PyPI package without downloading or installing it.

    Args:
        package_name (str): The name of the package to find the latest version for.

    Returns:
        (str | None): The latest version of the package, or None if unavailable.
    """
    import requests  # scoped as slow import

    try:
        requests.packages.urllib3.disable_warnings()  # Disable the InsecureRequestWarning
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=3)
        if response.status_code == 200:
            return response.json()["info"]["version"]
    except Exception:
        return None


@ThreadingLocked()
@functools.lru_cache
def check_font(font="Arial.ttf"):
    """Find font locally or download to user's configuration directory if it does not already exist.

    Args:
        font (str): Path or name of font.

    Returns:
        (Path | str): Resolved font file path.
    """
    from matplotlib import font_manager  # scope for faster 'import ultralytics'

    # Check USER_CONFIG_DIR
    name = Path(font).name
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    # Check system fonts
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]

    # Download to USER_CONFIG_DIR if missing
    url = f"{ASSETS_URL}/{name}"
    if downloads.is_url(url, check=True):
        downloads.safe_download(url=url, file=file)
        return file


def check_python(minimum: str = "3.8.0", hard: bool = True, verbose: bool = False) -> bool:
    """Check current python version against the required minimum version.

    Args:
        minimum (str): Required minimum version of python.
        hard (bool): If True, raise a ModuleNotFoundError if the requirement is not met.
        verbose (bool): If True, print warning message if requirement is not met.

    Returns:
        (bool): Whether the installed Python version meets the minimum constraints.
    """
    return check_version(PYTHON_VERSION, minimum, name="Python", hard=hard, verbose=verbose)


@TryExcept()
def check_apt_requirements(requirements):
    """Check if apt packages are installed and install missing ones.

    Args:
        requirements (list[str]): List of apt package names to check and install.
    """
    prefix = colorstr("red", "bold", "apt requirements:")
    # Check which packages are missing
    missing_packages = []
    for package in requirements:
        try:
            # Use dpkg -l to check if package is installed
            result = subprocess.run(["dpkg", "-l", package], capture_output=True, text=True, check=False)
            # Check if package is installed (look for "ii" status)
            if result.returncode != 0 or not any(
                line.startswith("ii") and package in line for line in result.stdout.splitlines()
            ):
                missing_packages.append(package)
        except Exception:
            # If check fails, assume package is not installed
            missing_packages.append(package)

    # Install missing packages if any
    if missing_packages:
        LOGGER.info(
            f"{prefix} Ultralytics requirement{'s' * (len(missing_packages) > 1)} {missing_packages} not found, attempting AutoUpdate..."
        )
        # Optionally update package list first
        cmd = (["sudo"] if is_sudo_available() else []) + ["apt", "update"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Build and run the install command
        cmd = (["sudo"] if is_sudo_available() else []) + ["apt", "install", "-y"] + missing_packages
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        LOGGER.info(f"{prefix} AutoUpdate success ✅")
        LOGGER.warning(f"{prefix} {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n")


def check_torchvision():
    """Check the installed versions of PyTorch and Torchvision to ensure they're compatible.

    This function checks the installed versions of PyTorch and Torchvision, and warns if they're incompatible according
    to the compatibility table based on: https://github.com/pytorch/vision#installation.
    """
    compatibility_table = {
        "2.10": ["0.25"],
        "2.9": ["0.24"],
        "2.8": ["0.23"],
        "2.7": ["0.22"],
        "2.6": ["0.21"],
        "2.5": ["0.20"],
        "2.4": ["0.19"],
        "2.3": ["0.18"],
        "2.2": ["0.17"],
        "2.1": ["0.16"],
        "2.0": ["0.15"],
        "1.13": ["0.14"],
        "1.12": ["0.13"],
    }

    # Check major and minor versions
    v_torch = ".".join(TORCH_VERSION.split("+", 1)[0].split(".")[:2])
    if v_torch in compatibility_table:
        compatible_versions = compatibility_table[v_torch]
        v_torchvision = ".".join(TORCHVISION_VERSION.split("+", 1)[0].split(".")[:2])
        if all(v_torchvision != v for v in compatible_versions):
            LOGGER.warning(
                f"torchvision=={v_torchvision} is incompatible with torch=={v_torch}.\n"
                f"Run 'pip install torchvision=={compatible_versions[0]}' to fix torchvision or "
                "'pip install -U torch torchvision' to update both.\n"
                "For a full compatibility table see https://github.com/pytorch/vision#installation"
            )


def check_suffix(file="yolo26n.pt", suffix=".pt", msg=""):
    """Check file(s) for acceptable suffix.

    Args:
        file (str | list[str]): File or list of files to check.
        suffix (str | tuple): Acceptable suffix or tuple of suffixes.
        msg (str): Additional message to display in case of error.
    """
    if file and suffix:
        if isinstance(suffix, str):
            suffix = {suffix}
        for f in file if isinstance(file, (list, tuple)) else [file]:
            if s := clean_url(f).rpartition(".")[-1].lower().strip():  # file suffix
                assert f".{s}" in suffix, f"{msg}{f} acceptable suffix is {suffix}, not .{s}"


def check_yolov5u_filename(file: str, verbose: bool = True) -> str:
    """Replace legacy YOLOv5 filenames with updated YOLOv5u filenames.

    Args:
        file (str): Filename to check and potentially update.
        verbose (bool): Whether to print information about the replacement.

    Returns:
        (str): Updated filename.
    """
    if "yolov3" in file or "yolov5" in file:
        if "u.yaml" in file:
            file = file.replace("u.yaml", ".yaml")  # i.e. yolov5nu.yaml -> yolov5n.yaml
        elif ".pt" in file and "u" not in file:
            original_file = file
            file = re.sub(r"(.*yolov5([nsmlx]))\.pt", "\\1u.pt", file)  # i.e. yolov5n.pt -> yolov5nu.pt
            file = re.sub(r"(.*yolov5([nsmlx])6)\.pt", "\\1u.pt", file)  # i.e. yolov5n6.pt -> yolov5n6u.pt
            file = re.sub(r"(.*yolov3(|-tiny|-spp))\.pt", "\\1u.pt", file)  # i.e. yolov3-spp.pt -> yolov3-sppu.pt
            if file != original_file and verbose:
                LOGGER.info(
                    f"PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.\nYOLOv5 'u' models are "
                    f"trained with https://github.com/ultralytics/ultralytics and feature improved performance vs "
                    f"standard YOLOv5 models trained with https://github.com/ultralytics/yolov5."
                )
    return file


def check_model_file_from_stem(model: str = "yolo11n") -> str | Path:
    """Return a model filename from a valid model stem.

    Args:
        model (str): Model stem to check.

    Returns:
        (str | Path): Model filename with appropriate suffix.
    """
    path = Path(model)
    if not path.suffix and path.stem in downloads.GITHUB_ASSETS_STEMS:
        return path.with_suffix(".pt")  # add suffix, i.e. yolo26n -> yolo26n.pt
    return model


def check_is_path_safe(basedir: Path | str, path: Path | str) -> bool:
    """Check if the resolved path is under the intended directory to prevent path traversal.

    Args:
        basedir (Path | str): The intended directory.
        path (Path | str): The path to check.

    Returns:
        (bool): True if the path is safe, False otherwise.
    """
    base_dir_resolved = Path(basedir).resolve()
    path_resolved = Path(path).resolve()

    return path_resolved.exists() and path_resolved.parts[: len(base_dir_resolved.parts)] == base_dir_resolved.parts


@functools.lru_cache
def check_imshow(warn=False):
    """Check if environment supports image displays.

    Args:
        warn (bool): Whether to warn if environment doesn't support image displays.

    Returns:
        (bool): True if environment supports image displays, False otherwise.
    """
    try:
        if LINUX:
            assert not IS_COLAB and not IS_KAGGLE
            assert "DISPLAY" in os.environ, "The DISPLAY environment variable isn't set."
        cv2.imshow("test", np.zeros((8, 8, 3), dtype=np.uint8))  # show a small 8-pixel image
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False


def print_args(args: dict | None = None, show_file=True, show_func=False):
    """Print function arguments (optional args dict).

    Args:
        args (dict, optional): Arguments to print.
        show_file (bool): Whether to show the file name.
        show_func (bool): Whether to show the function name.
    """

    def strip_auth(v):
        """Clean longer Ultralytics HUB URLs by stripping potential authentication information."""
        return clean_url(v) if (isinstance(v, str) and v.startswith("http") and len(v) > 100) else v

    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={strip_auth(v)}" for k, v in sorted(args.items())))


def cuda_device_count() -> int:
    """Get the number of NVIDIA GPUs available in the environment.

    Returns:
        (int): The number of NVIDIA GPUs available.
    """
    if IS_JETSON:
        # NVIDIA Jetson does not fully support nvidia-smi and therefore use PyTorch instead
        return torch.cuda.device_count()
    else:
        try:
            # Run the nvidia-smi command and capture its output
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader,nounits"], encoding="utf-8"
            )

            # Take the first line and strip any leading/trailing white space
            first_line = output.strip().split("\n", 1)[0]

            return int(first_line)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            # If the command fails, nvidia-smi is not found, or output is not an integer, assume no GPUs are available
            return 0


def cuda_is_available() -> bool:
    """Check if CUDA is available in the environment.

    Returns:
        (bool): True if one or more NVIDIA GPUs are available, False otherwise.
    """
    return cuda_device_count() > 0


def is_rockchip():
    """Check if the current environment is running on a Rockchip SoC.

    Returns:
        (bool): True if running on a Rockchip SoC, False otherwise.
    """
    if LINUX and ARM64:
        try:
            with open("/proc/device-tree/compatible") as f:
                dev_str = f.read()
                *_, soc = dev_str.split(",")
                if soc.replace("\x00", "").split("-", 1)[0] in RKNN_CHIPS:
                    return True
        except OSError:
            return False
    else:
        return False


def is_sudo_available() -> bool:
    """Check if the sudo command is available in the environment.

    Returns:
        (bool): True if the sudo command is available, False otherwise.
    """
    if WINDOWS:
        return False
    cmd = "sudo --version"
    return (
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode
        == 0
    )


# Run checks and define constants
check_python("3.8", hard=False, verbose=True)  # check python version
check_torchvision()  # check torch-torchvision compatibility

# Define constants
IS_PYTHON_3_8 = PYTHON_VERSION.startswith("3.8")
IS_PYTHON_3_9 = PYTHON_VERSION.startswith("3.9")
IS_PYTHON_3_10 = PYTHON_VERSION.startswith("3.10")
IS_PYTHON_3_12 = PYTHON_VERSION.startswith("3.12")
IS_PYTHON_3_13 = PYTHON_VERSION.startswith("3.13")

IS_PYTHON_MINIMUM_3_9 = check_python("3.9", hard=False)
IS_PYTHON_MINIMUM_3_10 = check_python("3.10", hard=False)
IS_PYTHON_MINIMUM_3_12 = check_python("3.12", hard=False)
IS_PYTHON_MINIMUM_3_13 = check_python("3.13", hard=False)
