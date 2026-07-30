import os
import sys
import platform
import re
from pathlib import Path
import functools
import random
from typing import Any
import math
import time
from copy import deepcopy
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from . import LOGGER, PYTHON_VERSION, TORCH_VERSION, NUM_THREADS, TORCHVISION_VERSION, WINDOWS
from .check import check_version

# Version checks (all default to version>=min_version)
TORCH_1_9 = check_version(TORCH_VERSION, "1.9.0")
TORCH_1_10 = check_version(TORCH_VERSION, "1.10.0")
TORCH_1_11 = check_version(TORCH_VERSION, "1.11.0")
TORCH_1_13 = check_version(TORCH_VERSION, "1.13.0")
TORCH_2_0 = check_version(TORCH_VERSION, "2.0.0")
TORCH_2_1 = check_version(TORCH_VERSION, "2.1.0")
TORCH_2_3 = check_version(TORCH_VERSION, "2.3.0")
TORCH_2_4 = check_version(TORCH_VERSION, "2.4.0")
TORCH_2_8 = check_version(TORCH_VERSION, "2.8.0")
TORCH_2_9 = check_version(TORCH_VERSION, "2.9.0")
TORCH_2_10 = check_version(TORCH_VERSION, "2.10.0")
TORCH_2_12 = check_version(TORCH_VERSION, "2.12.0")
TORCHVISION_0_10 = check_version(TORCHVISION_VERSION, "0.10.0")
TORCHVISION_0_11 = check_version(TORCHVISION_VERSION, "0.11.0")
TORCHVISION_0_13 = check_version(TORCHVISION_VERSION, "0.13.0")
TORCHVISION_0_18 = check_version(TORCHVISION_VERSION, "0.18.0")
if WINDOWS and check_version(TORCH_VERSION, "==2.4.0"):  # reject version 2.4.0 on Windows
    LOGGER.warning(
        "Known issue with torch==2.4.0 on Windows with CPU, recommend upgrading to torch>=2.4.1 to resolve "
        "https://github.com/ultralytics/ultralytics/issues/15049"
    )


class CPUInfo:
    """Provide cross-platform CPU brand and model information.

    Query platform-specific sources to retrieve a human-readable CPU descriptor and normalize it for consistent
    presentation across macOS, Linux, and Windows. If platform-specific probing fails, generic platform identifiers are
    used to ensure a stable string is always returned.

    Methods:
        name: Return the normalized CPU name using platform-specific sources with robust fallbacks.
        _clean: Normalize and prettify common vendor brand strings and frequency patterns.
        __str__: Return the normalized CPU name for string contexts.

    Examples:
        >>> CPUInfo.name()
        'Apple M4 Pro'
        >>> str(CPUInfo())
        'Intel Core i7-9750H 2.60GHz'
    """

    @staticmethod
    def name() -> str:
        """Return a normalized CPU model string from platform-specific sources."""
        try:
            if sys.platform == "darwin":
                # Query macOS sysctl for the CPU brand string
                s = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
                ).stdout.strip()
                if s:
                    return CPUInfo._clean(s)
            elif sys.platform.startswith("linux"):
                # Parse /proc/cpuinfo for the first "model name" entry
                p = Path("/proc/cpuinfo")
                if p.exists():
                    for line in p.read_text(errors="ignore").splitlines():
                        if "model name" in line:
                            return CPUInfo._clean(line.split(":", 1)[1])
            elif sys.platform.startswith("win"):
                try:
                    import winreg as wr

                    with wr.OpenKey(wr.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                        val, _ = wr.QueryValueEx(k, "ProcessorNameString")
                        if val:
                            return CPUInfo._clean(val)
                except Exception:
                    # Fall through to generic platform fallbacks on Windows registry access failure
                    pass
            # Generic platform fallbacks
            s = platform.processor() or getattr(platform.uname(), "processor", "") or platform.machine()
            return CPUInfo._clean(s or "Unknown CPU")
        except Exception:
            # Ensure a string is always returned even on unexpected failures
            s = platform.processor() or platform.machine() or ""
            return CPUInfo._clean(s or "Unknown CPU")

    @staticmethod
    def _clean(s: str) -> str:
        """Normalize and prettify a raw CPU descriptor string."""
        s = re.sub(r"\s+", " ", s.strip())
        s = s.replace("(TM)", "").replace("(tm)", "").replace("(R)", "").replace("(r)", "").strip()
        if m := re.search(r"(Intel.*?i\d[\w-]*) CPU @ ([\d.]+GHz)", s, re.IGNORECASE):
            return f"{m.group(1)} {m.group(2)}"
        if m := re.search(r"(AMD.*?Ryzen.*?[\w-]*) CPU @ ([\d.]+GHz)", s, re.IGNORECASE):
            return f"{m.group(1)} {m.group(2)}"
        return s

    def __str__(self) -> str:
        """Return the normalized CPU name."""
        return self.name()


class GPUInfo:
    """Manages NVIDIA GPU information via pynvml with robust error handling.

    Provides methods to query detailed GPU statistics (utilization, memory, temp, power) and select the most idle GPUs
    based on configurable criteria. It safely handles the absence or initialization failure of the pynvml library by
    logging warnings and disabling related features, preventing application crashes.

    Includes fallback logic using `torch.cuda` for basic device counting if NVML is unavailable during GPU
    selection. Manages NVML initialization and shutdown internally.

    Attributes:
        pynvml (module | None): The `pynvml` module if successfully imported and initialized, otherwise `None`.
        nvml_available (bool): Indicates if `pynvml` is ready for use. True if `nvmlInit()` succeeded, False otherwise.
        gpu_stats (list[dict[str, Any]]): A list of dictionaries, each holding stats for one GPU, populated on
        initialization and by `refresh_stats()`. Keys include: 'index', 'name', 'utilization' (%), 'memory_used' (MiB),
            'memory_total' (MiB), 'memory_free' (MiB), 'temperature' (C), 'power_draw' (W), 'power_limit' (W or 'N/A').
            Empty if NVML is unavailable or queries fail.

    Methods:
        refresh_stats: Refresh the internal gpu_stats list by querying NVML.
        print_status: Print GPU status in a compact table format using current stats.
        select_idle_gpu: Select the most idle GPUs based on utilization and free memory.
        shutdown: Shut down NVML if it was initialized.

    Examples:
        Initialize GPUInfo and print status
        >>> gpu_info = GPUInfo()
        >>> gpu_info.print_status()

        Select idle GPUs with minimum memory requirements
        >>> selected = gpu_info.select_idle_gpu(count=2, min_memory_fraction=0.2)
        >>> print(f"Selected GPU indices: {selected}")
    """

    def __init__(self):
        """Initialize GPUInfo, attempting to import and initialize pynvml."""
        self.pynvml: Any | None = None
        self.nvml_available: bool = False
        self.gpu_stats: list[dict[str, Any]] = []

        try:
            import pynvml  # scoped as slow import

            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.nvml_available = True
            self.refresh_stats()
        except Exception as e:
            LOGGER.warning(f"Failed to initialize pynvml, GPU stats disabled: {e}")

    def __del__(self):
        """Ensure NVML is shut down when the object is garbage collected."""
        self.shutdown()

    def shutdown(self):
        """Shut down NVML if it was initialized."""
        if self.nvml_available and self.pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml_available = False

    def refresh_stats(self):
        """Refresh the internal gpu_stats list by querying NVML."""
        self.gpu_stats = []
        if not self.nvml_available or not self.pynvml:
            return

        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            self.gpu_stats.extend(self._get_device_stats(i) for i in range(device_count))
        except Exception as e:
            LOGGER.warning(f"Error during device query: {e}")
            self.gpu_stats = []

    def _get_device_stats(self, index: int) -> dict[str, Any]:
        """Get stats for a single GPU device."""
        handle = self.pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)

        def safe_get(func, *args, default=-1, divisor=1):
            try:
                val = func(*args)
                return val // divisor if divisor != 1 and isinstance(val, (int, float)) else val
            except Exception:
                return default

        temp_type = getattr(self.pynvml, "NVML_TEMPERATURE_GPU", -1)

        return {
            "index": index,
            "name": self.pynvml.nvmlDeviceGetName(handle),
            "utilization": util.gpu if util else -1,
            "memory_used": memory.used >> 20 if memory else -1,  # Convert bytes to MiB
            "memory_total": memory.total >> 20 if memory else -1,
            "memory_free": memory.free >> 20 if memory else -1,
            "temperature": safe_get(self.pynvml.nvmlDeviceGetTemperature, handle, temp_type),
            "power_draw": safe_get(self.pynvml.nvmlDeviceGetPowerUsage, handle, divisor=1000),  # Convert mW to W
            "power_limit": safe_get(self.pynvml.nvmlDeviceGetEnforcedPowerLimit, handle, divisor=1000),
        }

    def print_status(self):
        """Print GPU status in a compact table format using current stats."""
        self.refresh_stats()
        if not self.gpu_stats:
            LOGGER.warning("No GPU stats available.")
            return

        stats = self.gpu_stats
        name_len = max(len(gpu.get("name", "N/A")) for gpu in stats)
        hdr = f"{'Idx':<3} {'Name':<{name_len}} {'Util':>6} {'Mem (MiB)':>15} {'Temp':>5} {'Pwr (W)':>10}"
        LOGGER.info(f"\n--- GPU Status ---\n{hdr}\n{'-' * len(hdr)}")

        for gpu in stats:
            u = f"{gpu['utilization']:>5}%" if gpu["utilization"] >= 0 else " N/A "
            m = f"{gpu['memory_used']:>6}/{gpu['memory_total']:<6}" if gpu["memory_used"] >= 0 else " N/A / N/A "
            t = f"{gpu['temperature']}C" if gpu["temperature"] >= 0 else " N/A "
            p = f"{gpu['power_draw']:>3}/{gpu['power_limit']:<3}" if gpu["power_draw"] >= 0 else " N/A "

            LOGGER.info(f"{gpu.get('index'):<3d} {gpu.get('name', 'N/A'):<{name_len}} {u:>6} {m:>15} {t:>5} {p:>10}")

        LOGGER.info(f"{'-' * len(hdr)}\n")

    def select_idle_gpu(
        self,
        count: int = 1,
        min_memory_fraction: float = 0,
        min_util_fraction: float = 0,
        indices: list[int] | None = None,
    ) -> list[int]:
        """Select the most idle GPUs based on utilization and free memory.

        Args:
            count (int): The number of idle GPUs to select.
            min_memory_fraction (float): Minimum free memory required as a fraction of total memory.
            min_util_fraction (float): Minimum free utilization rate required from 0.0 - 1.0.
            indices (list[int] | None): Restrict selection to these GPU indices, e.g. the GPUs visible under an external
                CUDA_VISIBLE_DEVICES restriction. None searches all GPUs.

        Returns:
            (list[int]): Indices of the selected GPUs, sorted by idleness (lowest utilization first).

        Notes:
             Returns fewer than 'count' if not enough qualify or exist.
             Returns empty list if NVML stats are unavailable or no GPUs meet the criteria.
        """
        assert min_memory_fraction <= 1.0, f"min_memory_fraction must be <= 1.0, got {min_memory_fraction}"
        assert min_util_fraction <= 1.0, f"min_util_fraction must be <= 1.0, got {min_util_fraction}"
        criteria = (
            f"free memory >= {min_memory_fraction * 100:.1f}% and free utilization >= {min_util_fraction * 100:.1f}%"
        )
        LOGGER.info(f"Searching for {count} idle GPUs with {criteria}...")

        if count <= 0:
            return []

        self.refresh_stats()
        if not self.gpu_stats:
            LOGGER.warning("NVML stats unavailable.")
            return []

        # Filter and sort eligible GPUs
        eligible_gpus = [
            gpu
            for gpu in self.gpu_stats
            if (indices is None or gpu["index"] in indices)
            and gpu.get("memory_free", 0) / gpu.get("memory_total", 1) >= min_memory_fraction
            and (100 - gpu.get("utilization", 100)) >= min_util_fraction * 100
        ]
        # Random tiebreaker prevents race conditions when multiple processes start simultaneously
        # and all GPUs appear equally idle (same utilization and free memory)
        eligible_gpus.sort(key=lambda x: (x.get("utilization", 101), -x.get("memory_free", 0), random.random()))

        # Select top 'count' indices
        selected = [gpu["index"] for gpu in eligible_gpus[:count]]

        if selected:
            if len(selected) < count:
                LOGGER.warning(f"Requested {count} GPUs but only {len(selected)} met the idle criteria.")
            LOGGER.info(f"Selected idle CUDA devices {selected}")
        else:
            LOGGER.warning(f"No GPUs met criteria ({criteria}).")

        return selected


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Ensure all processes in distributed training wait for the local master (rank 0) to complete a task first."""
    initialized = dist.is_available() and dist.is_initialized()
    use_ids = initialized and dist.get_backend() == "nccl"

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[torch.cuda.current_device()]) if use_ids else dist.barrier()
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[torch.cuda.current_device()]) if use_ids else dist.barrier()


def smart_inference_mode():
    """Apply torch.inference_mode() decorator if torch>=1.10.0, else torch.no_grad() decorator."""

    def decorate(fn):
        """Apply appropriate torch decorator for inference mode based on torch version."""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn  # already in inference_mode, act as a pass-through
        else:
            return (torch.inference_mode if TORCH_1_10 else torch.no_grad)()(fn)

    return decorate


def autocast(enabled: bool, device: str = "cuda"):
    """Get the appropriate autocast context manager based on PyTorch version and AMP setting.

    This function returns a context manager for automatic mixed precision (AMP) training that is compatible with both
    older and newer versions of PyTorch. It handles the differences in the autocast API between PyTorch versions.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
        device (str, optional): The device to use for autocast.

    Returns:
        (torch.amp.autocast): The appropriate autocast context manager.

    Examples:
        >>> with autocast(enabled=True):
        ...     # Your mixed precision operations here
        ...     pass

    Notes:
        - For PyTorch versions 1.13 and newer, it uses `torch.amp.autocast`.
        - For older versions, it uses `torch.cuda.amp.autocast`.
    """
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)


@functools.lru_cache
def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    from vision_ai_platform.packages.utils import PERSISTENT_CACHE  # avoid circular import error

    if "cpu_info" not in PERSISTENT_CACHE:
        try:
            PERSISTENT_CACHE["cpu_info"] = CPUInfo.name()
        except Exception:
            pass
    return PERSISTENT_CACHE.get("cpu_info", "unknown")


@functools.lru_cache
def get_gpu_info(index):
    """Return a string with system GPU information, i.e. 'Tesla T4, 15102MiB'."""
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"


def parse_device(device: str | int | list | tuple | torch.device = "") -> str:
    """Parse a device request of any form into a canonical device string.

    Args:
        device (str | int | list | tuple | torch.device, optional): Device request, e.g. 'cuda:0', '0,1', [0, 1], 'cpu',
            'mps', or '-1' to auto-select an idle GPU ('-1,-1' for two).

    Returns:
        (str): Canonical device string, e.g. '', 'cpu', 'mps', '0', or '0,1'.

    Examples:
        >>> parse_device("cuda:0")
        '0'

        >>> parse_device([0, 1])
        '0,1'

    Notes:
        Each '-1' is replaced with an idle GPU index. Requested ids exceeding the torch device count that match
        physical GPU ids visible under an external CUDA_VISIBLE_DEVICES restriction are translated to the
        corresponding torch indices, e.g. '3' -> '0' when CUDA_VISIBLE_DEVICES='3'; in-range ids are always torch
        indices, keeping parsing idempotent. Returned indices are relative to the active restriction, so strings
        persisted under one environment (e.g. resumed checkpoint args) address the same physical GPUs only in that
        environment.
    """
    if isinstance(device, torch.device) and device.type == "cuda" and device.index is None:
        return ""  # indexless torch.device('cuda') means the current CUDA device, i.e. the '' default request
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    if device == "cuda":
        device = "0"
    device = ",".join(str(int(x)) if x.isdigit() else x for x in device.split(",") if x)  # "0,,01" -> "0,1"
    # Visible physical ids normalized like requested ids and truncated to the torch device count, mirroring CUDA's
    # atoi-style parsing and its stop at the first invalid CVD entry
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    visible = [str(int(x)) if x.isdigit() else x for x in cvd.split(",") if x][: torch.cuda.device_count()]
    indices = [x for x in device.split(",") if x.isdigit()]  # requested ids, excluding '-1' and non-numeric tokens
    if indices and all(x in visible for x in indices) and any(int(x) >= torch.cuda.device_count() for x in indices):
        # Ids exceeding the torch device count can only be physical GPU ids under an external CUDA_VISIBLE_DEVICES
        # restriction -> translate to torch indices; in-range ids are torch indices, keeping repeated parses stable
        device = ",".join(str(visible.index(x)) if x.isdigit() else x for x in device.split(","))
    if "-1" in device:
        # Replace each -1 with an idle GPU or remove it; GPUInfo searches physical NVML ids among externally visible
        # GPUs only, translated back to torch indices under a CUDA_VISIBLE_DEVICES restriction
        parts = device.split(",")
        candidates = [int(x) for x in visible if x.isdigit()] if visible else None
        selected = GPUInfo().select_idle_gpu(count=parts.count("-1"), min_memory_fraction=0.2, indices=candidates)
        selected = [visible.index(str(x)) for x in selected] if visible else selected
        for i in range(len(parts)):
            if parts[i] == "-1":
                parts[i] = str(selected.pop(0)) if selected else ""
        device = ",".join(p for p in parts if p)
    return device


def select_device(device="", newline=False, verbose=True):
    """Select the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object. Options include 'cpu', 'cuda', '0',
            '0,1,2,3', 'mps', 'npu', 'npu:0', or '-1' for auto-select. Defaults to auto-selecting the first available
            GPU, or CPU if no GPU is available.
        newline (bool, optional): If True, adds a newline at the end of the log string.
        verbose (bool, optional): If True, logs the device information.

    Returns:
        (torch.device): Selected device.

    Examples:
        >>> select_device("cuda:0")
        device(type='cuda', index=0)

        >>> select_device("cpu")
        device(type='cpu')

    Notes:
        CUDA indices are torch device indices, which reflect any externally set CUDA_VISIBLE_DEVICES. This function
        never modifies CUDA_VISIBLE_DEVICES; an explicit single-GPU request is made the default CUDA device with
        torch.cuda.set_device() so that indexless 'cuda' operations land on it, while default '' requests (resolved
        to the current device) and multi-GPU requests (DDP ranks pin their own device in trainer._setup_ddp()) leave
        the current device untouched.
    """
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return device  # non-CUDA torch.device inputs pass through; cuda ones canonicalize via parse_device below
    elif str(device).startswith(("tpu", "intel", "vulkan")):
        return device

    s = f"Vision AI Platform 🚀 Python-{PYTHON_VERSION} torch-{TORCH_VERSION} "
    device = parse_device(device)

    # Huawei Ascend NPU
    if device.startswith("npu"):
        try:
            import torch_npu  # noqa
        except ImportError:
            raise ValueError(f"Invalid NPU 'device={device}'. Install 'torch_npu' at https://github.com/Ascend/pytorch")

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise ValueError(f"Invalid NPU 'device={device}' requested. Ascend NPU is not available.")

        # Parse 'npu' or 'npu:N' (multi-NPU not yet supported)
        suffix = device[3:]
        if suffix == "":
            idx = 0
        elif suffix.startswith(":") and suffix[1:].isdigit():
            idx = int(suffix[1:])
        else:
            raise ValueError(f"Invalid NPU 'device={device}' format. Use 'npu' or 'npu:0'.")

        n = torch.npu.device_count()
        if idx >= n:
            raise ValueError(f"Invalid NPU 'device={device}' requested. Only {n} NPU(s) available.")

        torch.npu.set_device(idx)
        if verbose:
            LOGGER.info(f"{s}NPU:{idx} ({torch.npu.get_device_name(idx)})\n")
        return torch.device(f"npu:{idx}")

    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}  # Apple Metal Performance Shaders (MPS)
    if not cpu and not mps and device:  # non-cpu device requested
        valid = all(x.isdigit() and int(x) < torch.cuda.device_count() for x in device.split(","))
        if not (torch.cuda.is_available() and valid):
            LOGGER.info(s)
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else [str(torch.cuda.current_device())]  # '' -> current default device
        space = " " * len(s)
        for i, d in enumerate(devices):
            s += f"{'' if i == 0 else space}CUDA:{d} ({get_gpu_info(int(d))})\n"
        arg = f"cuda:{devices[0]}"
        if device and len(devices) == 1:  # explicit single-GPU request only: '' never moves the current device, and
            torch.cuda.set_device(int(devices[0]))  # multi-GPU DDP ranks each pin their own device in _setup_ddp()
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        # Prefer MPS if available
        s += f"MPS ({get_cpu_info()})\n"
        arg = "mps"
    else:  # revert to CPU
        s += f"CPU ({get_cpu_info()})\n"
        arg = "cpu"

    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)  # reset OMP_NUM_THREADS for cpu training
    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


def unwrap_model(m: nn.Module) -> nn.Module:
    """Unwrap compiled and parallel models to get the base model.

    Args:
        m (nn.Module): A model that may be wrapped by torch.compile (._orig_mod) or parallel wrappers such as
            DataParallel/DistributedDataParallel (.module).

    Returns:
        (nn.Module): The unwrapped base model without compile or parallel wrappers.
    """
    while True:
        if hasattr(m, "_orig_mod") and isinstance(m._orig_mod, nn.Module):
            m = m._orig_mod
        elif hasattr(m, "module") and isinstance(m.module, nn.Module):
            m = m.module
        else:
            return m


def model_info(model, detailed=False, verbose=True, imgsz=640):
    """Print and return detailed model information layer by layer.

    Args:
        model (nn.Module): Model to analyze.
        detailed (bool, optional): Whether to print detailed layer information.
        verbose (bool, optional): Whether to print model information.
        imgsz (int | list, optional): Input image size.

    Returns:
        (tuple): Tuple containing:
            - n_l (int): Number of layers.
            - n_p (int): Number of parameters.
            - n_g (int): Number of gradients.
            - flops (float): GFLOPs.
    """
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    layers = __import__("collections").OrderedDict((n, m) for n, m in model.named_modules() if len(m._modules) == 0)
    n_l = len(layers)  # number of layers
    if detailed:
        h = f"{'layer':>5}{'name':>40}{'type':>20}{'gradient':>10}{'parameters':>12}{'shape':>20}{'mu':>10}{'sigma':>10}"
        LOGGER.info(h)
        for i, (mn, m) in enumerate(layers.items()):
            mn = mn.replace("module_list.", "")
            mt = m.__class__.__name__
            if len(m._parameters):
                for pn, p in m.named_parameters():
                    LOGGER.info(
                        f"{i:>5g}{f'{mn}.{pn}':>40}{mt:>20}{p.requires_grad!r:>10}{p.numel():>12g}{list(p.shape)!s:>20}{p.mean():>10.3g}{p.std():>10.3g}{str(p.dtype).replace('torch.', ''):>15}"
                    )
            else:  # layers with no learnable params
                LOGGER.info(f"{i:>5g}{mn:>40}{mt:>20}{False!r:>10}{0:>12g}{[]!s:>20}{'-':>10}{'-':>10}{'-':>15}")

    flops = get_flops(model, imgsz)  # imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320]
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")
    return n_l, n_p, n_g, flops


def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def model_info_for_loggers(trainer):
    """Return model info dict with useful model information.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer object containing model and validation data.

    Returns:
        (dict): Dictionary containing model parameters, GFLOPs, and inference speeds.

    Examples:
        YOLOv8n info for loggers
        >>> results = {
        ...     "model/parameters": 3151904,
        ...     "model/GFLOPs": 8.746,
        ...     "model/speed_ONNX(ms)": 41.244,
        ...     "model/speed_TensorRT(ms)": 3.211,
        ...     "model/speed_PyTorch(ms)": 18.755,
        ... }
    """
    results = {
        "model/parameters": get_num_params(trainer.model),
        "model/GFLOPs": round(get_flops(trainer.model), 3),
    }
    results["model/speed_PyTorch(ms)"] = round(trainer.validator.speed["inference"], 3)
    return results


def get_flops(model, imgsz=640):
    """Calculate FLOPs (floating point operations) for a model in GFLOPs.

    Attempts two calculation methods: first with a stride-based tensor for efficiency, then falls back to full image
    size if needed (e.g., for RTDETR models). Returns 0.0 if thop library is unavailable or calculation fails.

    Args:
        model (nn.Module): The model to calculate FLOPs for.
        imgsz (int | list, optional): Input image size.

    Returns:
        (float): The model's GFLOPs (billions of floating point operations).
    """
    try:
        import thop
    except ImportError:
        thop = None  # conda support without 'ultralytics-thop' installed

    if not thop:
        return 0.0  # if not installed return 0.0 GFLOPs

    try:
        model = unwrap_model(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]  # expand if int/float
        try:
            # Method 1: Use stride-based input tensor
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device, dtype=p.dtype)  # input image in BCHW
            flops = thop.profile(model, inputs=[im], verbose=False)[0] / 1e9 * 2  # stride GFLOPs
            return flops * imgsz[0] / stride * imgsz[1] / stride  # imgsz GFLOPs
        except Exception:
            # Method 2: Use actual image size (required for RTDETR models)
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device, dtype=p.dtype)  # input image in BCHW format
            return thop.profile(model, inputs=[im], verbose=False)[0] / 1e9 * 2  # imgsz GFLOPs
    except Exception:
        return 0.0


def initialize_weights(model):
    """Initialize model weights, biases, and module settings to default values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """Scale and pad an image tensor, optionally maintaining aspect ratio and padding to gs multiple.

    Args:
        img (torch.Tensor): Input image tensor.
        ratio (float, optional): Scaling ratio.
        same_shape (bool, optional): Whether to maintain the same shape.
        gs (int, optional): Grid size for padding.

    Returns:
        (torch.Tensor): Scaled and padded image tensor.
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from object 'b' to object 'a', with options to include/exclude certain attributes.

    Args:
        a (Any): Destination object to copy attributes to.
        b (Any): Source object to copy attributes from.
        include (tuple, optional): Attributes to include. If empty, all attributes are included.
        exclude (tuple, optional): Attributes to exclude.
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def intersect_dicts(da, db, exclude=()):
    """Return a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values.

    Args:
        da (dict): First dictionary.
        db (dict): Second dictionary.
        exclude (tuple, optional): Keys to exclude.

    Returns:
        (dict): Dictionary of intersecting keys with matching shapes.
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def time_sync():
    """Return PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d and BatchNorm2d layers for inference optimization.

    Args:
        conv (nn.Conv2d): Convolutional layer to fuse.
        bn (nn.BatchNorm2d): Batch normalization layer to fuse.

    Returns:
        (nn.Conv2d): The fused convolutional layer with gradients disabled.

    Examples:
        >>> conv = nn.Conv2d(3, 16, 3)
        >>> bn = nn.BatchNorm2d(16)
        >>> fused_conv = fuse_conv_and_bn(conv, bn)
    """
    # Compute fused weights: Conv2d weight is [out_channels, in_channels // groups, kH, kW], scale along axis 0
    bn_scale = bn.weight.div(torch.sqrt(bn.eps + bn.running_var))
    conv.weight.data = conv.weight * bn_scale.view(-1, 1, 1, 1)

    # Compute fused bias
    b_conv = (
        torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = bn_scale * b_conv + b_bn

    if conv.bias is None:
        conv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        conv.bias.data = fused_bias

    return conv.requires_grad_(False)


def fuse_deconv_and_bn(deconv, bn):
    """Fuse ConvTranspose2d and BatchNorm2d layers for inference optimization.

    Args:
        deconv (nn.ConvTranspose2d): Transposed convolutional layer to fuse.
        bn (nn.BatchNorm2d): Batch normalization layer to fuse.

    Returns:
        (nn.ConvTranspose2d): The fused transposed convolutional layer with gradients disabled.

    Examples:
        >>> deconv = nn.ConvTranspose2d(16, 3, 3)
        >>> bn = nn.BatchNorm2d(3)
        >>> fused_deconv = fuse_deconv_and_bn(deconv, bn)
    """
    if isinstance(bn, nn.Identity):  # ConvTranspose(bn=False) leaves bn as nn.Identity, nothing to fuse
        return deconv.requires_grad_(False)
    # Compute fused weights: ConvTranspose2d weight is [in_channels, out_channels // groups, kH, kW], so the
    # per-output-channel BN scale applies along axis 1 (group-mapped from axis 0), not axis 0 as for Conv2d.
    bn_scale = bn.weight.div(torch.sqrt(bn.eps + bn.running_var))
    w_scale = bn_scale.view(deconv.groups, -1).repeat_interleave(deconv.in_channels // deconv.groups, 0)
    deconv.weight.data = deconv.weight * w_scale[:, :, None, None]

    # Compute fused bias
    b_conv = (
        torch.zeros(deconv.out_channels, device=deconv.weight.device, dtype=deconv.weight.dtype)
        if deconv.bias is None
        else deconv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = bn_scale * b_conv + b_bn

    if deconv.bias is None:
        deconv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        deconv.bias.data = fused_bias

    return deconv.requires_grad_(False)


class ModelEMA:
    """Updated Exponential Moving Average (EMA) implementation.

    Keeps a moving average of everything in the model state_dict (parameters and buffers). For EMA details see
    References.

    To disable EMA set the `enabled` attribute to `False`.

    Attributes:
        ema (nn.Module): Copy of the model in evaluation mode.
        updates (int): Number of EMA updates.
        decay (function): Decay function that determines the EMA weight.
        enabled (bool): Whether EMA is enabled.

    References:
        - https://github.com/rwightman/pytorch-image-models
        - https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Initialize EMA for 'model' with given arguments.

        Args:
            model (nn.Module): Model to create EMA for.
            decay (float, optional): Maximum EMA decay rate.
            tau (int, optional): EMA decay time constant.
            updates (int, optional): Initial number of updates.
        """
        self.ema = deepcopy(unwrap_model(model)).eval()  # FP32 EMA
        if hasattr(self.ema, "teacher_model"):
            # DistillationModel: strip the teacher so the EMA does not carry a full duplicate copy.
            self.ema.teacher_model = None
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters.

        Args:
            model (nn.Module): Model to update EMA from.
        """
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = unwrap_model(model).state_dict()  # model state_dict
            ema_v, model_v = [], []
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    ema_v.append(v)
                    model_v.append(msd[k])
            if ema_v and TORCH_2_0 and (TORCH_2_4 or ema_v[0].device.type != "mps"):  # one kernel launch per op
                torch._foreach_lerp_(ema_v, model_v, 1 - d)
            else:  # _foreach_lerp_ needs torch>=2.0 and, on MPS, torch>=2.4
                for v, m in zip(ema_v, model_v):
                    v.mul_(d).add_(m, alpha=1 - d)

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Copy attributes from model to EMA, with options to include/exclude certain attributes.

        Args:
            model (nn.Module): Model to copy attributes from.
            include (tuple, optional): Attributes to include.
            exclude (tuple, optional): Attributes to exclude.
        """
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)