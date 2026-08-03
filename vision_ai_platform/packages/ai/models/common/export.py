from pathlib import Path
import os
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
import shutil

import numpy as np
import torch
import torch.nn as nn

from vision_ai_platform import __version__
from vision_ai_platform.packages.core import YOLOConfig, BaseExporter
from vision_ai_platform.packages.core.exporter import (
    FP16_FORMATS,
    INT8_FORMATS,
    W8A16_FORMATS,
    W8A32_FORMATS,
    FP32_UNSUPPORTED_FORMATS,
    QUANTIZE_PRECISIONS,
    QUANTIZE_DOCS_URL,
    export_formats,
    ClassMapModel,
    NormalizedExportWrapper,
)
from vision_ai_platform.packages.utils import (
    LOGGER,
    callbacks,
    TASK2CALIBRATIONDATA,
    LINUX,
    ARM64,
    is_jetson,
    DEFAULT_CFG,
    MACOS,
    MACOS_VERSION,
    TASK2DATA,
    SETTINGS,
    TORCH_VERSION,
    WINDOWS,
    YAML,
    colorstr,
    get_default_args,
)
from vision_ai_platform.packages.utils.ops import Profile
from vision_ai_platform.packages.utils.check import (
    check_version,
    check_imgsz,
    is_intel,
    check_requirements,
    IS_PYTHON_MINIMUM_3_13,
)
from vision_ai_platform.packages.utils.patches import arange_patch
from vision_ai_platform.packages.utils.device_utils import (
    select_device,
    TORCH_1_11,
    TORCH_1_13,
    TORCH_2_1,
    TORCH_2_3,
)
from vision_ai_platform.packages.utils.nms import TorchNMS
from vision_ai_platform.packages.utils.loss import batch_probiou
from vision_ai_platform.packages.utils.files import file_size
from vision_ai_platform.packages.ai.data.dataloader import build_dataloader
from vision_ai_platform.packages.ai.data.utils import check_class_names, check_cls_dataset, check_det_dataset
from vision_ai_platform.packages.ai.data.yolo import build_yolo_dataset, ClassificationDataset
from vision_ai_platform.packages.ai.nn.modules import (
    Detect, Segment, Segment26, RTDETRDecoder, Classify, C2f, SemanticSegment
)
from vision_ai_platform.packages.ai.nn.tasks import (
    ClassificationModel, DetectionModel, SegmentationModel, WorldModel, DepthModel
)
from vision_ai_platform.packages.ai.nn.autobackend import default_class_names, AutoBackend


def validate_args(format, passed_args, valid_args):
    """Validate arguments based on the export format.

    Args:
        format (str): The export format.
        passed_args (SimpleNamespace): The arguments used during export.
        valid_args (list): List of valid arguments for the format.

    Raises:
        AssertionError: If an unsupported argument is used, or if the format lacks supported argument listings.
    """
    export_args = ["dynamic", "keras", "nms", "batch", "fraction", "data", "optimize"]

    assert valid_args is not None, f"ERROR ❌️ valid arguments for '{format}' not listed."
    custom = {"batch": 1, "data": None, "device": None}  # exporter defaults
    default_args = DEFAULT_CFG.model_copy(update=custom)
    if passed_args.quantize is not None:  # 32/None (FP32) is universal except FP32_UNSUPPORTED_FORMATS
        options = [label for label, formats in QUANTIZE_PRECISIONS if format in formats]
        if format not in FP32_UNSUPPORTED_FORMATS:
            options.append("32 (FP32)")
        hint = f"format='{format}' supports quantize={', '.join(options) or 'none'} (or None for FP32). See {QUANTIZE_DOCS_URL}"
        if passed_args.quantize == 16:  # FP16
            assert format in FP16_FORMATS, f"ERROR ❌️ quantize=16 (FP16) is not supported; {hint}"
        elif passed_args.quantize == 8:  # INT8
            assert format in INT8_FORMATS, f"ERROR ❌️ quantize=8 (INT8) is not supported; {hint}"
        elif passed_args.quantize == "w8a16":  # INT8 weights + 16-bit activations (FP16; INT16 on LiteRT)
            assert format in W8A16_FORMATS, f"ERROR ❌️ quantize='w8a16' is not supported; {hint}"
        elif passed_args.quantize == "w8a32":  # INT8 weights + FP32 activations (dynamic/weight-only INT8)
            assert format in W8A32_FORMATS, f"ERROR ❌️ quantize='w8a32' is not supported; {hint}"
        elif passed_args.quantize == 32:  # FP32
            assert format not in FP32_UNSUPPORTED_FORMATS, f"ERROR ❌️ quantize=32 (FP32) is not supported; {hint}"
    for arg in export_args:
        not_default = getattr(passed_args, arg, getattr(default_args, arg, None)) != getattr(default_args, arg, None)
        if not_default:
            assert arg in valid_args, f"ERROR ❌️ argument '{arg}' is not supported for format='{format}'"


def try_export(inner_func):
    """YOLO export decorator, i.e. @try_export."""
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Export a model."""
        prefix = inner_args["prefix"]
        dt = 0.0
        try:
            with Profile() as dt:
                f = inner_func(*args, **kwargs)  # exported file/dir or tuple of (file/dir, *)
            path = f if isinstance(f, (str, Path)) else f[0]
            mb = file_size(path)
            assert mb > 0.1, f"{mb:.3f} MB output model too small (likely corrupt or unsupported ops)"
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as '{path}' ({mb:.1f} MB)")
            return f
        except Exception as e:
            dependency_help = (
                " Ultralytics Platform runs exports in the cloud with no local dependencies required. "
                "Visit https://platform.ultralytics.com."
                if isinstance(e, ImportError)
                else ""
            )
            LOGGER.error(f"{prefix} export failure {dt.t:.1f}s: {e}{dependency_help}")
            raise e

    return outer_func


class Exporter(BaseExporter):
    def __init__(self, cfg: YOLOConfig, _callbacks: dict | None = None):
        """Initialize the Exporter class.

        Args:
            cfg (str | Path | dict | SimpleNamespace, optional): Configuration file path or configuration object.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        super().__init__(cfg, _callbacks)
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def __call__(self, model=None) -> str:
        """Export a model and return the final exported path as a string.

        Returns:
            (str): Path to the exported file or directory (the last export artifact).
        """
        t = time.time()
        fmt = self.cfg.format.lower()  # to lowercase
        if fmt in {"tensorrt", "trt"}:  # 'engine' aliases
            fmt = "engine"
        if fmt in {"mlmodel", "mlpackage", "mlprogram", "apple", "ios", "coreml"}:  # 'coreml' aliases
            fmt = "coreml"
        if fmt in {"tflite", "tfjs"}:  # deprecated formats, replaced by the unified Google LiteRT export
            LOGGER.warning(
                f"format='{fmt}' is deprecated as of 8.4.83 and has been replaced by the unified Google LiteRT "
                f"format. Exporting format='litert' instead. See https://docs.ultralytics.com/integrations/litert/"
            )
            fmt = self.cfg.format = "litert"
        fmts_dict = export_formats()
        fmts = tuple(fmts_dict["Argument"][1:])  # available export formats
        if fmt not in fmts:
            import difflib

            # Get the closest match if format is invalid
            matches = difflib.get_close_matches(fmt, fmts, n=1, cutoff=0.6)  # 60% similarity required to match
            if not matches:
                msg = "Model is already in PyTorch format." if fmt == "pt" else f"Invalid export format='{fmt}'."
                raise ValueError(f"{msg} Valid formats are {fmts}")
            LOGGER.warning(f"Invalid export format='{fmt}', updating to format='{matches[0]}'")
            fmt = matches[0]
        is_tf_format = fmt in {"saved_model", "pb", "edgetpu"}

        # Device
        self.dla = None
        if fmt == "engine" and self.cfg.device is None:
            LOGGER.warning("TensorRT requires GPU export, automatically assigning device=0")
            self.cfg.device = "0"
        if fmt == "engine" and "dla" in str(self.cfg.device):  # convert int/list to str first
            device_str = str(self.cfg.device)
            self.dla = device_str.rsplit(":", 1)[-1]
            self.cfg.device = "0"  # update device to "0"
            assert self.dla in {"0", "1"}, f"Expected device 'dla:0' or 'dla:1', but got {device_str}."
        self.device = select_device("cpu" if self.cfg.device is None else self.cfg.device)

        # Argument compatibility checks
        fmt_keys = dict(zip(fmts_dict["Argument"], fmts_dict["Arguments"]))[fmt]
        validate_args(fmt, self.cfg, fmt_keys)
        if fmt in {"deepx", "axelera", "imx", "edgetpu", "qnn", "hailo"} and self.cfg.quantize not in {8, "w8a16"}:
            if self.cfg.quantize == 32:
                raise ValueError(
                    f"{fmt} export only supports INT8, but got an explicit quantize=32 (FP32) request. "
                    f"See {QUANTIZE_DOCS_URL}"
                )
            LOGGER.warning(f"{fmt} export requires INT8 quantization, enabling it.")
            self.cfg.quantize = "w8a16" if fmt == "qnn" else 8
        if fmt in {"axelera", "hailo"} and not self.cfg.data:
            self.cfg.data = TASK2CALIBRATIONDATA.get(model.task)
        if fmt == "hailo":
            assert LINUX and not ARM64, "Hailo export is only supported on Linux x86_64."
            blocks = {str(x[2]) for x in model.yaml.get("backbone", []) + model.yaml.get("head", [])}
            family = Path(getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")).stem.lower() or (
                "yolov8" if "C2f" in blocks else "yolo11" if {"C3k2", "C2PSA"} <= blocks else ""
            )
            if isinstance(model.model[-1], Segment26):
                raise ValueError("Hailo export does not currently support YOLO26 segmentation models.")
            if (
                model.task not in {"detect", "segment"}
                or type(model.model[-1]) not in {Detect, Segment}
                or not family.startswith(("yolov8", "yolo11", "yolo26"))
            ):
                raise ValueError(
                    "Hailo export currently supports YOLOv8, YOLO11, and YOLO26 detection models "
                    "and YOLOv8/YOLO11 segmentation models."
                )
            if self.cfg.end2end is not None:
                raise ValueError(
                    "Hailo export selects the model output path automatically; remove the end2end argument."
                )
            if self.cfg.opset not in {None, 11}:
                raise ValueError("Hailo export requires opset=11.")
            self.cfg.name = str(self.cfg.name or "hailo8l").lower()
            hailo_archs = ("hailo8", "hailo8l", "hailo10h", "hailo15h", "hailo15l")
            if self.cfg.name not in hailo_archs:
                raise ValueError(f"Invalid Hailo architecture '{self.cfg.name}'. Valid names are {hailo_archs}.")
        if fmt == "axelera":
            if model.task == "segment" and any(isinstance(m, Segment26) for m in model.modules()):
                raise ValueError("Axelera export does not currently support YOLO26 segmentation models.")
        if fmt == "imx":
            if not self.cfg.nms and model.task in {"detect", "pose", "segment"}:
                LOGGER.warning("IMX export requires nms=True, setting nms=True.")
                self.cfg.nms = True
            if model.task not in {"detect", "pose", "classify", "segment"}:
                raise ValueError(
                    "IMX export only supported for detection, pose estimation, classification, and segmentation models."
                )
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if hasattr(model, "end2end"):
            if self.cfg.end2end is not None:
                model.end2end = self.cfg.end2end
            if fmt in {"rknn", "ncnn", "executorch", "paddle", "imx", "edgetpu", "qnn"}:
                # Disable end2end branch for certain export formats as they does not support topk
                model.end2end = False
                LOGGER.warning(f"{fmt.upper()} export does not support end2end models, disabling end2end branch.")
            if fmt == "litert" and self.cfg.quantize in {8, "w8a16"}:
                # Static activation quantization collapses the end2end class-index output; export raw and run NMS later
                model.end2end = False
                LOGGER.warning("LiteRT INT8 export does not support end2end models, disabling end2end branch.")
            if fmt == "engine":
                try:
                    import tensorrt as trt

                    if check_version(trt.__version__, "<8.5.0"):
                        # https://github.com/ultralytics/ultralytics/issues/24607
                        model.end2end = False
                        LOGGER.warning(
                            "TensorRT versions earlier than 8.5.0 do not support the Mod operator in end-to-end models, disabling the end2end branch. "
                            "Please upgrade TensorRT to 8.5.0 or later to enable end2end export."
                        )

                    if (
                        self.cfg.quantize == 8
                        and check_version(trt.__version__, ">=10.3.0,<10.4.0")  # JetPack 6 builds report 10.3.0.x
                        and is_jetson(jetpack=6)
                    ):
                        # https://github.com/ultralytics/ultralytics/issues/23841
                        model.end2end = False
                        LOGGER.warning(
                            "TensorRT 10.3.0 on JetPack 6 with int8 has known end2end build issues, disabling end2end branch. "
                            "For a fix, see https://docs.ultralytics.com/guides/nvidia-jetson/#why-does-my-tensorrt-int8-export-disable-end2end-on-jetpack-6"
                            ""
                        )
                except ImportError:
                    pass
        if self.cfg.quantize == 16 and fmt == "torchscript" and self.device.type == "cpu":
            raise ValueError("FP16 TorchScript export is only supported on GPU, i.e. use device=0.")
        self.imgsz = check_imgsz(self.cfg.imgsz, stride=model.stride, min_dim=2)  # check image size
        if self.cfg.nms and model.task == "semantic":
            LOGGER.warning("'nms=True' is not valid for semantic segmentation models. Forcing 'nms=False'.")
            self.cfg.nms = False
        if fmt == "coreml" and self.cfg.nms and model.task != "detect":
            LOGGER.warning("CoreML 'nms=True' is only supported for detect models. Forcing 'nms=False'.")
            self.cfg.nms = False
        if self.cfg.nms:
            assert not isinstance(model, ClassificationModel), "'nms=True' is not valid for classification models."
            assert not is_tf_format or TORCH_1_13, "TensorFlow exports with NMS require torch>=1.13"
            assert fmt != "onnx" or TORCH_1_13, "ONNX export with NMS requires torch>=1.13"
            if getattr(model, "end2end", False) or isinstance(model.model[-1], RTDETRDecoder):
                LOGGER.warning("'nms=True' is not available for end2end models. Forcing 'nms=False'.")
                self.cfg.nms = False
            self.cfg.conf = self.cfg.conf or 0.25  # set conf default value for nms export
        if fmt == "mnn" and self.cfg.nms:
            if self.cfg.dynamic:
                raise ValueError("Alibaba MNN export does not support combining 'dynamic=True' with 'nms=True'.")
            if model.task not in {"detect", "pose"}:
                raise ValueError("Alibaba MNN export with 'nms=True' only supports detect and pose models.")
        if (fmt in {"engine", "coreml"} or self.cfg.nms) and self.cfg.dynamic and self.cfg.batch == 1:
            LOGGER.warning(
                f"'dynamic=True' model with '{'nms=True' if self.cfg.nms else f'format={self.cfg.format}'}' requires max batch size, i.e. 'batch=16'"
            )
        if fmt == "edgetpu":
            if not LINUX or ARM64:
                raise SystemError(
                    "Edge TPU export only supported on non-aarch64 Linux. See https://coral.ai/docs/edgetpu/compiler"
                )
            elif self.cfg.batch != 1:  # see github.com/ultralytics/ultralytics/pull/13420
                LOGGER.warning("Edge TPU export requires batch size 1, setting batch=1.")
                self.cfg.batch = 1
        if isinstance(model, WorldModel):
            LOGGER.warning(
                "YOLOWorld (original version) export is not supported to any format. "
                "YOLOWorldv2 models (i.e. 'yolov8s-worldv2.pt') only support export to "
                "(torchscript, onnx, openvino, engine, coreml) formats. "
                "See https://docs.ultralytics.com/models/yolo-world for details."
            )
            model.clip_model = None  # openvino int8 export error: https://github.com/ultralytics/ultralytics/pull/18445
        if self.cfg.quantize in {8, "w8a16"} and not self.cfg.data:
            self.cfg.data = DEFAULT_CFG.data or TASK2DATA[getattr(model, "task", "detect")]  # assign default data
            LOGGER.warning(
                f"INT8 export requires a missing 'data' arg for calibration. Using default 'data={self.cfg.data}'."
            )
        # Recommend OpenVINO if export and Intel CPU
        if SETTINGS.get("openvino_msg"):
            if is_intel():
                LOGGER.info(
                    "💡 ProTip: Export to OpenVINO format for best performance on Intel hardware."
                    " Learn more at https://docs.ultralytics.com/integrations/openvino/"
                )
            SETTINGS["openvino_msg"] = False

        # Input
        im = torch.zeros(self.cfg.batch, model.yaml.get("channels", 3), *self.imgsz).to(self.device)
        file = Path(
            getattr(model, "pt_path", None) or getattr(model, "yaml_file", None) or model.yaml.get("yaml_file", "")
        )
        if file.suffix in {".yaml", ".yml"}:
            file = Path(file.name)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        model = model.fuse()

        if fmt == "edgetpu":
            from vision_ai_platform.packages.utils.export.tensorflow import tf_wrapper

            model = tf_wrapper(model)
        for m in model.modules():
            if isinstance(m, (Classify, SemanticSegment)):
                m.export = True
                m.format = self.cfg.format
                # Semantic argmax bake needs an integer graph output; TensorRT supports uint8 outputs only on TRT>=10
                # (Jetson TRT 8.x rejects them). Read the version from the package name to avoid importing tensorrt here.
                if isinstance(m, SemanticSegment) and fmt == "engine":
                    cuda_major = (torch.version.cuda or "12").split(".")[0]
                    m.bake_argmax = check_version(f"tensorrt-cu{cuda_major}", ">=10.0.0") or check_version(
                        "tensorrt", ">=10.0.0"
                    )
            if isinstance(m, (Detect, RTDETRDecoder)):  # includes all Detect subclasses like Segment, Pose, OBB
                m.dynamic = self.cfg.dynamic
                m.export = True
                m.format = self.cfg.format
                # Clamp max_det to available queries/anchors (required for TensorRT compatibility)
                available = (
                    m.num_queries
                    if isinstance(m, RTDETRDecoder)
                    else sum(int(self.imgsz[0] / s) * int(self.imgsz[1] / s) for s in model.stride.tolist())
                )
                m.max_det = min(self.cfg.max_det, available)
                m.agnostic_nms = self.cfg.agnostic_nms
                m.xyxy = self.cfg.nms and fmt != "coreml"
                m.shape = None  # reset cached shape for new export input size
                if hasattr(model, "pe") and hasattr(m, "fuse") and not hasattr(m, "lrpc"):  # for YOLOE models
                    m.fuse(model.pe.to(self.device))
            elif isinstance(m, C2f) and not is_tf_format:
                # EdgeTPU does not support FlexSplitV while split provides cleaner ONNX graph
                m.forward = m.forward_split

        if model.task == "semantic" and fmt in {"qnn", "coreml"}:
            # NPU-targeted semantic exports ship a compact uint8 class map instead of float logits: emitting logits
            # forces consumers to dequantize and argmax ~20M floats on the CPU every frame (measured erratic
            # 123-1065 ms on Hexagon). Not applied to LiteRT, where the GPU delegate cannot compile ArgMax (int64
            # indices) and a whole-graph CPU fallback is slower than GPU logits + consumer-side argmax. Python
            # predict/val accept both forms.
            model = ClassMapModel(model)

        y = None
        for _ in range(2):  # dry runs
            y = NMSModel(model, self.cfg)(im) if self.cfg.nms and fmt not in {"coreml", "imx"} else model(im)
        if self.cfg.quantize == 16 and fmt in {"onnx", "torchscript"} and self.device.type != "cpu":
            im, model = im.half(), model.half()  # to FP16

        # Assign
        self.im = im
        self.model = model
        self.file = file
        self.output_shape = (
            tuple(y.shape)
            if isinstance(y, torch.Tensor)
            else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for x in y)
        )
        self.pretty_name = Path(self.model.yaml.get("yaml_file", self.file)).stem.replace("yolo", "YOLO")
        data = model.args["data"] if hasattr(model, "args") and isinstance(model.args, dict) else ""
        description = f"Ultralytics {self.pretty_name} model {f'trained on {data}' if data else ''}"
        self.metadata = {
            "description": description,
            "author": "Ultralytics",
            "date": datetime.now().isoformat(),
            "version": __version__,
            "stride": int(max(model.stride)),
            "task": model.task,
            "head": type(model.model[-1]).__name__,
            "batch": self.cfg.batch,
            "imgsz": self.imgsz,
            "names": model.names,
            "args": {k: v for k, v in self.cfg if k in fmt_keys},
            "channels": model.yaml.get("channels", 3),
            "end2end": getattr(model, "end2end", False),
        }  # model metadata
        if self.dla is not None:
            self.metadata["dla"] = self.dla  # make sure `AutoBackend` uses correct dla device if it has one
        if model.task == "pose":
            self.metadata["kpt_shape"] = model.model[-1].kpt_shape
            if hasattr(model, "kpt_names"):
                self.metadata["kpt_names"] = model.kpt_names

        LOGGER.info(
            f"\n{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and "
            f"output shape(s) {self.output_shape} ({file_size(file):.1f} MB)"
        )
        self.run_callbacks("on_export_start")

        # Export
        if is_tf_format:
            f, keras_model = self.export_saved_model()
            if fmt == "pb":
                f = self.export_pb(keras_model=keras_model)
            if fmt == "edgetpu":
                f = self.export_edgetpu(tflite_model=Path(f) / f"{self.file.stem}_full_integer_quant.tflite")
        else:
            f = getattr(self, f"export_{fmt}")()

        # Finish
        if f:
            square = self.imgsz[0] == self.imgsz[1]
            s = (
                ""
                if square
                else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not "
                f"work. Use export 'imgsz={max(self.imgsz)}' if val is required."
            )
            imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")
            q = "quantize=16" if self.cfg.quantize == 16 else ""  # FP16 inference flag for the val/predict hint
            inference_commands = (
                f"\nPredict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q}"
                f"\nValidate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}"
                if fmt in AutoBackend._BACKEND_MAP
                else ""
            )
            LOGGER.info(
                f"\nExport complete ({time.time() - t:.1f}s)"
                f"\nResults saved to {colorstr('bold', Path(f).resolve())}"
                f"{inference_commands}"
                f"\nVisualize:       https://netron.app"
            )

        self.run_callbacks("on_export_end")
        return f  # path to final export artifact

    def get_int8_calibration_dataloader(self, prefix=""):
        """Build and return a dataloader for calibration of INT8 models."""
        LOGGER.info(f"{prefix} collecting INT8 calibration images from 'data={self.cfg.data}'")
        cfg = self.cfg.model_copy()
        cfg.imgsz = max(self.imgsz)
        if self.model.task == "classify":
            import torchvision.transforms as T  # scope for faster 'import ultralytics'

            data = check_cls_dataset(self.cfg.data, split=self.cfg.split)
            dataset = ClassificationDataset(data[self.cfg.split or "val"], args=cfg, augment=False)
            if self.cfg.fraction < 1.0:
                dataset.samples = dataset.samples[: round(len(dataset.samples) * self.cfg.fraction)]
            # INT8 backends divide images by 255, so emit uint8 [0, 255] center-cropped like classify inference
            dataset.torch_transforms = T.Compose([T.Resize(cfg.imgsz), T.CenterCrop(cfg.imgsz), T.PILToTensor()])
        else:
            data = check_det_dataset(self.cfg.data, split=self.cfg.split)
            dataset = build_yolo_dataset(
                cfg,
                data[self.cfg.split or "val"],
                self.cfg.batch,
                data,
                mode="val",
                fraction=self.cfg.fraction,
            )
        if hasattr(dataset, "transforms") and hasattr(dataset.transforms.transforms[0], "new_shape"):
            dataset.transforms.transforms[0].new_shape = self.imgsz  # LetterBox with non-square imgsz
        n = len(dataset)
        if n < 1:
            raise ValueError(f"The calibration dataset must have at least 1 image, but found {n} images.")
        batch = min(self.cfg.batch, n)
        if n < self.cfg.batch:
            LOGGER.warning(
                f"{prefix} calibration dataset has only {n} images, reducing calibration batch size to {batch}."
            )
        return build_dataloader(dataset, batch=batch, workers=0, drop_last=True)  # required for batch loading

    @try_export
    def export_torchscript(self, prefix=colorstr("TorchScript:")):
        """Export YOLO model to TorchScript format."""
        from vision_ai_platform.packages.utils.export.torchscript import torch2torchscript

        return torch2torchscript(
            model=NMSModel(self.model, self.cfg) if self.cfg.nms else self.model,
            im=self.im,
            output_file=self.file.with_suffix(".torchscript"),
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        """Export YOLO model to ONNX format."""
        requirements = ["onnx>=1.16.1,<1.19.0" if self.cfg.format == "rknn" else "onnx>=1.12.0,<2.0.0"]
        if self.cfg.simplify or (self.cfg.format == "onnx" and self.cfg.quantize == 8):
            # Pass onnxruntime variants as interchangeable candidates so AutoUpdate keeps an installed build
            # (e.g. onnxruntime-qnn for QNN export) instead of reinstalling stable onnxruntime and breaking its ABI.
            ort = "onnxruntime-gpu" if "cuda" in self.device.type else "onnxruntime"
            requirements += [(ort, "onnxruntime", "onnxruntime-gpu", "onnxruntime-qnn")]
        if self.cfg.simplify:
            requirements += ["onnxslim>=0.1.82"]
        check_requirements(requirements)
        import onnx

        from vision_ai_platform.packages.utils.export.engine import best_onnx_opset, torch2onnx

        opset = self.cfg.opset or best_onnx_opset(onnx, cuda="cuda" in self.device.type, quantize=self.cfg.quantize)
        assert not isinstance(self.model.model[-1], RTDETRDecoder) or opset >= 16, "RTDETR export requires opset>=16"
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset}...")
        if self.cfg.nms:
            assert TORCH_1_13, f"'nms=True' ONNX export requires torch>=1.13 (found torch=={TORCH_VERSION})"

        f = str(self.file.with_suffix(".onnx"))
        output_names = ["output0", "output1"] if self.model.task == "segment" else ["output0"]
        dynamic = self.cfg.dynamic
        if dynamic:
            dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
            if isinstance(self.model, SegmentationModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 116, 8400)
                dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
            elif isinstance(self.model, DepthModel):
                dynamic["output0"] = {0: "batch", 2: "height", 3: "width"}  # shape(1,1,640,640) dense map, not anchors
            elif isinstance(self.model, DetectionModel):
                dynamic["output0"] = {0: "batch", 2: "anchors"}  # shape(1, 84, 8400)
            if self.cfg.nms:  # only batch size is dynamic with NMS
                dynamic["output0"].pop(2)
        if self.cfg.nms and self.model.task == "obb":
            self.cfg.opset = opset  # for NMSModel
            self.cfg.simplify = True  # fix OBB runtime error related to topk

        model = NMSModel(self.model, self.cfg) if self.cfg.nms else self.model
        # Normalize coordinates by input size so RKNN's per-tensor INT8 scale preserves class scores.
        if (
            self.cfg.format == "rknn"
            and self.cfg.quantize == 8
            and self.model.task in {"detect", "segment", "pose", "obb"}
            and not self.metadata["end2end"]
        ):
            from vision_ai_platform.packages.utils.export.engine import _NormalizeCoords

            model = _NormalizeCoords(
                model,
                int(self.im.shape[2]),
                int(self.im.shape[3]),
                self.model.task,
                len(self.metadata["names"]),
                self.metadata.get("kpt_shape"),
            )

        with arange_patch(dynamic=bool(dynamic), quantize=self.cfg.quantize, fmt=self.cfg.format):
            torch2onnx(
                model,
                self.im,
                f,
                opset=opset,
                input_names=["images"],
                output_names=output_names,
                dynamic=dynamic or None,
            )

        # Checks
        model_onnx = onnx.load(f)  # load onnx model

        # Simplify
        if self.cfg.simplify:
            try:
                import onnxslim

                LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
                model_onnx = onnxslim.slim(model_onnx)

            except Exception as e:
                LOGGER.warning(f"{prefix} simplifier failure: {e}")

        # CANN requires the optional score-threshold input on ONNX NonMaxSuppression nodes. Scores were already
        # filtered by args.conf in NMSModel, so zero preserves the graph's semantics.
        if self.cfg.format == "ascend":
            for i, node in enumerate(model_onnx.graph.node):
                if node.op_type == "NonMaxSuppression" and len(node.input) == 4:
                    threshold_name = f"ascend_nms_score_threshold_{i}"
                    node.input.append(threshold_name)
                    model_onnx.graph.initializer.append(
                        onnx.helper.make_tensor(threshold_name, onnx.TensorProto.FLOAT, [1], [0.0])
                    )

        # Metadata
        for k, v in self.metadata.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        # IR version
        if getattr(model_onnx, "ir_version", 0) > 10:
            LOGGER.info(f"{prefix} limiting IR version {model_onnx.ir_version} to 10 for ONNXRuntime compatibility...")
            model_onnx.ir_version = 10

        # FP16 conversion for CPU export (GPU exports are already FP16 from model.half() during tracing)
        if self.cfg.quantize == 16 and self.cfg.format == "onnx" and self.device.type == "cpu":
            try:
                from onnxruntime.transformers import float16

                LOGGER.info(f"{prefix} converting to FP16...")
                model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
            except Exception as e:
                LOGGER.warning(f"{prefix} FP16 conversion failure: {e}")

        onnx.save(model_onnx, f)
        if self.cfg.quantize == 8 and self.cfg.format == "onnx":
            from vision_ai_platform.packages.utils.export.onnx import onnx_int8_quantize

            source = Path(f)
            f_int8 = str(source.with_name(f"{source.stem}_int8{source.suffix}"))
            f = onnx_int8_quantize(
                source,
                f_int8,
                self.get_int8_calibration_dataloader(prefix),
                self._transform_fn,
                batch=0 if self.cfg.dynamic else self.cfg.batch,
                prefix=prefix,
            )
            source.unlink(missing_ok=True)
        return f

    @try_export
    def export_openvino(self, prefix=colorstr("OpenVINO:")):
        """Export YOLO model to OpenVINO format."""
        from vision_ai_platform.packages.utils.export.openvino import torch2openvino

        # OpenVINO <= 2025.1.0 error on macOS 15.4+: https://github.com/openvinotoolkit/openvino/issues/30023
        check_requirements("openvino>=2025.2.0" if MACOS and MACOS_VERSION >= "15.4" else "openvino>=2024.0.0")
        import openvino as ov

        assert TORCH_2_1, f"OpenVINO export requires torch>=2.1 but torch=={TORCH_VERSION} is installed"

        def serialize(ov_model, file):
            """Set RT info, serialize, and save metadata YAML."""
            ov_model.set_rt_info("YOLO", ["model_info", "model_type"])
            ov_model.set_rt_info(True, ["model_info", "reverse_input_channels"])
            ov_model.set_rt_info(114, ["model_info", "pad_value"])
            ov_model.set_rt_info([255.0], ["model_info", "scale_values"])
            ov_model.set_rt_info(self.cfg.iou, ["model_info", "iou_threshold"])
            ov_model.set_rt_info([v.replace(" ", "_") for v in self.model.names.values()], ["model_info", "labels"])
            if self.model.task != "classify":
                ov_model.set_rt_info("fit_to_window_letterbox", ["model_info", "resize_type"])

            ov.save_model(ov_model, file, compress_to_fp16=self.cfg.quantize == 16)
            YAML.save(Path(file).parent / "metadata.yaml", self.metadata)  # add metadata.yaml

        calibration_dataset, ignored_scope = None, None
        if self.cfg.quantize == 8:
            check_requirements("packaging>=23.2")  # must be installed first to build nncf wheel
            check_requirements("nncf>=2.14.0,<3.0.0" if not TORCH_2_3 else "nncf>=2.14.0")
            import nncf

            calibration_dataset = nncf.Dataset(self.get_int8_calibration_dataloader(prefix), self._transform_fn)
            if isinstance(self.model.model[-1], Detect):
                # Includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect
                head_module_name = ".".join(list(self.model.named_modules())[-1][0].split(".")[:2])
                ignored_scope = nncf.IgnoredScope(patterns=[f".*{head_module_name}(/|\\.dfl).*"], types=["Sigmoid"])

        ov_model = torch2openvino(
            model=NMSModel(self.model, self.cfg) if self.cfg.nms else self.model,
            im=self.im,
            dynamic=self.cfg.dynamic,
            quantize=self.cfg.quantize,
            calibration_dataset=calibration_dataset,
            ignored_scope=ignored_scope,
            prefix=prefix,
        )

        suffix = f"_{'int8_' if self.cfg.quantize == 8 else ''}openvino_model{os.sep}"
        f = str(self.file).replace(self.file.suffix, suffix)
        f_ov = str(Path(f) / self.file.with_suffix(".xml").name)

        serialize(ov_model, f_ov)
        return f

    @try_export
    def export_paddle(self, prefix=colorstr("PaddlePaddle:")):
        """Export YOLO model to PaddlePaddle format."""
        from vision_ai_platform.packages.utils.export.paddle import torch2paddle

        model = NormalizedExportWrapper(self.model)
        return torch2paddle(
            model=model,
            im=self.im,
            output_dir=str(self.file).replace(self.file.suffix, f"_paddle_model{os.sep}"),
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_litert(self, prefix=colorstr("LiteRT:")):
        """Export YOLO model to LiteRT format using litert_torch with optional INT8 quantization.

        Supports ``quantize=8`` (static INT8, int8 weights + int8 activations, requires calibration ``data``),
        ``quantize='w8a16'`` (static, int8 weights + int16 activations, requires calibration ``data``) and
        ``quantize='w8a32'`` (dynamic/weight-only INT8, int8 weights + FP32 activations, no calibration needed).
        """
        assert MACOS or (LINUX and not ARM64), "LiteRT export only supported on Linux x86 and macOS"
        from vision_ai_platform.packages.utils.export.litert import torch2litert

        return torch2litert(
            self.model,
            self.im,
            self.file,
            quantize=self.cfg.quantize,
            calibration_dataset=self.get_int8_calibration_dataloader(prefix)
            if self.cfg.quantize in {8, "w8a16"}
            else None,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_mnn(self, prefix=colorstr("MNN:")):
        """Export YOLO model to MNN format using MNN https://github.com/alibaba/MNN."""
        from vision_ai_platform.packages.utils.export.mnn import onnx2mnn

        return onnx2mnn(
            onnx_file=self.export_onnx(),
            output_file=self.file.with_suffix(".mnn"),
            quantize=self.cfg.quantize,
            metadata=self.metadata,
            prefix=prefix,
        )

    @try_export
    def export_ncnn(self, prefix=colorstr("NCNN:")):
        """Export YOLO model to NCNN format using PNNX https://github.com/pnnx/pnnx."""
        from vision_ai_platform.packages.utils.export.ncnn import torch2ncnn

        model = NormalizedExportWrapper(self.model)
        return torch2ncnn(
            model=model,
            im=self.im,
            output_dir=str(self.file).replace(self.file.suffix, "_ncnn_model/"),
            quantize=self.cfg.quantize,
            metadata=self.metadata,
            device=self.device,
            prefix=prefix,
        )

    @try_export
    def export_coreml(self, prefix=colorstr("CoreML:")):
        """Export YOLO model to CoreML format."""
        mlmodel = self.cfg.format.lower() == "mlmodel"  # legacy *.mlmodel export format requested
        from vision_ai_platform.packages.utils.export.coreml import IOSDetectModel, pipeline_coreml, torch2coreml

        # numpy 2.4.x breaks coremltools CoreML export https://github.com/apple/coremltools/issues/2633
        check_requirements(["coremltools>=9.0", "numpy>=1.14.5,<=2.3.5"])
        import coremltools as ct

        assert not WINDOWS, "CoreML export is not supported on Windows, please run on macOS or Linux."
        assert TORCH_1_11, "CoreML export requires torch>=1.11"
        f = self.file.with_suffix(".mlmodel" if mlmodel else ".mlpackage")
        if f.is_dir():
            shutil.rmtree(f)

        # TODO CoreML Segment and Pose model pipelining; 'nms=True' is forced off for non-detect tasks upstream
        model = IOSDetectModel(self.model, self.im, mlprogram=not mlmodel) if self.cfg.nms else self.model

        if self.cfg.dynamic:
            h, w = self.imgsz
            lb_h = lb_w = 32
            if getattr(self.model, "end2end", False):
                # end2end graphs bake TopK k=max_det, so the smallest declared input must still supply >= k anchors
                # or CoreML rejects the model at load; shrink the range proportionally from the traced default size
                stride = int(self.model.stride.max())
                r = self.model.model[-1].max_det / sum(int(h / s) * int(w / s) for s in self.model.stride.tolist())
                lb_h = max(lb_h, int(np.ceil(h * r**0.5 / stride)) * stride)
                lb_w = max(lb_w, int(np.ceil(w * r**0.5 / stride)) * stride)
            input_shape = ct.Shape(
                shape=(
                    ct.RangeDim(lower_bound=1, upper_bound=self.cfg.batch, default=1),
                    self.im.shape[1],
                    ct.RangeDim(lower_bound=lb_h, upper_bound=h * 2, default=h),
                    ct.RangeDim(lower_bound=lb_w, upper_bound=w * 2, default=w),
                )
            )
            inputs = [ct.TensorType("image", shape=input_shape)]
        else:
            inputs = [ct.ImageType("image", shape=self.im.shape, scale=1 / 255, bias=[0.0, 0.0, 0.0])]

        quantize = 16 if self.cfg.nms and not mlmodel and self.cfg.quantize is None else self.cfg.quantize
        self.metadata["args"]["quantize"] = quantize
        ct_model = torch2coreml(
            model=model,
            inputs=inputs,
            im=self.im,
            classifier_names=list(self.model.names.values()) if self.model.task == "classify" else None,
            mlmodel=mlmodel,
            quantize=quantize,
            metadata=self.metadata,
            prefix=prefix,
        )

        if self.cfg.nms:
            ct_model = pipeline_coreml(
                ct_model,
                self.output_shape,
                weights_dir=None if mlmodel else ct_model.weights_dir,
                metadata=self.metadata,
                mlmodel=mlmodel,
                iou=self.cfg.iou,
                conf=self.cfg.conf,
                agnostic_nms=self.cfg.agnostic_nms,
                prefix=prefix,
            )

        if self.model.task == "classify":
            ct_model.user_defined_metadata.update({"com.apple.coreml.model.preview.type": "imageClassifier"})

        try:
            ct_model.save(str(f))  # save *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )
            f = f.with_suffix(".mlmodel")
            ct_model.save(str(f))
        return f

    @try_export
    def export_engine(self, prefix=colorstr("TensorRT")):
        """Export YOLO model to TensorRT format https://developer.nvidia.com/tensorrt."""
        assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx = self.export_onnx()  # run before TRT import https://github.com/ultralytics/ultralytics/issues/7016
        from vision_ai_platform.packages.utils.export.engine import onnx2engine

        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = self.file.with_suffix(".engine")  # TensorRT engine file
        onnx2engine(
            f_onnx,
            f,
            self.cfg.workspace,
            self.cfg.quantize,
            self.cfg.dynamic,
            self.im.shape,
            dla=self.dla,
            dataset=self.get_int8_calibration_dataloader(prefix) if self.cfg.quantize == 8 else None,
            metadata=self.metadata,
            verbose=self.cfg.verbose,
            prefix=prefix,
        )

        return f

    @try_export
    def export_saved_model(self, prefix=colorstr("TensorFlow SavedModel")):
        """Export YOLO model to TensorFlow SavedModel format."""
        assert not (MACOS and IS_PYTHON_MINIMUM_3_13), (
            "TensorFlow exports not supported on macOS with Python>=3.13: the ai-edge-litert macOS wheel fails to load "
            "(missing libpywrap_litert_common.dylib). TensorFlow export works on Linux Python 3.13."
        )
        from vision_ai_platform.packages.utils.export.tensorflow import onnx2saved_model

        f = Path(str(self.file).replace(self.file.suffix, "_saved_model"))
        if f.is_dir():
            shutil.rmtree(f)  # delete output folder

        # Export to TF
        images = None
        if self.cfg.quantize == 8 and self.cfg.data:
            images = [batch["img"] for batch in self.get_int8_calibration_dataloader(prefix)]
            images = (
                torch.nn.functional.interpolate(torch.cat(images, 0).float(), size=self.imgsz)
                .permute(0, 2, 3, 1)
                .numpy()
                .astype(np.float32)
            )

        # Export to ONNX
        if isinstance(self.model.model[-1], RTDETRDecoder):
            self.cfg.opset = self.cfg.opset or 19
            assert self.cfg.opset <= 19, "RTDETR TensorFlow export requires opset<=19"
        self.cfg.simplify = True
        f_onnx = self.export_onnx()  # ensure ONNX is available
        keras_model = onnx2saved_model(
            f_onnx,
            f,
            quantize=self.cfg.quantize,
            images=images,
            disable_group_convolution=self.cfg.format == "edgetpu",
            cuda=self.device.type == "cuda",
            prefix=prefix,
        )
        YAML.save(f / "metadata.yaml", self.metadata)  # add metadata.yaml
        # Add TFLite metadata
        for file in f.rglob("*.tflite"):
            file.unlink() if "quant_with_int16_act.tflite" in str(file) else self._add_tflite_metadata(file)

        return str(f), keras_model  # or keras_model = tf.saved_model.load(f, tags=None, options=None)

    @try_export
    def export_pb(self, keras_model, prefix=colorstr("TensorFlow GraphDef")):
        """Export YOLO model to TensorFlow GraphDef *.pb format https://github.com/leimao/Frozen-Graph-TensorFlow."""
        from vision_ai_platform.packages.utils.export.tensorflow import keras2pb

        return keras2pb(keras_model, output_file=self.file.with_suffix(".pb"), prefix=prefix)

    @try_export
    def export_edgetpu(self, tflite_model="", prefix=colorstr("Edge TPU")):
        """Export YOLO model to Edge TPU format https://coral.ai/docs/edgetpu/models-intro/."""
        from vision_ai_platform.packages.utils.export.tensorflow import tflite2edgetpu

        output_file = tflite2edgetpu(tflite_file=tflite_model, output_dir=tflite_model.parent, prefix=prefix)
        self._add_tflite_metadata(output_file)
        return output_file

    @try_export
    def export_deepx(self, prefix=colorstr("DEEPX:")):
        """Export YOLO model to DEEPX format."""
        assert LINUX and not ARM64, "DEEPX export only supported on non-aarch64 Linux"
        from vision_ai_platform.packages.utils.export.deepx import onnx2deepx

        f = self.export_onnx()
        return onnx2deepx(
            onnx_file=f,
            imgsz=self.imgsz,
            dataset=self.get_int8_calibration_dataloader(prefix),
            metadata=self.metadata,
            optimize=self.cfg.optimize,
            prefix=prefix,
        )


class NMSModel(torch.nn.Module):
    """Model wrapper with embedded NMS for Detect, Segment, Pose and OBB."""

    def __init__(self, model: nn.Module, cfg: YOLOConfig):
        """Initialize the NMSModel.

        Args:
            model (torch.nn.Module): The model to wrap with NMS postprocessing.
            cfg (YOLOConfig): The export arguments.
        """
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.obb = model.task == "obb"
        self.is_tf = self.cfg.format == "saved_model"

    def forward(self, x):
        """Perform inference with NMS post-processing. Supports Detect, Segment, OBB and Pose.

        Args:
            x (torch.Tensor): The preprocessed tensor with shape (B, C, H, W).

        Returns:
            (torch.Tensor | tuple): Tensor of shape (B, max_det, 4 + 2 + extra_shape) where B is the batch size, or a
                tuple of (detections, proto) for segmentation models.
        """
        from torchvision.ops import nms

        preds = self.model(x)
        pred = preds[0] if isinstance(preds, tuple) else preds
        kwargs = dict(device=pred.device, dtype=pred.dtype)
        bs = pred.shape[0]
        pred = pred.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        extra_shape = pred.shape[-1] - (4 + len(self.model.names))  # extras from Segment, OBB, Pose
        if self.cfg.dynamic and self.cfg.batch > 1:  # batch size needs to always be same due to loop unroll
            pad = torch.zeros(torch.max(torch.tensor(self.cfg.batch - bs), torch.tensor(0)), *pred.shape[1:], **kwargs)
            pred = torch.cat((pred, pad))
        boxes, scores, extras = pred.split([4, len(self.model.names), extra_shape], dim=2)
        scores, classes = scores.max(dim=-1)
        self.cfg.max_det = min(pred.shape[1], self.cfg.max_det)  # in case num_anchors < max_det
        # (N, max_det, 4 coords + 1 class score + 1 class label + extra_shape).
        out = torch.zeros(pred.shape[0], self.cfg.max_det, boxes.shape[-1] + 2 + extra_shape, **kwargs)
        for i in range(bs):
            box, cls, score, extra = boxes[i], classes[i], scores[i], extras[i]
            mask = score > self.cfg.conf
            if self.is_tf or (self.cfg.format == "onnx" and self.obb):
                # TFLite GatherND error if mask is empty
                score *= mask
                # Explicit length otherwise reshape error, hardcoded to `self.cfg.max_det * 5`
                mask = score.topk(min(self.cfg.max_det * 5, score.shape[0])).indices
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]
            nmsbox = box.clone()
            # `8` is the minimum value experimented to get correct NMS results for obb
            multiplier = 8 if self.obb else 1 / max(len(self.model.names), 1)
            # Normalize boxes for NMS since large values for class offset causes issue with int8 quantization
            nmsbox = multiplier * (nmsbox / torch.tensor(x.shape[2:], **kwargs).max())
            if not self.cfg.agnostic_nms:  # class-wise NMS
                end = 2 if self.obb else 4
                # fully explicit expansion otherwise reshape error
                cls_offset = cls.view(cls.shape[0], 1).expand(cls.shape[0], end)
                offbox = nmsbox[:, :end] + cls_offset * multiplier
                nmsbox = torch.cat((offbox, nmsbox[:, end:]), dim=-1)
            nms_fn = (
                partial(
                    TorchNMS.fast_nms,
                    use_triu=not (
                        self.is_tf
                        or (self.cfg.opset or 14) < 14
                        or (self.cfg.format == "openvino" and self.cfg.quantize == 8)  # OpenVINO INT8 error with triu
                    ),
                    iou_func=batch_probiou,
                    exit_early=False,
                )
                if self.obb
                else nms
            )
            keep = nms_fn(
                torch.cat([nmsbox, extra], dim=-1) if self.obb else nmsbox,
                score,
                self.cfg.iou,
            )[: self.cfg.max_det]
            dets = torch.cat(
                [box[keep], score[keep].view(-1, 1), cls[keep].view(-1, 1).to(out.dtype), extra[keep]], dim=-1
            )
            # Zero-pad to max_det size to avoid reshape error
            pad = (0, 0, 0, self.cfg.max_det - dets.shape[0])
            out[i] = torch.nn.functional.pad(dets, pad)
        return (out[:bs], preds[1]) if self.model.task == "segment" else out[:bs]
