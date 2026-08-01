from pathlib import Path
import time
from copy import deepcopy
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn

from vision_ai_platform import __version__
from vision_ai_platform.packages.core import ExporterConfig, BaseExporter
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
)
from vision_ai_platform.packages.utils import (
    LOGGER,
    callbacks,
    TASK2CALIBRATIONDATA,
    LINUX,
    ARM64,
    is_jetson,
    DEFAULT_CFG,
    TASK2DATA,
    SETTINGS,
    colorstr,
    get_default_args
)
from vision_ai_platform.packages.utils.ops import Profile
from vision_ai_platform.packages.utils.check import check_version, check_imgsz, is_intel
from vision_ai_platform.packages.utils.device_utils import (
    select_device,
    TORCH_1_13,
)
from vision_ai_platform.packages.utils.nms import TorchNMS
from vision_ai_platform.packages.utils.loss import batch_probiou
from vision_ai_platform.packages.utils.files import file_size
from vision_ai_platform.packages.ai.data.utils import check_class_names
from vision_ai_platform.packages.ai.nn.modules import (
    Detect, Segment, Segment26, RTDETRDecoder, Classify, C2f, SemanticSegment
)
from vision_ai_platform.packages.ai.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, WorldModel
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
    def __init__(self, cfg: ExporterConfig, _callbacks: dict | None = None):
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


class NMSModel(torch.nn.Module):
    """Model wrapper with embedded NMS for Detect, Segment, Pose and OBB."""

    def __init__(self, model: nn.Module, args: ExporterConfig):
        """Initialize the NMSModel.

        Args:
            model (torch.nn.Module): The model to wrap with NMS postprocessing.
            args (ExporterConfig): The export arguments.
        """
        super().__init__()
        self.model = model
        self.args = args
        self.obb = model.task == "obb"
        self.is_tf = self.args.format == "saved_model"

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
        if self.args.dynamic and self.args.batch > 1:  # batch size needs to always be same due to loop unroll
            pad = torch.zeros(torch.max(torch.tensor(self.args.batch - bs), torch.tensor(0)), *pred.shape[1:], **kwargs)
            pred = torch.cat((pred, pad))
        boxes, scores, extras = pred.split([4, len(self.model.names), extra_shape], dim=2)
        scores, classes = scores.max(dim=-1)
        self.args.max_det = min(pred.shape[1], self.args.max_det)  # in case num_anchors < max_det
        # (N, max_det, 4 coords + 1 class score + 1 class label + extra_shape).
        out = torch.zeros(pred.shape[0], self.args.max_det, boxes.shape[-1] + 2 + extra_shape, **kwargs)
        for i in range(bs):
            box, cls, score, extra = boxes[i], classes[i], scores[i], extras[i]
            mask = score > self.args.conf
            if self.is_tf or (self.args.format == "onnx" and self.obb):
                # TFLite GatherND error if mask is empty
                score *= mask
                # Explicit length otherwise reshape error, hardcoded to `self.args.max_det * 5`
                mask = score.topk(min(self.args.max_det * 5, score.shape[0])).indices
            box, score, cls, extra = box[mask], score[mask], cls[mask], extra[mask]
            nmsbox = box.clone()
            # `8` is the minimum value experimented to get correct NMS results for obb
            multiplier = 8 if self.obb else 1 / max(len(self.model.names), 1)
            # Normalize boxes for NMS since large values for class offset causes issue with int8 quantization
            nmsbox = multiplier * (nmsbox / torch.tensor(x.shape[2:], **kwargs).max())
            if not self.args.agnostic_nms:  # class-wise NMS
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
                        or (self.args.opset or 14) < 14
                        or (self.args.format == "openvino" and self.args.quantize == 8)  # OpenVINO INT8 error with triu
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
                self.args.iou,
            )[: self.args.max_det]
            dets = torch.cat(
                [box[keep], score[keep].view(-1, 1), cls[keep].view(-1, 1).to(out.dtype), extra[keep]], dim=-1
            )
            # Zero-pad to max_det size to avoid reshape error
            pad = (0, 0, 0, self.args.max_det - dets.shape[0])
            out[i] = torch.nn.functional.pad(dets, pad)
        return (out[:bs], preds[1]) if self.model.task == "segment" else out[:bs]
