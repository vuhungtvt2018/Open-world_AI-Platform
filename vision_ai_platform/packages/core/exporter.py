from __future__ import annotations

import json
import os
import shutil
import time
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Union, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from .config import ExporterConfig

def export_formats():
    """Return a dictionary of Ultralytics YOLO export formats."""
    x = [
        ["PyTorch", "-", ".pt", True, True, [], "base"],
        [
            "TorchScript",
            "torchscript",
            ".torchscript",
            True,
            True,
            ["batch", "quantize", "nms", "dynamic"],
            "base",
        ],
        [
            "ONNX",
            "onnx",
            ".onnx",
            True,
            True,
            ["batch", "data", "dynamic", "quantize", "opset", "simplify", "nms", "fraction"],
            "base",
        ],
        [
            "OpenVINO",
            "openvino",
            "_openvino_model",
            True,
            False,
            ["batch", "data", "dynamic", "quantize", "nms", "fraction"],
            "base",
        ],
        [
            "TensorRT",
            "engine",
            ".engine",
            False,
            True,
            ["batch", "data", "dynamic", "quantize", "simplify", "nms", "fraction"],
            "base",
        ],
        ["CoreML", "coreml", ".mlpackage", True, False, ["batch", "dynamic", "quantize", "nms"], "coreml"],
        [
            "TensorFlow SavedModel",
            "saved_model",
            "_saved_model",
            True,
            True,
            ["batch", "data", "fraction", "quantize", "keras", "nms"],
            "tensorflow",
        ],
        ["TensorFlow GraphDef", "pb", ".pb", True, True, ["batch"], "tensorflow"],
        [
            "TensorFlow Edge TPU",
            "edgetpu",
            "_edgetpu.tflite",
            True,
            False,
            ["data", "fraction", "quantize"],
            "tensorflow",
        ],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True, ["batch"], "base"],
        ["NCNN", "ncnn", "_ncnn_model", True, True, ["batch", "quantize"], "ncnn"],
        ["LiteRT", "litert", ".tflite", True, False, ["batch", "quantize", "data", "fraction"], "litert"],
        ["MNN", "mnn", ".mnn", True, True, ["batch", "dynamic", "quantize", "opset", "simplify", "nms"], "mnn"],
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments", "Env"], zip(*x)))


# Export precision support per format. Unset/32 requests are FP32 except for formats listed in FP32_UNSUPPORTED_FORMATS.
FP16_FORMATS = frozenset({"torchscript", "onnx", "openvino", "engine", "coreml", "mnn", "ncnn", "rknn"})
INT8_FORMATS = frozenset(
    {
        "onnx",
        "openvino",
        "engine",
        "coreml",
        "saved_model",
        "edgetpu",
        "mnn",
        "imx",
        "rknn",
        "axelera",
        "deepx",
        "hailo",
        "litert",
    }
)
W8A16_FORMATS = frozenset(
    {"coreml", "imx", "qnn", "litert"}
)  # INT8 weights + 16-bit activations (FP16; INT16 on LiteRT)
W8A32_FORMATS = frozenset({"litert"})  # INT8 weights + FP32 activations (dynamic/weight-only INT8, no calibration)
FP32_UNSUPPORTED_FORMATS = frozenset({"edgetpu", "imx", "rknn", "axelera", "deepx", "qnn", "hailo"})
# (label, supporting formats) per quantize precision, used to list valid options in errors. 32/None (FP32) is universal except FP32_UNSUPPORTED_FORMATS.
QUANTIZE_PRECISIONS = (
    ("16 (FP16)", FP16_FORMATS),
    ("8 (INT8)", INT8_FORMATS),
    ("'w8a16' (INT8 weights + INT16 activations)", W8A16_FORMATS),
    ("'w8a32' (dynamic INT8)", W8A32_FORMATS),
)

# Quantization aliases: maps any accepted `quantize` value to its canonical form. The common case is the integer
# bit-width 8 (INT8) / 16 (FP16) / 32 (FP32); the verbose w<weights>a<activations> strings are accepted too and
# collapse to the same ints, except the mixed-precision schemes that have no bit-width shorthand: 'w8a16' (INT8
# weights + FP16 activations) and 'w8a32' (INT8 weights + FP32 activations, i.e. LiteRT dynamic/weight-only INT8).
QUANTIZE_ALIASES = {
    "8": 8,
    "16": 16,
    "32": 32,
    "int8": 8,
    "fp16": 16,
    "fp32": 32,
    "w8a8": 8,
    "w16a16": 16,
    "w32a32": 32,
    "w8a16": "w8a16",
    "w8a32": "w8a32",
}
QUANTIZE_DOCS_URL = "https://docs.ultralytics.com/modes/export/#quantization-options"
QUANTIZE_VALID_VALUES = "8, 16, 32, 'int8', 'fp16', 'fp32', 'w8a8', 'w16a16', 'w8a16', or 'w8a32'"


class BaseExporter(ABC):
    """A class for exporting YOLO models to various formats.

    This class provides functionality to export YOLO models to different formats including ONNX, TensorRT, CoreML,
    TensorFlow, and others. It handles format validation, device selection, model preparation, and the actual export
    process for each supported format.

    Attributes:
        cfg (ExporterConfig): Configuration arguments for the exporter.
        callbacks (dict): Dictionary of callback functions for different export events.
        im (torch.Tensor): Input tensor for model inference during export.
        model (torch.nn.Module): The YOLO model to be exported.
        file (Path): Path to the model file being exported.
        output_shape (tuple): Shape of the model output tensor(s).
        pretty_name (str): Formatted model name for display purposes.
        metadata (dict): Model metadata including description, author, version, etc.
        device (torch.device): Device on which the model is loaded.
        imgsz (list): Input image size for the model.

    Methods:
        __call__: Main export method that handles the export process.
        get_int8_calibration_dataloader: Build dataloader for INT8 calibration.
        export_torchscript: Export model to TorchScript format.
        export_onnx: Export model to ONNX format.
        export_openvino: Export model to OpenVINO format.
        export_paddle: Export model to PaddlePaddle format.
        export_mnn: Export model to MNN format.
        export_ncnn: Export model to NCNN format.
        export_coreml: Export model to CoreML format.
        export_engine: Export model to TensorRT format.
        export_saved_model: Export model to TensorFlow SavedModel format.
        export_pb: Export model to TensorFlow GraphDef format.
        export_edgetpu: Export model to Edge TPU format.
        export_deepx: Export model to DEEPX format.

    Examples:
        Export a YOLO26 model to TorchScript format
        >>> from ultralytics.engine.exporter import Exporter
        >>> exporter = Exporter()
        >>> exporter(model="yolo26n.pt")  # exports to yolo26n.torchscript

        Export with specific arguments
        >>> args = {"format": "onnx", "dynamic": True, "quantize": 8, "data": "coco8.yaml"}
        >>> exporter = Exporter(overrides=args)
        >>> exporter(model="yolo26n.pt")
    """

    def __init__(self, cfg: ExporterConfig, _callbacks: dict | None = None):
        """Initialize the Exporter class.

        Args:
            cfg (ExporterConfig): Configuration object.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.cfg = cfg
        self.callbacks = None

    @abstractmethod
    def __call__(self, model=None) -> str:
        """Export a model and return the final exported path as a string.

        Returns:
            (str): Path to the exported file or directory (the last export artifact).
        """
        raise NotImplementedError()

    @abstractmethod
    def export_torchscript(self, prefix="TorchScript:"):
        """Export YOLO model to TorchScript format."""
        raise NotImplementedError()

    @abstractmethod
    def export_onnx(self, prefix="ONNX:"):
        """Export YOLO model to ONNX format."""
        raise NotImplementedError()

    @abstractmethod
    def export_openvino(self, prefix="OpenVINO:"):
        """Export YOLO model to OpenVINO format."""
        raise NotImplementedError()

    @abstractmethod
    def export_paddle(self, prefix="PaddlePaddle:"):
        """Export YOLO model to PaddlePaddle format."""
        raise NotImplementedError()

    @abstractmethod
    def export_litert(self, prefix="LiteRT:"):
        """Export YOLO model to LiteRT format using litert_torch with optional INT8 quantization.

        Supports ``quantize=8`` (static INT8, int8 weights + int8 activations, requires calibration ``data``),
        ``quantize='w8a16'`` (static, int8 weights + int16 activations, requires calibration ``data``) and
        ``quantize='w8a32'`` (dynamic/weight-only INT8, int8 weights + FP32 activations, no calibration needed).
        """
        raise NotImplementedError()

    @abstractmethod
    def export_mnn(self, prefix="MNN:"):
        """Export YOLO model to MNN format using MNN https://github.com/alibaba/MNN."""
        raise NotImplementedError()

    @abstractmethod
    def export_ncnn(self, prefix="NCNN:"):
        """Export YOLO model to NCNN format using PNNX https://github.com/pnnx/pnnx."""
        raise NotImplementedError()

    @abstractmethod
    def export_coreml(self, prefix="CoreML:"):
        """Export YOLO model to CoreML format."""
        raise NotImplementedError()

    @abstractmethod
    def export_engine(self, prefix="TensorRT"):
        """Export YOLO model to TensorRT format https://developer.nvidia.com/tensorrt."""
        raise NotImplementedError()

    @abstractmethod
    def export_saved_model(self, prefix="TensorFlow SavedModel"):
        """Export YOLO model to TensorFlow SavedModel format."""
        raise NotImplementedError()

    @abstractmethod
    def export_pb(self, keras_model, prefix="TensorFlow GraphDef"):
        """Export YOLO model to TensorFlow GraphDef *.pb format https://github.com/leimao/Frozen-Graph-TensorFlow."""
        raise NotImplementedError()

    @abstractmethod
    def export_edgetpu(self, tflite_model="", prefix="Edge TPU"):
        """Export YOLO model to Edge TPU format https://coral.ai/docs/edgetpu/models-intro/."""
        raise NotImplementedError()

    @abstractmethod
    def export_deepx(self, prefix="DEEPX:"):
        """Export YOLO model to DEEPX format."""
        raise NotImplementedError()

    def _add_tflite_metadata(self, file):
        """Add metadata to *.tflite models per https://ai.google.dev/edge/litert/models/metadata."""
        import zipfile

        with zipfile.ZipFile(file, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(self.metadata, indent=2))

    @staticmethod
    def _transform_fn(data_item) -> np.ndarray:
        """Quantization preprocessing transform for INT8 calibration (Axelera, OpenVINO, ONNX, QNN)."""
        data_item: torch.Tensor = data_item["img"] if isinstance(data_item, dict) else data_item
        assert data_item.dtype == torch.uint8, "Input image must be uint8 for the quantization preprocessing"
        im = data_item.numpy().astype(np.float32) / 255.0  # uint8 to fp16/32 and 0 - 255 to 0.0 - 1.0
        return im[None] if im.ndim == 3 else im

    def add_callback(self, event: str, callback):
        """Append the given callback to the specified event."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Execute all callbacks for a given event."""
        for callback in self.callbacks.get(event, []):
            callback(self)


class ExportWrapper(torch.nn.Module):
    """Base for export-time model wrappers: stores the wrapped model and forwards attribute lookups.

    Subclasses adapt a fused YOLO model's inference I/O for a specific deployment contract (layout, output
    reduction) while the exporter keeps interacting with the wrapper as if it were the model itself.
    """

    def __init__(self, model):
        """Wrap a fused YOLO `model` prepared for export."""
        super().__init__()
        # Stored under a private name so attribute forwarding resolves `wrapper.model` to the wrapped model's own
        # `model` (its nn.Sequential), keeping exporter code like `self.model.model[-1]` working unchanged.
        self._model = model
        self.task = model.task

    def __getattr__(self, name):
        """Forward attribute lookups (model, names, stride, yaml, args, ...) to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


class NormalizedExportWrapper(ExportWrapper):
    """Normalize YOLO-style outputs for X2Paddle tracing.

    Some YOLO variants return dictionaries or nested lists containing tensors, for example
    ``{"boxes": tensor, "feats": [tensor, tensor]}``. X2Paddle can choke on these mixed
    output structures during tracing. Flattening them to a simple tuple of tensors keeps the
    export path compatible while preserving the values needed for export.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.names = getattr(model, "names", None)
        self.stride = getattr(model, "stride", None)
        self.yaml = getattr(model, "yaml", None)
        self.args = getattr(model, "args", None)
        self.pt_path = getattr(model, "pt_path", None)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _flatten_outputs(value):
        flat = []

        def visit(item):
            if isinstance(item, torch.Tensor):
                flat.append(item)
            elif isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)

        visit(value)
        return tuple(flat)

    def forward(self, x: torch.Tensor):
        output = self._model(x)
        flat_output = self._flatten_outputs(output)
        return flat_output if flat_output else (x,)


class ClassMapModel(ExportWrapper):
    """Reduces semantic-segmentation logits to a compact integer class map for export.

    Applied to QNN and Core ML semantic exports, where the argmax runs on the NPU: deployment consumers want per-pixel
    class indices, and shipping float logits instead forces a dequantize + argmax over large tensors (~20M values at
    1024px) on the consumer's CPU every frame - measured as both slow and highly variable on mobile
    NPUs. The argmax cannot live in the model's own forward because it is non-differentiable (training needs
    logits), so it is attached here at export time, mirroring how `NMSModel` adds suppression only for export.

    Attributes:
        task (str): The wrapped model's task ("semantic").
        dtype (torch.dtype): Class-index dtype; uint8 unless the model has more than 256 classes.
    """

    def __init__(self, model):
        """Wrap a fused semantic `model` so export emits class indices instead of logits."""
        super().__init__(model)
        # uint8 quarters the NPU->CPU output transfer vs int32 and Core ML promotes it to int32 in-spec;
        # int32 only when more than 256 classes make uint8 indices ambiguous.
        self.dtype = torch.uint8 if len(model.names) <= 256 else torch.int32

    def forward(self, x):
        """Run the wrapped model and return a `[N, H, W]` integer class map instead of float logits."""
        y = self._model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        # Single-channel (binary) models threshold the logit, matching predict/val semantics for nc == 1
        return (y.argmax(1) if y.shape[1] > 1 else y[:, 0].gt(0)).to(self.dtype)