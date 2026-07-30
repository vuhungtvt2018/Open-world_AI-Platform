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
    ]
    return dict(zip(["Format", "Argument", "Suffix", "CPU", "GPU", "Arguments", "Env"], zip(*x)))


def prepare_dummy_input(
    imgsz: List[int], batch: int = 1, device: str = "cpu"
) -> torch.Tensor:
    """Generate a dummy input tensor for model tracing and export."""
    if len(imgsz) == 1:
        imgsz = [imgsz[0], imgsz[0]]
    shape = (batch, 3, imgsz[0], imgsz[1])
    return torch.zeros(shape, device=torch.device(device))


def parse_imgsz(imgsz: Union[int, List[int], Tuple[int, ...]]) -> List[int]:
    """Parse image size into a 2-element list [height, width]."""
    if isinstance(imgsz, int):
        return [imgsz, imgsz]
    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            return [imgsz[0], imgsz[0]]
        return list(imgsz[:2])
    return [640, 640]


class Exporter:
    """Class responsible for handling export workflows across multiple formats.

    Args:
        cfg: Configuration object or path/dict to construct ExporterConfig.
        overrides: Keyword arguments overriding the base configuration.
        _callbacks: Optional dictionary of registered callback functions.
    """

    def __init__(
        self,
        cfg: Optional[Union[ExporterConfig, Dict[str, Any]]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        _callbacks: Optional[Dict[str, list]] = None,
    ):
        if isinstance(cfg, ExporterConfig):
            self.args = cfg
        elif isinstance(cfg, dict):
            self.args = ExporterConfig.from_dict(cfg)
        else:
            self.args = ExporterConfig()

        if overrides:
            for k, v in overrides.items():
                if hasattr(self.args, k):
                    setattr(self.args, k, v)

        self.callbacks = _callbacks or {}
        self.im: Optional[torch.Tensor] = None
        self.model: Optional[nn.Module] = None
        self.file: Optional[Path] = None
        self.device = torch.device(self.args.device)

    def add_callback(self, event: str, callback: callable) -> None:
        """Register a callback function for export events."""
        self.callbacks.setdefault(event, []).append(callback)

    def run_callbacks(self, event: str) -> None:
        """Execute registered callbacks for an event tag."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare torch model for export (eval mode, precision conversion)."""
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad = False

        if self.args.half and self.device.type != "cpu":
            model = model.half()
        return model

    def __call__(
        self,
        model: nn.Module,
        file_path: Union[str, Path] = "yolo_model.pt",
    ) -> str:
        """Executes the model export and returns the saved file path."""
        t_start = time.time()
        self.file = Path(file_path)
        self.model = self._prepare_model(model)
        
        # Prepare input dummy tensor matching precision and device
        self.im = prepare_dummy_input(
            self.args.imgsz, batch=self.args.batch, device=self.args.device
        )
        if self.args.half and self.device.type != "cpu":
            self.im = self.im.half()

        fmt = self.args.format
        self.run_callbacks("on_export_start")

        # Route export execution based on target format
        export_fn_map = {
            "torchscript": self.export_torchscript,
            "onnx": self.export_onnx,
            "openvino": self.export_openvino,
            "engine": self.export_engine,
        }

        if fmt in export_fn_map:
            output_file = export_fn_map[fmt]()
        else:
            raise NotImplementedError(f"Exporter function for '{fmt}' is not implemented yet.")

        self.run_callbacks("on_export_end")
        print(f"Export completed in {time.time() - t_start:.2f}s. Saved to: {output_file}")
        return str(output_file)

    # --- Format Exporter Implementations ---

    def export_torchscript(self) -> Path:
        """Export PyTorch model to TorchScript tracing format."""
        output_path = self.file.with_suffix(".torchscript")
        ts_model = torch.jit.trace(self.model, self.im, strict=False)
        ts_model.save(str(output_path))
        return output_path

    def export_onnx(self) -> Path:
        """Export PyTorch model to ONNX format."""
        output_path = self.file.with_suffix(".onnx")
        dynamic_axes = None
        if self.args.dynamic:
            dynamic_axes = {
                "images": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch"},
            }

        torch.onnx.export(
            self.model,
            self.im,
            str(output_path),
            verbose=False,
            opset_version=self.args.opset or 17,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        if self.args.simplify:
            try:
                import onnx
                import onnxslim

                onnx_model = onnx.load(str(output_path))
                slimmed = onnxslim.slim(onnx_model)
                onnx.save(slimmed, str(output_path))
            except ImportError:
                print("`onnxslim` / `onnx` not installed; skipping graph simplification.")

        return output_path

    def export_openvino(self) -> Path:
        """Export model to OpenVINO IR format."""
        try:
            import openvino as ov
        except ImportError as err:
            raise ImportError("OpenVINO is required for this export. Run `pip install openvino`.") from err

        onnx_path = self.export_onnx()
        output_dir = self.file.with_suffix("_openvino_model")
        ov_model = ov.convert_model(onnx_path)
        ov.save_model(ov_model, str(output_dir / self.file.with_suffix(".xml").name))
        return output_dir

    def export_engine(self) -> Path:
        """Export model to TensorRT engine format."""
        try:
            import tensorrt as trt
        except ImportError as err:
            raise ImportError("TensorRT SDK is required for '.engine' exports.") from err

        # Requires generating intermediate ONNX file
        onnx_path = self.export_onnx()
        output_path = self.file.with_suffix(".engine")

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()

        if self.args.workspace:
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, int(self.args.workspace * (1024**3))
            )

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError(f"Failed to parse ONNX graph: {parser.get_error(0)}")

        plan = builder.build_serialized_network(network, config)
        with open(output_path, "wb") as f:
            f.write(plan)

        return output_path