from __future__ import annotations

import inspect
import os
from collections.abc import Iterator
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .results import BaseResults

class BaseModel(nn.Module, ABC):
    """A base class for implementing YOLO models, unifying APIs across different model types.

    This class provides a common interface for various operations related to YOLO models, such as training, validation,
    prediction, exporting, and benchmarking. It handles different types of models, including those loaded from local
    files, Ultralytics HUB, or Triton Server.

    Attributes:
        callbacks (dict): A dictionary of callback functions for various events during model operations.
        predictor (BasePredictor): The predictor object used for making predictions.
        model (torch.nn.Module): The underlying PyTorch model.
        trainer (BaseTrainer): The trainer object used for training the model.
        ckpt (dict): The checkpoint data if the model is loaded from a *.pt file.
        cfg (str): The configuration of the model if loaded from a *.yaml file.
        ckpt_path (str): The path to the checkpoint file.
        overrides (dict): A dictionary of overrides for model configuration.
        metrics (ultralytics.utils.metrics.DetMetrics): The latest training/validation metrics.
        session (HUBTrainingSession): The Ultralytics HUB session, if applicable.
        task (str): The type of task the model is intended for.
        model_name (str): The name of the model.

    Methods:
        __call__: Alias for the predict method, enabling the model instance to be callable.
        _new: Initialize a new model based on a configuration file.
        _load: Load a model from a checkpoint file.
        _check_is_pytorch_model: Ensure that the model is a PyTorch model.
        reset_weights: Reset the model's weights to their initial state.
        load: Load model weights from a specified file.
        save: Save the current state of the model to a file.
        info: Log or return information about the model.
        fuse: Fuse Conv2d and BatchNorm2d layers for optimized inference.
        predict: Perform predictions on given image sources.
        track: Perform object tracking.
        val: Validate the model on a dataset.
        benchmark: Benchmark the model on various export formats.
        export: Export the model to different formats.
        train: Train the model on a dataset.
        tune: Perform hyperparameter tuning.
        _apply: Apply a function to the model's tensors.
        add_callback: Add a callback function for an event.
        clear_callback: Clear all callbacks for an event.
        reset_callbacks: Reset all callbacks to their default functions.

    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> results = model.predict("image.jpg")
        >>> model.train(data="coco8.yaml", epochs=3)
        >>> metrics = model.val()
        >>> model.export(format="onnx")
    """

    def __init__(
        self,
        model: str | Path | BaseModel = "yolo26n.pt",
        task: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize a new instance of the YOLO model class.

        This constructor sets up the model based on the provided model path or name. It handles various types of model
        sources, including local files, Ultralytics HUB models, and Triton Server models. The method initializes several
        important attributes of the model and prepares it for operations like training, prediction, or export.

        Args:
            model (str | Path | Model): Path or name of the model to load or create. Can be a local file path, a model
                name from Ultralytics HUB, a Triton Server model, or an already initialized Model instance.
            task (str, optional): The specific task for the model. If None, it will be inferred from the config.
            verbose (bool): If True, enables verbose output during the model's initialization and subsequent operations.

        Raises:
            FileNotFoundError: If the specified model file does not exist or is inaccessible.
            ValueError: If the model file or configuration is invalid or unsupported.
            ImportError: If required dependencies for specific model types (like HUB SDK) are not installed.
        """
        if isinstance(model, BaseModel):
            self.__dict__ = model.__dict__  # accepts an already initialized Model
            return
        super().__init__()
        self.callbacks = None
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = {}  # if loaded from *.pt
        self.cfg_path = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        self.task = task  # task type
        self.model_name = None  # model name
        model = str(model).strip()

        # Load or create new YOLO model
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to avoid deterministic warnings
        if str(model).endswith((".yaml", ".yml")):
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

        # Delete super().training for accessing self.model.training
        del self.training

    def __call__(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Iterator[BaseResults | torch.Tensor] | list[BaseResults] | list[torch.Tensor]:
        """Alias for the predict method, enabling the model instance to be callable for predictions.

        This method simplifies the process of making predictions by allowing the model instance to be called directly
        with the required arguments.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | list | tuple): The source of the image(s)
                to make predictions on. Can be a file path, URL, PIL image, numpy array, PyTorch tensor, or a list/tuple
                of these.
            stream (bool): If True, treat the input source as a continuous stream for predictions.
            **kwargs (Any): Additional keyword arguments to configure the prediction process.

        Returns:
            (Iterator[vision_ai_platform.packages.ai.models.common.results.Results | torch.Tensor] | list[vision_ai_platform.packages.ai.models.common.results.Results] |
            list[torch.Tensor]): Prediction results or embeddings, streamed when `stream=True`.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> results = model("https://ultralytics.com/images/bus.jpg")
            >>> for r in results:
            ...     print(f"Detected {len(r)} objects in image")
        """
        return self.predict(source, stream, **kwargs)

    @abstractmethod
    def _new(self, cfg_path: str, task=None, model=None, verbose=False) -> None:
        """Initialize a new model and infer the task type from model definitions.

        Creates a new model instance based on the provided configuration file. Loads the model configuration, infers the
        task type if not specified, and initializes the model using the appropriate class from the task map.

        Args:
            cfg_path (str): Path to the model configuration file in YAML format.
            task (str, optional): The specific task for the model. If None, it will be inferred from the config.
            model (type[torch.nn.Module], optional): A custom model class. If provided, it will be used instead of the
                default model class from the task map.
            verbose (bool): If True, displays model information during loading.

        Raises:
            ValueError: If the configuration file is invalid or the task cannot be inferred.
            ImportError: If the required dependencies for the specified task are not installed.
        """
        pass

    @abstractmethod
    def _load(self, weights: str, task=None) -> None:
        """Load a model from a checkpoint file or initialize it from a weights file.

        This method handles loading models from either .pt checkpoint files or other weight file formats. It sets up the
        model, task, and related attributes based on the loaded weights.

        Args:
            weights (str): Path to the model weights file to be loaded.
            task (str, optional): The task associated with the model. If None, it will be inferred from the model.

        Raises:
            FileNotFoundError: If the specified weights file does not exist or is inaccessible.
            ValueError: If the weights file format is unsupported or invalid.
        """
        pass

    def _check_is_pytorch_model(self) -> None:
        """Check if the model is a PyTorch model and raise TypeError if it's not.

        This method verifies that the model is either a PyTorch module or a .pt file. It's used to ensure that certain
        operations that require a PyTorch model are only performed on compatible model types.

        Raises:
            TypeError: If the model is not a PyTorch module or a .pt file. The error message provides detailed
                information about supported model formats and operations.

        Examples:
            >>> model = Model("yolo26n.pt")
            >>> model._check_is_pytorch_model()  # No error raised
            >>> model = Model("yolo26n.onnx")
            >>> model._check_is_pytorch_model()  # Raises TypeError
        """
        pt_str = isinstance(self.model, (str, Path)) and str(self.model).rpartition(".")[-1] == "pt"
        pt_module = isinstance(self.model, torch.nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolo26n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> BaseModel:
        """Reset the model's weights to their initial state.

        This method iterates through all modules in the model and resets their parameters if they have a
        'reset_parameters' method. It also ensures that all parameters have 'requires_grad' set to True, enabling them
        to be updated during training.

        Returns:
            (Model): The instance of the class with reset weights.

        Raises:
            TypeError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo26n.pt")
            >>> model.reset_weights()
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    @abstractmethod
    def load(self, weights: str | Path = "yolo26n.pt") -> BaseModel:
        """Load parameters from the specified weights file into the model.

        This method supports loading weights from a file or directly from a weights object. It matches parameters by
        name and shape and transfers them to the model.

        Args:
            weights (str | Path): Path to the weights file or a weights object.

        Returns:
            (Model): The instance of the class with loaded weights.

        Raises:
            TypeError: If the model is not a PyTorch model.
        """
        pass

    def save(self, filename: str | Path = "saved_model.pt") -> None:
        """Save the current model state to a file.

        This method exports the model's checkpoint (ckpt) to the specified filename. It includes metadata such as the
        date, Ultralytics version, license information, and a link to the documentation.

        Args:
            filename (str | Path): The name of the file to save the model to.

        Raises:
            TypeError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo26n.pt")
            >>> model.save("my_model.pt")
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from vision_ai_platform import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
        }
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True, imgsz: int | list[int, int] = 640):
        """Display model information.

        This method provides an overview or detailed information about the model, depending on the arguments
        passed. It can control the verbosity of the output.

        Args:
            detailed (bool): If True, shows detailed information about the model layers and parameters.
            verbose (bool): If True, prints the information and returns model summary. If False, returns None.
            imgsz (int | list[int, int]): Input image size used for FLOPs calculation.

        Returns:
            (tuple): A tuple containing the number of layers (int), number of parameters (int), number of gradients
                (int), and GFLOPs (float). Returns None if verbose is False.

        Examples:
            >>> model = Model("yolo26n.pt")
            >>> model.info()  # Prints model summary and returns tuple
            >>> model.info(detailed=True)  # Prints detailed info and returns tuple
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose, imgsz=imgsz)

    def fuse(self) -> BaseModel:
        """Fuse Conv2d and BatchNorm2d layers in the model for optimized inference.

        This method iterates through the model's modules and fuses consecutive Conv2d and BatchNorm2d layers into a
        single layer. This fusion can significantly improve inference speed by reducing the number of operations and
        memory accesses required during forward passes.

        The fusion process typically involves folding the BatchNorm2d parameters (mean, variance, weight, and
        bias) into the preceding Conv2d layer's weights and biases. This results in a single Conv2d layer that
        performs both convolution and normalization in one step.

        Examples:
            >>> model = Model("yolo26n.pt")
            >>> model.fuse()
            >>> # Model is now fused and ready for optimized inference
        """
        self._check_is_pytorch_model()
        self.model.fuse()
        return self

    def embed(
        self,
        source: str | Path | int | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Iterator[torch.Tensor] | list[torch.Tensor]:
        """Generate image embeddings based on the provided source.

        This method is a wrapper around the 'predict()' method, returning feature embeddings from image sources. By
        default, embeddings are extracted from the second-to-last model layer. Pass `embed=[layer_index]` in `kwargs` to
        select specific layers.

        Args:
            source (str | Path | int | list | tuple | np.ndarray | torch.Tensor): The source of the image for generating
                embeddings. Can be a file path, URL, numpy array, etc.
            stream (bool): If True, predictions are streamed.
            **kwargs (Any): Additional keyword arguments for configuring the embedding process.

        Returns:
            (Iterator[torch.Tensor] | list[torch.Tensor]): Image embeddings, streamed when `stream=True`.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> image = "https://ultralytics.com/images/bus.jpg"
            >>> embeddings = model.embed(image)
            >>> results = model.predict(image)
            >>> print(embeddings[0].shape)
            >>> print(results[0].boxes.shape)
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)

    @abstractmethod
    def predict(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> Iterator[BaseResults | torch.Tensor] | list[BaseResults] | list[torch.Tensor]:
        """Perform predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments. It
        supports predictions with custom predictors or the default predictor method. The method handles different types
        of image sources and can operate in a streaming mode.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | list | tuple): The source of the image(s)
                to make predictions on. Accepts various types including file paths, URLs, PIL images, numpy arrays, and
                torch tensors.
            stream (bool): If True, treats the input source as a continuous stream for predictions.
            predictor (BasePredictor, optional): An instance of a custom predictor class for making predictions. If
                None, the method uses a default predictor.
            **kwargs (Any): Additional keyword arguments for configuring the prediction process. These include `embed`
                for returning feature embeddings from specified layers.

        Returns:
            (Iterator[vision_ai_platform.packages.ai.models.common.results.Results | torch.Tensor] | list[vision_ai_platform.packages.ai.models.common.results.Results] |
            list[torch.Tensor]): Prediction results or embeddings, streamed when `stream=True`.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> results = model.predict(source="path/to/image.jpg", conf=0.25)
            >>> for r in results:
            ...     print(r.boxes.data)  # print detection bounding boxes

        Notes:
            - If 'source' is not provided, it defaults to the ASSETS constant with a warning.
            - The method sets up a new predictor if not already present and updates its arguments with each call.
            - For SAM-type models, 'prompts' can be passed as a keyword argument.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def track(
        self,
        source: str | Path | int | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> list[BaseResults]:
        """Conduct object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It handles
        various input sources such as file paths or video streams, and supports customization through keyword arguments.
        The method registers trackers if not already present and can persist them between calls.

        Args:
            source (str | Path | int | list | tuple | np.ndarray | torch.Tensor, optional): Input source for object
                tracking. Can be a file path, URL, or video stream.
            stream (bool): If True, treats the input source as a continuous video stream.
            persist (bool): If True, persists trackers between different calls to this method.
            **kwargs (Any): Additional keyword arguments for configuring the tracking process.

        Returns:
            (list[vision_ai_platform.packages.ai.models.common.results.Results]): A list of tracking results, each a Results object.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # print tracking IDs

        Notes:
            - This method sets a default confidence threshold of 0.1 for ByteTrack-based tracking.
            - The tracking mode is explicitly set in the keyword arguments.
            - Batch size is set to 1 for tracking in videos.
        """
        raise NotImplementedError()

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        """Validate the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for customization through various settings. It
        supports validation with a custom validator or the default validation approach. The method combines default
        configurations, method-specific defaults, and user-provided arguments to configure the validation process.

        Args:
            validator (ultralytics.engine.validator.BaseValidator, optional): An instance of a custom validator class
                for validating the model.
            **kwargs (Any): Arbitrary keyword arguments for customizing the validation process.

        Returns:
            (ultralytics.utils.metrics.DetMetrics): Validation metrics obtained from the validation process. The
                specific metrics type depends on the task (e.g., DetMetrics, SegmentMetrics,
                PoseMetrics, ClassifyMetrics).

        Raises:
            TypeError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # Print mAP50-95
        """
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    @abstractmethod
    def export(
        self,
        **kwargs: Any,
    ) -> str:
        """Export the model to a different format suitable for deployment.

        This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
        purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
        defaults, and any additional arguments provided.

        Args:
            **kwargs (Any): Arbitrary keyword arguments for export configuration. Common options include:
                - format (str): Export format (e.g., 'onnx', 'engine', 'coreml').
                - quantize (int | str): Precision, e.g. 16 (FP16) or 8 (INT8); 32/None is FP32.
                - device (str): Device to run the export on.
                - workspace (int): Maximum memory workspace size for TensorRT engines.
                - nms (bool): Add Non-Maximum Suppression (NMS) module to model.
                - simplify (bool): Simplify ONNX model.

        Returns:
            (str): The path to the exported model file.

        Raises:
            TypeError: If the model is not a PyTorch model.
            ValueError: If an unsupported export format is specified.
            RuntimeError: If the export process fails due to errors.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> model.export(format="onnx", dynamic=True, simplify=True)
            'path/to/exported/model.onnx'
        """
        raise NotImplementedError()

    @abstractmethod
    def train(
        self,
        trainer=None,
        **kwargs: Any,
    ):
        """Train the model using the specified dataset and training configuration.

        This method facilitates model training with a range of customizable settings. It supports training with a custom
        trainer or the default training approach. The method handles scenarios such as resuming training from a
        checkpoint, integrating with Ultralytics HUB, and updating model and configuration after training.

        When using Ultralytics HUB, if the session has a loaded model, the method prioritizes HUB training arguments and
        warns if local arguments are provided. It checks for pip updates and combines default configurations,
        method-specific defaults, and user-provided arguments to configure the training process.

        Args:
            trainer (BaseTrainer, optional): Custom trainer instance for model training. If None, uses default.
            **kwargs (Any): Arbitrary keyword arguments for training configuration. Common options include:
                - data (str): Path to dataset configuration file.
                - epochs (int): Number of training epochs.
                - batch (int): Batch size for training.
                - imgsz (int): Input image size.
                - device (str): Device to run training on (e.g., 'cuda', 'cpu').
                - workers (int): Number of worker threads for data loading.
                - optimizer (str): Optimizer to use for training.
                - lr0 (float): Initial learning rate.
                - patience (int): Epochs to wait for no observable improvement for early stopping of training.
                - augmentations (list[Callable]): List of augmentation functions to apply during training.

        Returns:
            (ultralytics.utils.metrics.DetMetrics | dict | None): Training metrics if available and training is
                successful; otherwise, None. The specific metrics type depends on the task. When `data` is a list or
                tuple of datasets, the base model is fine-tuned on each in series and a {dataset: metrics} dict is
                returned.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> results = model.train(data="coco8.yaml", epochs=3)
            >>> multi = model.train(data=["coco8.yaml", "african-wildlife.yaml"], epochs=3)  # fine-tune across datasets
        """
        raise NotImplementedError()

    def _apply(self, fn) -> BaseModel:
        """Apply a function to model parameters, buffers, and tensors.

        This method extends the functionality of the parent class's _apply method by additionally resetting the
        predictor and updating the device in the model's overrides. It's typically used for operations like moving the
        model to a different device or changing its precision.

        Args:
            fn (Callable): A function to be applied to the model's tensors. This is typically a method like to(), cpu(),
                cuda(), half(), or float().

        Returns:
            (Model): The model instance with the function applied and updated attributes.

        Raises:
            TypeError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo26n.pt")
            >>> model = model._apply(lambda t: t.cuda())  # Move model to GPU
        """
        self._check_is_pytorch_model()
        self = super()._apply(fn)
        self.predictor = None  # reset predictor as device may have changed
        self.overrides["device"] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def device(self) -> torch.device:
        """Get the device on which the model's parameters are allocated.

        This property determines the device (CPU or GPU) where the model's parameters are currently stored. It is
        applicable only to models that are instances of torch.nn.Module.

        Returns:
            (torch.device | None): The device (CPU/GPU) of the model, or None if the model is not a torch.nn.Module
                instance.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> print(model.device)
            device(type='cuda', index=0)  # if CUDA is available
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        return next(self.model.parameters()).device if isinstance(self.model, torch.nn.Module) else None

    @property
    def transforms(self):
        """Retrieve the transformations applied to the input data of the loaded model.

        This property returns the transformations if they are defined in the model. The transforms typically include
        preprocessing steps like resizing, normalization, and data augmentation that are applied to input data before it
        is fed into the model.

        Returns:
            (object | None): The transform object of the model if available, otherwise None.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"Model transforms: {transforms}")
            ... else:
            ...     print("No transforms defined for this model.")
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """Add a callback function for a specified event.

        This method allows registering custom callback functions that are triggered on specific events during model
        operations such as training or inference. Callbacks provide a way to extend and customize the behavior of the
        model at various stages of its lifecycle.

        Args:
            event (str): The name of the event to attach the callback to. Must be a valid event name recognized by the
                Ultralytics framework.
            func (Callable): The callback function to be registered. This function will be called when the specified
                event occurs.

        Examples:
            >>> def on_train_start(trainer):
            ...     print("Training is starting!")
            >>> model = YOLO("yolo26n.pt")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """Clear all callback functions registered for a specified event.

        This method removes all custom and default callback functions associated with the given event. It resets the
        callback list for the specified event to an empty list, effectively removing all registered callbacks for that
        event.

        Args:
            event (str): The name of the event for which to clear the callbacks. This should be a valid event name
                recognized by the Ultralytics callback system.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> model.add_callback("on_train_start", lambda: print("Training started"))
            >>> model.clear_callback("on_train_start")
            >>> # All callbacks for 'on_train_start' are now removed

        Notes:
            - This method affects both custom callbacks added by the user and default callbacks
              provided by the Ultralytics framework.
            - After calling this method, no callbacks will be executed for the specified event
              until new ones are added.
            - Use with caution as it removes all callbacks, including essential ones that might
              be required for proper functioning of certain operations.
        """
        self.callbacks[event] = []

    @abstractmethod
    def reset_callbacks(self) -> None:
        """Reset all callbacks to their default functions.

        This method reinstates the default callback functions for all events, removing any custom callbacks that were
        previously added. It iterates through all default callback events and replaces the current callbacks with the
        default ones.

        The default callbacks are defined in the 'callbacks.default_callbacks' dictionary, which contains predefined
        functions for various events in the model's lifecycle, such as on_train_start, on_epoch_end, etc.

        This method is useful when you want to revert to the original set of callbacks after making custom
        modifications, ensuring consistent behavior across different runs or experiments.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # All callbacks are now reset to their default functions
        """
        pass

    @staticmethod
    def _reset_ckpt_args(args: dict[str, Any]) -> dict[str, Any]:
        """Reset specific arguments when loading a PyTorch model checkpoint.

        This method filters the input arguments dictionary to retain only a specific set of keys that are considered
        important for model loading. It's used to ensure that only relevant arguments are preserved when loading a model
        from a checkpoint, discarding any unnecessary or potentially conflicting settings.

        Args:
            args (dict[str, Any]): A dictionary containing various model arguments and settings.

        Returns:
            (dict[str, Any]): A new dictionary containing only the specified include keys from the input arguments.

        Examples:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _smart_load(self, key: str):
        """Intelligently load the appropriate module based on the model task.

        This method dynamically selects and returns the correct module (model, trainer, validator, or predictor) based
        on the current task of the model and the provided key. It uses the task_map dictionary to determine the
        appropriate module to load for the specific task.

        Args:
            key (str): The type of module to load. Must be one of 'model', 'trainer', 'validator', or 'predictor'.

        Returns:
            (object): The loaded module class corresponding to the specified key and current task.

        Raises:
            NotImplementedError: If the specified key is not supported for the current task.

        Examples:
            >>> model = Model(task="detect")
            >>> predictor_class = model._smart_load("predictor")
            >>> trainer_class = model._smart_load("trainer")
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(f"'{name}' model does not support '{mode}' mode for '{self.task}' task.") from e

    @property
    def task_map(self) -> dict:
        """Provide a mapping from model tasks to corresponding classes for different modes.

        This property method returns a dictionary that maps each supported task (e.g., detect, segment, semantic,
        classify) to a nested dictionary. The nested dictionary contains mappings for different operational modes
        (model, trainer, validator, predictor) to their respective class implementations.

        The mapping allows for dynamic loading of appropriate classes based on the model's task and the desired
        operational mode. This facilitates a flexible and extensible architecture for handling various tasks and modes
        within the Ultralytics framework.

        Returns:
            (dict[str, dict[str, Any]]): A dictionary mapping task names to nested dictionaries. Each nested dictionary
                contains mappings for 'model', 'trainer', 'validator', and 'predictor' keys to their respective class
                implementations for that task.

        Examples:
            >>> model = Model("yolo26n.pt")
            >>> task_map = model.task_map
            >>> detect_predictor = task_map["detect"]["predictor"]
            >>> segment_trainer = task_map["segment"]["trainer"]
        """
        raise NotImplementedError("Please provide task map for your model!")

    def eval(self):
        """Sets the model to evaluation mode.

        This method changes the model's mode to evaluation, which affects layers like dropout and batch normalization
        that behave differently during training and evaluation. In evaluation mode, these layers use running statistics
        rather than computing batch statistics, and dropout layers are disabled.

        Returns:
            (Model): The model instance with evaluation mode set.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> model.eval()
            >>> # Model is now in evaluation mode for inference
        """
        self.model.eval()
        return self

    def __getattr__(self, name):
        """Enable accessing model attributes directly through the Model class.

        This method provides a way to access attributes of the underlying model directly through the Model class
        instance. It first checks if the requested attribute is 'model', in which case it returns the model from
        the module dictionary. Otherwise, it delegates the attribute lookup to the underlying model.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            (Any): The requested attribute value.

        Raises:
            AttributeError: If the requested attribute does not exist in the model.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> print(model.stride)  # Access model.stride attribute
            >>> print(model.names)  # Access model.names attribute
        """
        return self._modules["model"] if name == "model" else getattr(self.model, name)
