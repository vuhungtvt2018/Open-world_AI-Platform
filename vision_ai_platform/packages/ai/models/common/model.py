from abc import ABC
from pathlib import Path
from PIL import Image
from typing import Any, Iterator

import numpy as np
import torch

from vision_ai_platform.packages.core import BaseModel, YOLOConfig, ExporterConfig
from vision_ai_platform.packages.core.results import Results
from vision_ai_platform.packages.utils import (
    LOGGER,
    RANK,
    callbacks,
    DEFAULT_CFG_DICT,
    check,
    SETTINGS,
    ASSETS,
    YAML,
    TASK2DATA,
)
from vision_ai_platform.packages.utils.files import get_save_dir, get_latest_run
from vision_ai_platform.packages.ai.nn.tasks import guess_model_task, yaml_model_load, load_checkpoint

class Model(BaseModel, ABC):
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
        super().__init__(model, task, verbose)

        self.callbacks = callbacks.get_default_callbacks()

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
        cfg_dict = yaml_model_load(cfg_path)
        self.cfg_path = cfg_path
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task
        self.model_name = cfg_path    

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
        if weights.lower().startswith(check.REMOTE_FILE_PREFIXES):
            weights = check.check_file(weights, download_dir=SETTINGS["weights_dir"])  # download and return local file
        weights = check.check_model_file_from_stem(weights)  # add suffix, i.e. yolo26n -> yolo26n.pt

        if str(weights).rpartition(".")[-1] == "pt":
            self.model, self.ckpt = load_checkpoint(weights)
            self.task = self.model.task
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = check.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

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
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights  # remember the weights for DDP training
            weights, self.ckpt = load_checkpoint(weights)
        self.model.load(weights)
        return self

    def predict(
        self,
        source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> Iterator[Results | torch.Tensor] | list[Results] | list[torch.Tensor]:
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
            (Iterator[vision_ai_platform.packages.core.results.Results | torch.Tensor] | list[vision_ai_platform.packages.core.results.Results] |
            list[torch.Tensor]): Prediction results or embeddings, streamed when `stream=True`.

        Notes:
            - If 'source' is not provided, it defaults to the ASSETS constant with a warning.
            - The method sets up a new predictor if not already present and updates its arguments with each call.
            - For SAM-type models, 'prompts' can be passed as a keyword argument.
        """
        if source is None:
            source = "https://ultralytics.com/images/boats.jpg" if self.task == "obb" else ASSETS
            LOGGER.warning(f"'source' is missing. Using 'source={source}'.")
        
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "rect": True, "embed": None}
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor or self.predictor.cfg.device != args.get("device", self.predictor.cfg.device):
            predictor_cfg = YOLOConfig().model_copy(update=args)
            save_dir = get_save_dir(predictor_cfg)
            self.predictor = (predictor or self._smart_load("predictor"))(predictor_cfg, save_dir, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        else:  # only update args if predictor is already setup
            self.predictor.cfg = self.predictor.cfg.model_copy(update=args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.cfg)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor(source=source, stream=stream)

    def track(
        self,
        source: str | Path | int | list | tuple | np.ndarray | torch.Tensor = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> list[Results]:
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
            (list[vision_ai_platform.packages.core.results.Results]): A list of tracking results, each a Results object.

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
        if not hasattr(self.predictor, "trackers"):
            from vision_ai_platform.packages.ai.tracker import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

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
        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
            if any(kwargs):
                LOGGER.warning("using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args  # overwrite kwargs

        overrides = YAML.load(check.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            # NOTE: handle the case when 'cfg' includes 'data'.
            "data": (overrides.get("data") if kwargs.get("cfg") else None)
            or DEFAULT_CFG_DICT["data"]
            or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train", "session": self.session}  # prioritizes rightmost args
        pretrained = kwargs.get("pretrained", overrides.get("pretrained", True) if kwargs.get("cfg") else True)
        if args.get("resume"):
            if args["resume"] is True:  # resume=True (boolean) uses current model as checkpoint
                if self.ckpt and self.ckpt.get("epoch", -1) >= 0 and self.ckpt.get("optimizer") is not None:
                    args["resume"] = self.ckpt_path
                else:
                    LOGGER.warning(
                        f"model '{self.ckpt_path}' is not a resumable training checkpoint "
                        f"(missing epoch/optimizer state). Use 'resume' only to continue incomplete training. "
                        f"Starting new training instead."
                    )
                    args["resume"] = False

        trainer_cfg = YOLOConfig().model_copy(update=args)
        save_dir = get_save_dir(trainer_cfg)
        self.trainer = (trainer or self._smart_load("trainer"))(cfg=trainer_cfg, save_dir=save_dir, _callbacks=self.callbacks)
        if not args.get("resume") and self.ckpt:
            # Reuse the already-loaded checkpoint model to avoid re-resolving remote weight sources during trainer setup.
            weights = None if pretrained is False else self.model
            if isinstance(pretrained, (str, Path)):
                weights, _ = load_checkpoint(pretrained)
            self.trainer.model = self.trainer.get_model(weights=weights, cfg=self.model.yaml)
            self.model = self.trainer.model

        self.trainer.train()
        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Training completed but no checkpoint was saved. Expected {self.trainer.best} or {self.trainer.last}."
                )
            self.model, self.ckpt = load_checkpoint(ckpt)
            self.overrides = self._reset_ckpt_args(self.model.args)
            self.metrics = getattr(self.trainer.validator, "metrics", None)
            if self.metrics is None and self.ckpt:  # recover from checkpoint under DDP (validator runs in subprocess)
                self.metrics = self.ckpt.get("train_metrics")
        return self.metrics

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
        self._check_is_pytorch_model()
        from .export import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # reset to avoid multi-GPU errors
            "verbose": False,
        }  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        export_cfg = ExporterConfig().model_copy(update=args)
        return Exporter(cfg=export_cfg, _callbacks=self.callbacks)(model=self.model)

    @property
    def names(self) -> dict[int, str]:
        """Retrieve the class names associated with the loaded model.

        This property returns the class names if they are defined in the model. It checks the class names for validity
        using the 'check_class_names' function from the ultralytics.nn.autobackend module. If the predictor is not
        initialized, it sets it up before retrieving the names.

        Returns:
            (dict[int, str]): A dictionary of class names associated with the model, where keys are class indices and
                values are the corresponding class names.

        Raises:
            AttributeError: If the model or predictor does not have a 'names' attribute.

        Examples:
            >>> model = YOLO("yolo26n.pt")
            >>> print(model.names)
            {0: 'person', 1: 'bicycle', 2: 'car', ...}
        """
        from vision_ai_platform.packages.ai.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:  # export formats will not have predictor defined until predict() is called
            predict_cfg = YOLOConfig().model_copy(update=self.overrides)
            save_dir = get_save_dir(predict_cfg)
            predictor = self._smart_load("predictor")(cfg=predict_cfg, save_dir=save_dir, _callbacks=self.callbacks)
            predictor.setup_model(model=self.model, verbose=False)  # do not mess with self.predictor.model args
            return predictor.model.names
        return self.predictor.model.names

    