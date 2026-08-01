from typing import Optional, Dict, List, Callable, Union, Any
from pathlib import Path
from abc import ABC

import cv2
import numpy as np
import torch

from vision_ai_platform.packages.core import BasePredictor, YOLOConfig
from vision_ai_platform.packages.utils import LOGGER, callbacks, MACOS, WINDOWS, ops, colorstr
from vision_ai_platform.packages.utils.check import check_imshow, check_imgsz
from vision_ai_platform.packages.utils.device_utils import select_device, attempt_compile, smart_inference_mode
from vision_ai_platform.packages.utils.files import increment_path, get_save_dir
from vision_ai_platform.packages.ai.data.augment import LetterBox
from vision_ai_platform.packages.ai.nn.autobackend import AutoBackend
from vision_ai_platform.packages.ai.data.dataloader import load_inference_source


STREAM_WARNING = """
Inference results will accumulate in RAM unless `stream=True` is passed, which can cause out-of-memory errors for large
sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""


class Predictor(BasePredictor, ABC):
    def __init__(
        self,
        cfg: YOLOConfig,
        save_dir: Optional[str | Path] = None,
        _callbacks: Optional[Dict[str, List[Callable]]] = None,
    ):
        """Initialize the custom predictor with YOLOConfig.

        Args:
            cfg (YOLOConfig): Pydantic or structured configuration instance.
            save_dir (str | Path, optional): Directory to save the prediction results
            _callbacks (dict, optional): Registered callback dictionary.
        """
        super().__init__(cfg, save_dir, _callbacks)
        if self.save_dir is None:
            self.save_dir = get_save_dir(cfg)
        if self.cfg.show:
            self.cfg.show = check_imshow(warn=True)

        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def setup_model(self, model: Union[str, Path, torch.nn.Module, None] = None, verbose: bool = True):
        """Initialize backend model and prepare for evaluation mode.

        Args:
            model (str | Path | torch.nn.Module): Model to load or use.
            verbose (bool): Whether to print verbose output.
        """
        if hasattr(model, "end2end"):
            if self.cfg.end2end is not None:
                model.end2end = self.cfg.end2end
            if model.end2end:
                # Keep head top-k >= 300 so `classes` filtering in NMS sees all candidates before `max_det` truncation
                model.set_head_attr(max_det=max(self.cfg.max_det, 300), agnostic_nms=self.cfg.agnostic_nms)
        self.model = AutoBackend(
            model=model or self.cfg.model,
            device=select_device(self.cfg.device, verbose=verbose),
            dnn=self.cfg.dnn,
            data=self.cfg.data,
            fp16=self.cfg.quantize == 16,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # update device
        self.cfg.quantize = 16 if self.model.fp16 else None  # record actual inference precision
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.cfg.imgsz = self.model.imgsz  # reuse imgsz from export metadata
        self.model.eval()
        self.model = attempt_compile(self.model, device=self.device, mode=self.cfg.compile)

    def setup_source(self, source: Any, stride: Optional[int] = None):
        """Configure video/image dataloader source.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor): Source for
                inference.
            stride (int, optional): Model stride for image size checking.
        """
        self.imgsz = check_imgsz(self.cfg.imgsz, stride=stride or self.model.stride, min_dim=2)  # check image size
        self.dataset = load_inference_source(
            source=source,
            batch=self.cfg.batch,
            vid_stride=self.cfg.vid_stride,
            buffer=self.cfg.stream_buffer,
            channels=getattr(self.model, "channels", 3),
        )
        self.source_type = self.dataset.source_type
        if (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # long sequence
            import torchvision  # noqa (import here triggers torchvision NMS use in nms.py)

            if not getattr(self, "stream", True):  # videos
                LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """Pre-transform input image before inference.

        Args:
            im (list[np.ndarray]): List of images with shape [(H, W, 3) x N].

        Returns:
            (list[np.ndarray]): List of transformed images.
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.cfg.rect
            and (self.model.format == "pt" or (getattr(self.model, "dynamic", False) and self.model.format != "imx")),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]

    def inference(self, im: torch.Tensor, *args, **kwargs):
        """Execute model forward pass."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.cfg.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.cfg.augment, visualize=visualize, embed=self.cfg.embed, *args, **kwargs)

    def save_predicted_images(self, save_path: Path, frame: int = 0):
        """Save predictions to image or video files.
        
        Args:
            save_path (Path): Path to save the results.
            frame (int): Frame number for video mode.
        """
        im = self.plotted_img

        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = self.save_dir / f"{save_path.stem}_frames"  # save frames to a separate directory
            if save_path not in self.vid_writer:  # new video
                if self.cfg.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            # Save video
            self.vid_writer[save_path].write(im)
            if self.cfg.save_frames:
                cv2.imwrite(f"{frames_path}/{save_path.stem}_{frame}.jpg", im)

        # Save images
        else:
            cv2.imwrite(str(save_path.with_suffix(".jpg")), im)  # save to JPG for best support

    def __call__(self, source: Any = None, model: Any = None, stream: bool = False, *args, **kwargs):
        """Call method enabling predictor execution as callable.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Yields:
            (vision_ai_platform.packages.core.results.Results): Results objects.
        """
        self.stream = stream
        if stream:
            return self._stream_inference(source, model, *args, **kwargs)
        else:
            return list(self._stream_inference(source, model, *args, **kwargs))

    @smart_inference_mode()
    def _stream_inference(self, source=None, model=None, *args, **kwargs):
        """Stream inference on input source and save results to file.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Yields:
            (vision_ai_platform.packages.core.results.Results): Results objects.
        """
        if self.cfg.verbose:
            LOGGER.info("")

        # Setup model
        if self.model is None:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.cfg.source)

            # Check if save_dir/ label file exists
            if self.cfg.save or self.cfg.save_txt:
                (self.save_dir / "labels" if self.cfg.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(
                    imgsz=(
                        1 if self.model.format in {"pt", "triton"} else self.dataset.bs,
                        self.model.channels,
                        *self.imgsz,
                    )
                )
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for batch in self.dataset:
                self.batch = batch
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.cfg.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                        if (
                            self.cfg.verbose
                            or self.cfg.save
                            or self.cfg.save_txt
                            or self.cfg.save_crop
                            or self.cfg.show
                        ):
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break

                # Print batch results
                if self.cfg.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.cfg.show:
            cv2.destroyAllWindows()  # close any open windows

        # Print final results
        if self.cfg.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.cfg.batch, self.seen), getattr(self.model, 'channels', 3), *im.shape[2:])}" % t
            )
        if self.cfg.save or self.cfg.save_txt or self.cfg.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.cfg.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")
    