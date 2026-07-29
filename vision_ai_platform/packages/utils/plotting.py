from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from . import threaded
from .logger import LOGGER
from .annotator import colors, Annotator

def _gaussian_filter1d(y, sigma: int = 3, truncate: float = 4.0) -> np.ndarray:
    """Smooth a 1D array with a Gaussian kernel (NumPy replacement for scipy.ndimage.gaussian_filter1d).

    Args:
        y (np.ndarray): Input 1D array to smooth.
        sigma (int): Standard deviation of the Gaussian kernel.
        truncate (float): Truncate the kernel at this many standard deviations.

    Returns:
        (np.ndarray): Smoothed 1D array with the same length as the input.
    """
    y = np.asarray(y, dtype=float)
    radius = int(truncate * sigma + 0.5)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    # scipy 'reflect' boundary mode is equivalent to NumPy 'symmetric'
    return np.convolve(np.pad(y, radius, mode="symmetric"), kernel, mode="valid")


def xywhr2xyxyxyxy(x):
    """Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4] format.

    Args:
        x (np.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format with shape (N, 5) or (B, N, 5). Rotation
            values should be in radians from [-pi/4, 3pi/4).

    Returns:
        (np.ndarray | torch.Tensor): Converted corner points with shape (N, 4, 2) or (B, N, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def xywh2xyxy(x):
    """Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is
    the top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x, y, width, height) format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    if isinstance(x, torch.Tensor):
        y = torch.empty_like(x)
    else:
        y = np.empty_like(x)
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


class Plotter:
    """Class for plotting and visualizing training metrics and results from CSV logs,
    """

    def __init__(
        self,
        csv_path: Union[str, Path] = "results.csv",
        save_dir: Optional[Union[str, Path]] = None,
        smooth_sigma: int = 3,
    ):
        """Initialize the Plotter object.

        Args:
            csv_path (Union[str, Path]): Path to the results CSV file.
            save_dir (Optional[Union[str, Path]]): Directory where plots will be saved.
                Defaults to the parent directory of `csv_path`.
            smooth_sigma (int): Standard deviation for Gaussian smoothing.
        """
        self.csv_path = Path(csv_path)
        self.save_dir = Path(save_dir) if save_dir else self.csv_path.parent
        self.smooth_sigma = smooth_sigma

    def plot_results(
        self,
        save_filename: str = "results.png",
        on_plot: Optional[Callable[[Path], None]] = None,
    ) -> Path:
        """Plot training losses and validation metrics from the CSV log and save as an image.

        Args:
            save_filename (str): Output filename for the generated plot.
            on_plot (Optional[Callable[[Path], None]]): Optional callback function triggered after saving.

        Returns:
            Path: Output path of the saved plot.
        """
        import matplotlib.pyplot as plt
        import polars as pl

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Results file not found at: {self.csv_path}")

        # Read CSV file and clean column names
        df = pl.read_csv(self.csv_path, infer_schema_length=None)
        df.columns = df.columns.str.strip()

        # Identify plot columns (ignoring epoch/step index columns)
        metric_cols = [c for c in df.columns if c.lower() not in {"epoch", "step"}]
        num_metrics = len(metric_cols)

        if num_metrics == 0:
            raise ValueError("No valid metric columns found in the CSV file.")

        # Determine grid shape (aim for 2 rows by default, capped at max 5 columns per row)
        ncols = min(5, num_metrics)
        nrows = (num_metrics + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.5 * ncols, 3 * nrows), tight_layout=True
        )
        axes = np.atleast_1d(axes).ravel()

        # Plot each column metric
        x = df["epoch"] if "epoch" in df.columns else np.arange(len(df))

        for idx, col in enumerate(metric_cols):
            ax = axes[idx]
            y_raw = df[col].to_numpy()

            # Raw values (faded)
            ax.plot(x, y_raw, color="tab:blue", alpha=0.3, linewidth=1.5, label="raw")

            # Smoothed values using the custom Gaussian filter
            y_smooth = _gaussian_filter1d(y_raw, sigma=self.smooth_sigma)
            ax.plot(
                x, y_smooth, color="tab:blue", alpha=1.0, linewidth=2.0, label="smooth"
            )

            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)

        # Hide unused axes in the grid
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis("off")

        # Save plot figure
        save_path = self.save_dir / save_filename
        self.save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Trigger callback if provided
        if on_plot:
            on_plot(save_path)

        return save_path

    def plot_multitrain_results(self, scores: dict, key: str = "fitness"):
        """Plot per-dataset metrics from a multi-dataset training run as a bar chart with the cross-dataset mean.

        Args:
            scores (dict): Mapping of dataset name to its scalar metric value.
            key (str): Name of the plotted metric, used as the y-axis label.

        Returns:
            (Path): Path to the saved figure.
        """
        import matplotlib.pyplot as plt

        mean = sum(scores.values()) / len(scores)
        fig, ax = plt.subplots(figsize=(max(6.0, len(scores) * 0.45), 5), tight_layout=True)
        ax.bar(range(len(scores)), list(scores.values()), color="#042AFF")
        ax.axhline(mean, color="orange", linestyle="--", label=f"mean = {mean:.3f}")
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(list(scores), rotation=90)
        ax.set_ylabel(key)
        ax.set_title(f"MultiTrainer results across {len(scores)} datasets")
        ax.legend()
        fname = Path(self.save_dir) / "multitrain_results.png"
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        return fname

    def plot_labels(self, boxes, cls, names=(), on_plot=None):
        """Plot training labels including class histograms and box statistics.

        Args:
            boxes (np.ndarray): Bounding box coordinates in format [x, y, width, height].
            cls (np.ndarray): Class indices.
            names (dict, optional): Dictionary mapping class indices to class names.
            save_dir (Path, optional): Directory to save the plot.
            on_plot (Callable, optional): Function to call after plot is saved.
        """
        import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'
        import polars
        from matplotlib.colors import LinearSegmentedColormap

        # Plot dataset labels
        LOGGER.info(f"Plotting labels to {self.save_dir / 'labels.jpg'}... ")
        nc = int(cls.max() + 1)  # number of classes
        boxes = boxes[:1000000]  # limit to 1M boxes
        x = polars.DataFrame(boxes, schema=["x", "y", "width", "height"])

        # Matplotlib labels
        subplot_3_4_color = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])
        ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
        y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
        for i in range(nc):
            y[2].patches[i].set_color([x / 255 for x in colors(i)])
        ax[0].set_ylabel("instances")
        if 0 < len(names) < 30:
            ax[0].set_xticks(range(len(names)))
            ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
            ax[0].bar_label(y[2])
        else:
            ax[0].set_xlabel("classes")
        boxes = np.column_stack([0.5 - boxes[:, 2:4] / 2, 0.5 + boxes[:, 2:4] / 2]) * 1000
        img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
        for class_id, box in zip(cls[:500], boxes[:500]):
            ImageDraw.Draw(img).rectangle(box.tolist(), width=1, outline=colors(class_id))  # plot
        ax[1].imshow(img)
        ax[1].axis("off")

        ax[2].hist2d(x["x"], x["y"], bins=50, cmap=subplot_3_4_color)
        ax[2].set_xlabel("x")
        ax[2].set_ylabel("y")
        ax[3].hist2d(x["width"], x["height"], bins=50, cmap=subplot_3_4_color)
        ax[3].set_xlabel("width")
        ax[3].set_ylabel("height")
        for a in {0, 1, 2, 3}:
            for s in {"top", "right", "left", "bottom"}:
                ax[a].spines[s].set_visible(False)

        fname = self.save_dir / "labels.jpg"
        plt.savefig(fname, dpi=200)
        plt.close()
        if on_plot:
            on_plot(fname)
    
    @threaded
    def plot_images(
        self,
        labels: dict[str, Any],
        images: torch.Tensor | np.ndarray | None = None,
        paths: list[str] | None = None,
        fname: str = "images.jpg",
        names: dict[int, str] | None = None,
        on_plot: Callable | None = None,
        max_size: int = 1920,
        max_subplots: int = 16,
        save: bool = True,
        conf_thres: float = 0.25,
        show_labels: bool = True,
        show_conf: bool = True,
    ) -> np.ndarray | None:
        """Plot image grid with labels, bounding boxes, masks, and keypoints.

        Args:
            labels (dict[str, Any]): Dictionary containing detection data with keys like 'cls', 'bboxes', 'conf', 'masks',
                'keypoints', 'batch_idx', 'img'.
            images (torch.Tensor | np.ndarray): Batch of images to plot. Shape: (batch_size, channels, height, width).
            paths (list[str] | None): List of file paths for each image in the batch.
            fname (str): Output filename for the plotted image grid.
            names (dict[int, str] | None): Dictionary mapping class indices to class names.
            on_plot (Callable | None): Callback function to be called after saving the plot.
            max_size (int): Maximum size of the output image grid.
            max_subplots (int): Maximum number of subplots in the image grid.
            save (bool): Whether to save the plotted image grid to a file.
            conf_thres (float): Confidence threshold for displaying detections.
            show_labels (bool): Whether to display class labels.
            show_conf (bool): Whether to display confidence values.

        Returns:
            (np.ndarray | None): Plotted image grid as a numpy array if save is False, None otherwise.

        Notes:
            This function supports both tensor and numpy array inputs. It will automatically
            convert tensor inputs to numpy arrays for processing.

            Channel Support:
            - 1 channel: Grayscale
            - 2 channels: Third channel added as zeros
            - 3 channels: Used as-is (standard RGB)
            - 4+ channels: Cropped to first 3 channels
        """
        images = np.zeros((0, 3, 640, 640), dtype=np.float32) if images is None else images
        for k in ("cls", "bboxes", "conf", "masks", "keypoints", "batch_idx", "images", "semantic_mask", "depth"):
            if k not in labels:
                continue
            if k == "cls" and labels[k].ndim == 2:
                labels[k] = labels[k].squeeze(1)  # squeeze if shape is (n, 1)
            if isinstance(labels[k], torch.Tensor):
                labels[k] = labels[k].cpu().numpy()

        cls = labels.get("cls", np.zeros(0, dtype=np.int64))
        batch_idx = labels.get("batch_idx", np.zeros(cls.shape, dtype=np.int64))
        bboxes = labels.get("bboxes", np.zeros(0, dtype=np.float32))
        confs = labels.get("conf", None)
        masks = labels.get("masks", np.zeros(0, dtype=np.uint8))
        kpts = labels.get("keypoints", np.zeros(0, dtype=np.float32))
        semantic_masks = labels.get("semantic_mask", np.zeros(0, dtype=np.int64))
        depth_maps = labels.get("depth", np.zeros(0, dtype=np.float32))
        images = labels.get("img", images)  # default to input images

        if len(images) and isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()

        # Handle 2-ch and n-ch images
        c = images.shape[1]
        if c == 2:
            zero = np.zeros_like(images[:, :1])
            images = np.concatenate((images, zero), axis=1)  # pad 2-ch with a black channel
        elif c > 3:
            images = images[:, :3]  # crop multispectral images to first 3 channels

        bs, _, h, w = images.shape  # batch size, _, height, width
        bs = min(bs, max_subplots)  # limit plot images
        ns = np.ceil(bs**0.5)  # number of subplots (square)
        if np.max(images[0]) <= 1:
            images *= 255  # de-normalise (optional)

        # Build Image
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

        # Resize (optional)
        scale = max_size / ns / max(h, w)
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

        # Annotate
        fs = int((h + w) * ns * 0.01)  # font size
        fs = max(fs, 18)  # ensure that the font size is large enough to be easily readable.
        annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=str(names))
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
            if paths:
                annotator.text([x + 5, y + 5], text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
            if len(cls) > 0:
                idx = batch_idx == i
                classes = cls[idx].astype("int")
                labels = confs is None
                conf = confs[idx] if confs is not None else None  # check for confidence presence (label vs pred)

                if len(bboxes):
                    boxes = bboxes[idx]
                    if len(boxes):
                        if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                            boxes[..., [0, 2]] *= w  # scale to pixels
                            boxes[..., [1, 3]] *= h
                        elif scale < 1:  # absolute coords need scale if image scales
                            boxes[..., :4] *= scale
                    boxes[..., 0] += x
                    boxes[..., 1] += y
                    is_obb = boxes.shape[-1] == 5  # xywhr
                    boxes = xywhr2xyxyxyxy(boxes) if is_obb else xywh2xyxy(boxes)
                    for j, box in enumerate(boxes.astype(np.int64).tolist()):
                        c = classes[j]
                        color = colors(c)
                        c = names.get(c, c) if names else c
                        if labels or conf[j] > conf_thres:
                            conf_text = f"{conf[j]:.1f}" if conf is not None else ""
                            label = f"{c}" if show_labels else ""
                            label += f" {conf_text}".strip() if show_conf else ""
                            annotator.box_label(box, label, color=color)

                elif len(classes):
                    for c in classes:
                        color = colors(c)
                        c = names.get(c, c) if names else c
                        label = f"{c}" if labels else f"{c} {conf[0]:.1f}"
                        annotator.text([x, y], label, txt_color=color, box_color=(64, 64, 64, 128))

                # Plot keypoints
                if len(kpts):
                    kpts_ = kpts[idx].copy()
                    if len(kpts_):
                        if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01:  # if normalized with tolerance .01
                            kpts_[..., 0] *= w  # scale to pixels
                            kpts_[..., 1] *= h
                        elif scale < 1:  # absolute coords need scale if image scales
                            kpts_ *= scale
                    kpts_[..., 0] += x
                    kpts_[..., 1] += y
                    for j in range(len(kpts_)):
                        if labels or conf[j] > conf_thres:
                            annotator.kpts(kpts_[j], conf_thres=conf_thres)

                # Plot masks
                if len(masks):
                    if idx.shape[0] == masks.shape[0] and masks.max() <= 1:  # overlap_mask=False
                        image_masks = masks[idx]
                    else:  # overlap_mask=True
                        image_masks = masks[[i]]  # (1, 640, 640)
                        nl = idx.sum()
                        index = np.arange(1, nl + 1).reshape((nl, 1, 1))
                        image_masks = (image_masks == index).astype(np.float32)

                    im = np.asarray(annotator.im).copy()
                    for j in range(len(image_masks)):
                        if labels or conf[j] > conf_thres:
                            color = colors(classes[j])
                            mh, mw = image_masks[j].shape
                            if mh != h or mw != w:
                                mask = image_masks[j].astype(np.uint8)
                                mask = cv2.resize(mask, (w, h))
                                mask = mask.astype(bool)
                            else:
                                mask = image_masks[j].astype(bool)
                            try:
                                im[y : y + h, x : x + w, :][mask] = (
                                    im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                                )
                            except Exception:
                                pass
                    annotator.fromarray(im)

            # Plot semantic masks
            if len(semantic_masks) and i < len(semantic_masks):
                mask = semantic_masks[i]
                mh, mw = mask.shape
                if mh != h or mw != w:
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                im = np.asarray(annotator.im).copy()
                sub_annotator = Annotator(np.ascontiguousarray(im[y : y + h, x : x + w]), line_width=1, pil=False)
                sub_annotator.semantic_mask(mask, alpha=0.4)
                im[y : y + h, x : x + w] = sub_annotator.im
                annotator.fromarray(im)

            # Plot depth maps
            if len(depth_maps) and i < len(depth_maps):
                d = depth_maps[i]
                if d.ndim == 3:
                    d = d.squeeze(0)
                dh, dw = d.shape
                if dh != h or dw != w:
                    d = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                im = np.asarray(annotator.im).copy()
                # The mosaic deviates from the Annotator BGR-buffer convention (it holds RGB), so convert the patch
                # to BGR for the overlay, then back to RGB for the mosaic.
                sub_bgr = cv2.cvtColor(np.ascontiguousarray(im[y : y + h, x : x + w]), cv2.COLOR_RGB2BGR)
                sub_annotator = Annotator(sub_bgr, line_width=1, pil=False)
                sub_annotator.depth_map(d, alpha=0.6)
                im[y : y + h, x : x + w] = cv2.cvtColor(sub_annotator.im, cv2.COLOR_BGR2RGB)
                annotator.fromarray(im)

        if not save:
            return np.asarray(annotator.im)
        save_path = self.save_dir / fname
        annotator.im.save(save_path)  # save
        if on_plot:
            on_plot(save_path)