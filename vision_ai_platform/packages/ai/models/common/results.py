from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from vision_ai_platform.packages.core import BaseResults
from vision_ai_platform.packages.utils import LOGGER
from vision_ai_platform.packages.utils.annotator import Annotator, colors
from vision_ai_platform.packages.utils.plotting import save_one_box

class Results(BaseResults):
    def new(self):
        """Create a new Results object with the same image, path, names, and speed attributes.

        Returns:
            (Results): A new Results object with copied attributes from the original instance.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def copy(self):
        """Create a new Results object with the same image, path, names, and speed attributes.

        Returns:
            (Results): A new Results object with copied attributes from the original instance.
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    # def update(
    #     self,
    #     boxes: torch.Tensor | None = None,
    #     masks: torch.Tensor | None = None,
    #     probs: torch.Tensor | None = None,
    #     obb: torch.Tensor | None = None,
    #     keypoints: torch.Tensor | None = None,
    #     semantic_mask: torch.Tensor | None = None,
    #     depth: torch.Tensor | None = None,
    # ):
    #     """Update the Results object with new detection data.

    #     This method allows updating the boxes, masks, keypoints, probabilities, and oriented bounding boxes (OBB) of
    #     the Results object. It ensures that boxes are clipped to the original image shape.

    #     Args:
    #         boxes (torch.Tensor | None): A tensor of shape (N, 6) containing bounding box coordinates and confidence
    #             scores. The format is (x1, y1, x2, y2, conf, class).
    #         masks (torch.Tensor | None): A tensor of shape (N, H, W) containing segmentation masks.
    #         probs (torch.Tensor | None): A tensor of shape (num_classes,) containing class probabilities.
    #         obb (torch.Tensor | None): A tensor of shape (N, 7) or (N, 8) containing oriented bounding box coordinates.
    #         keypoints (torch.Tensor | None): A tensor of shape (N, K, 3) containing keypoints, where K=17 for persons.
    #         semantic_mask (torch.Tensor | None): A tensor of shape (H, W) containing class IDs for semantic
    #             segmentation.
    #         depth (torch.Tensor | None): A tensor of shape (H, W) containing per-pixel depth values.

    #     Examples:
    #         >>> results = model("image.jpg")
    #         >>> new_boxes = torch.tensor([[100, 100, 200, 200, 0.9, 0]])
    #         >>> results[0].update(boxes=new_boxes)
    #     """
    #     if boxes is not None:
    #         self.boxes = Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
    #     if masks is not None:
    #         self.masks = Masks(masks, self.orig_shape)
    #     if probs is not None:
    #         self.probs = Probs(probs)
    #     if obb is not None:
    #         self.obb = OBB(obb, self.orig_shape)
    #     if keypoints is not None:
    #         self.keypoints = Keypoints(keypoints, self.orig_shape)
    #     if semantic_mask is not None:
    #         self.semantic_mask = SemanticMask(semantic_mask, self.orig_shape)
    #     if depth is not None:
    #         self.depth = DepthMap(depth, self.orig_shape)

    def plot(
        self,
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: np.ndarray | torch.Tensor | None = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Plot detection results on an input BGR image.

        Args:
            conf (bool): Whether to plot detection confidence scores.
            line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
            font_size (float | None): Font size for text. If None, scaled to image size.
            font (str): Font to use for text.
            pil (bool): Whether to return the image as a PIL Image.
            img (np.ndarray | torch.Tensor | None): Image to plot on. Tensor images must be contiguous HWC BGR uint8. If
                None, uses the original image.
            kpt_radius (int): Radius of drawn keypoints.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot labels of bounding boxes.
            boxes (bool): Whether to plot bounding boxes.
            masks (bool): Whether to plot masks.
            probs (bool): Whether to plot classification probabilities.
            show (bool): Whether to display the annotated image.
            save (bool): Whether to save the annotated image.
            filename (str | None): Filename to save image if save is True.
            color_mode (str): Specify the color mode, e.g., 'instance' or 'class'.
            txt_color (tuple[int, int, int]): Text color in BGR format for classification output.

        Returns:
            (np.ndarray | PIL.Image.Image): Annotated image as a NumPy array (BGR) or PIL image (RGB) if `pil=True`.

        Examples:
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     im = result.plot(pil=True)
            ...     im.show()
        """
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).byte().cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        if pred_boxes is not None and (show_boxes or (pred_masks and show_masks)):
            pred_boxes = pred_boxes.cpu()  # one host transfer avoids per-box GPU syncs in the color and label loops
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            pred_mask_data = torch.as_tensor(pred_masks.data)  # no-op for torch, converts a numpy() result
            idx = (
                pred_boxes.id
                if pred_boxes and pred_boxes.is_track and color_mode == "instance"
                else pred_boxes.cls
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))
            )
            annotator.masks(pred_mask_data, colors=[colors(x, True) for x in idx])

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for i, d in enumerate(reversed(pred_boxes)):
                c = int(d.cls.item())  # .item() works for torch and numpy alike; int()/float() need 0-d since numpy 2.4
                d_conf, id = float(d.conf.item()) if conf else None, int(d.id.item()) if d.is_track else None
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {d_conf:.2f}" if conf else name) if labels else (f"{d_conf:.2f}" if conf else None)
                box = d.xyxyxyxy.squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        c
                        if color_mode == "class"
                        else id
                        if id is not None
                        else i
                        if color_mode == "instance"
                        else None,
                        True,
                    ),
                )

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = "\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=txt_color, box_color=(64, 64, 64, 128))  # RGBA box

        # Plot Semantic Segmentation results
        if self.semantic_mask is not None and show_masks:
            sem_mask = self.semantic_mask.data
            if isinstance(sem_mask, torch.Tensor):
                sem_mask = sem_mask.cpu().numpy()
            annotator.semantic_mask(sem_mask, alpha=0.5)

        # Plot Depth results — blend colorized depth heatmap over the image
        if self.depth is not None and show_masks:
            d = self.depth.data
            d = d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
            annotator.depth_map(d)

        # Plot Pose results
        if self.keypoints is not None:
            for i, k in enumerate(reversed(self.keypoints.cpu().numpy().data)):  # one host transfer, no per-kpt syncs
                annotator.kpts(
                    k,
                    self.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename or f"results_{Path(self.path).name}")

        return annotator.result(pil)

    def save_txt(self, txt_file: str | Path, save_conf: bool = False) -> str:
        """Save detection results to a text file.

        Args:
            txt_file (str | Path): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo26n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_txt("output.txt")

        Notes:
            - The file will contain one line per detection or classification with the following structure:
              - For detections: `class x_center y_center width height [confidence] [track_id]`
              - For classifications: `confidence class_name`
              - For masks and keypoints, the specific formats will vary accordingly.
            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.
            - Existing contents of the file will not be overwritten; new results will be appended.
            - This method does not support Semantic Segmentation tasks.
        """
        if self.semantic_mask is not None:
            LOGGER.warning("Semantic Segmentation task does not support `save_txt`.")
            return str(txt_file)
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls.item()), float(d.conf.item()), int(d.id.item()) if d.is_track else None
                line = (c, *(d.xyxyxyxyn.reshape(-1) if is_obb else d.xywhn.reshape(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = kpts[j].xyn
                    if kpts[j].has_visible:
                        kpt = torch.cat((torch.as_tensor(kpt), torch.as_tensor(kpts[j].conf)[..., None]), 2)
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(text + "\n" for text in texts)

        return str(txt_file)

    def save_crop(self, save_dir: str | Path, file_name: str | Path = Path("im.jpg")):
        """Save cropped detection images to specified directory.

        This method saves cropped images of detected objects to a specified directory. Each crop is saved in a
        subdirectory named after the object's class, with the filename based on the input file_name.

        Args:
            save_dir (str | Path): Directory path where cropped images will be saved.
            file_name (str | Path): Base filename for the saved cropped images.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_crop(save_dir="path/to/crops", file_name="detection")

        Notes:
            - This method does not support Classify, Oriented Bounding Box (OBB), or Semantic Segmentation tasks.
            - Crops are saved as 'save_dir/class_name/file_name.jpg'.
            - The method will create necessary subdirectories if they don't exist.
            - Original image is copied before cropping to avoid modifying the original.
        """
        if self.probs is not None:
            LOGGER.warning("Classify task does not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("OBB task does not support `save_crop`.")
            return
        if self.semantic_mask is not None:
            LOGGER.warning("Semantic Segmentation task does not support `save_crop`.")
            return
        if self.depth is not None:
            LOGGER.warning("Depth task does not support `save_crop`.")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls.item())] / Path(file_name).with_suffix(".jpg"),
                BGR=True,
            )
