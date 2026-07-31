from __future__ import annotations

from pathlib import Path
import torch

from vision_ai_platform.packages.core.results import Results
from vision_ai_platform.packages.utils import LOGGER
from vision_ai_platform.packages.utils.plotting import save_one_box

def save_txt(results: Results, txt_file: str | Path, save_conf: bool = False) -> str:
    """Save detection results to a text file.

    Args:
        txt_file (str | Path): Path to the output text file.
        save_conf (bool): Whether to include confidence scores in the output.

    Returns:
        (str): Path to the saved text file.

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
    if results.semantic_mask is not None:
        LOGGER.warning("Semantic Segmentation task does not support `save_txt`.")
        return str(txt_file)
    is_obb = results.obb is not None
    boxes = results.obb if is_obb else results.boxes
    masks = results.masks
    probs = results.probs
    kpts = results.keypoints
    texts = []
    if probs is not None:
        # Classify
        [texts.append(f"{probs.data[j]:.2f} {results.names[j]}") for j in probs.top5]
    elif boxes:
        # Detect/segment/pose
        for j, d in enumerate(boxes):
            c, conf, id = int(d.cls), float(d.conf), int(d.id.item()) if d.is_track else None
            line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
            if masks:
                seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                line = (c, *seg)
            if kpts is not None:
                kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                line += (*kpt.reshape(-1).tolist(),)
            line += (conf,) * save_conf + (() if id is None else (id,))
            texts.append(("%g " * len(line)).rstrip() % line)

    if texts:
        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
        with open(txt_file, "a", encoding="utf-8") as f:
            f.writelines(text + "\n" for text in texts)

    return str(txt_file)

def save_crop(results: Results, save_dir: str | Path, file_name: str | Path = Path("im.jpg")):
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
    if results.probs is not None:
        LOGGER.warning("Classify task does not support `save_crop`.")
        return
    if results.obb is not None:
        LOGGER.warning("OBB task does not support `save_crop`.")
        return
    if results.semantic_mask is not None:
        LOGGER.warning("Semantic Segmentation task does not support `save_crop`.")
        return
    for d in results.boxes:
        save_one_box(
            d.xyxy,
            results.orig_img.copy(),
            file=Path(save_dir) / results.names[int(d.cls)] / Path(file_name).with_suffix(".jpg"),
            BGR=True,
        )