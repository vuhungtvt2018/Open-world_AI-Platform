"""
packages/utils/session.py
-------------------------
Stateless helper for creating inspection session directories.
Belongs to Layer 2 (utils) — no dependency on any layer above.
"""
import os
from datetime import datetime
from packages.utils.utils import ensure_dirs


def make_session_dir(base_dir: str):
    """Create a timestamped session directory structure under base_dir/sessions/<timestamp>/
    and return tuple of all sub-directory paths.
    """
    sess = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base_dir, "sessions", sess)

    original             = os.path.join(path, "original")
    crops                = os.path.join(path, "crops")
    masks                = os.path.join(path, "masks")
    bboxes               = os.path.join(path, "bboxes")
    ng_detected          = os.path.join(path, "ng_detected")
    anom_bboxes          = os.path.join(path, "anomaly_bboxes")
    anom_crops           = os.path.join(path, "anomaly_crops")
    anom_crop_vis        = os.path.join(path, "anomaly_crops_vis")
    keypoint_crops       = os.path.join(path, "keypoint_crops")
    masks_keypoint_crops = os.path.join(path, "masks_keypoint_crops")

    ensure_dirs(
        original, crops, masks, bboxes,
        ng_detected, anom_bboxes, anom_crops, anom_crop_vis,
        keypoint_crops, masks_keypoint_crops
    )

    return (
        path, original, crops, masks,
        bboxes, ng_detected, anom_bboxes, anom_crops,
        anom_crop_vis, keypoint_crops, masks_keypoint_crops
    )
