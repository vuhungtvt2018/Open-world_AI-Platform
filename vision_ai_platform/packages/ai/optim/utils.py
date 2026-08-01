from pathlib import Path
from typing import Any
from datetime import datetime
import os

import torch

from vision_ai_platform import __version__
from vision_ai_platform.packages.utils import LOGGER, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS
from vision_ai_platform.packages.utils.patches import torch_load

def strip_optimizer(f: str | Path = "best.pt", s: str = "", updates: dict[str, Any] | None = None) -> dict[str, Any]:
    """Strip optimizer from 'f' to finalize training, optionally save as 's'.

    Args:
        f (str | Path): File path to model to strip the optimizer from.
        s (str, optional): File path to save the model with stripped optimizer to. If not provided, 'f' will be
            overwritten.
        updates (dict, optional): A dictionary of updates to overlay onto the checkpoint before saving.

    Returns:
        (dict): The combined checkpoint dictionary.
    """
    try:
        x = torch_load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"
        assert "model" in x, "'model' missing from checkpoint"
    except Exception as e:
        LOGGER.warning(f"Skipping {f}, not a valid Ultralytics model: {e}")
        return {}

    metadata = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }

    # Update model
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with EMA

    # Unwrap DistillationModel to save only the student model
    from vision_ai_platform.packages.ai.nn.distill_model import DistillationModel

    if isinstance(x["model"], DistillationModel):
        x["model"]._remove_feature_hooks()
        x["model"] = x["model"].student_model

    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  # convert from IterableSimpleNamespace to dict
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None  # strip loss criterion
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False

    # Update other keys
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}  # combine args
    for k in "optimizer", "best_fitness", "ema", "updates", "scaler":  # keys
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']

    # Save
    combined = {**metadata, **x, **(updates or {})}
    torch.save(combined, s or f)  # combine dicts (prefer to the right)
    mb = os.path.getsize(s or f) / 1e6  # file size
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
    return combined