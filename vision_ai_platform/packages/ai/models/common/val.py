from pathlib import Path
from abc import ABC
from tqdm import tqdm
import json
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from vision_ai_platform.packages.core import BaseValidator, YOLOConfig
from vision_ai_platform.packages.utils import (
    LOGGER,
    callbacks,
    LOCAL_RANK,
    RANK,
    emojis,
    colorstr,
)
from vision_ai_platform.packages.utils.check import check_imgsz
from vision_ai_platform.packages.utils.files import get_save_dir, get_latest_run
from vision_ai_platform.packages.utils.device_utils import (
    smart_inference_mode,
    torch_distributed_zero_first,
    select_device,
    attempt_compile,
    unwrap_model,
    autocast
)
from vision_ai_platform.packages.utils.ops import Profile, linear_sum_assignment
from vision_ai_platform.packages.ai.data.utils import (
    convert_ndjson_to_yolo_if_needed,
    check_cls_dataset,
    check_det_dataset,
)
from vision_ai_platform.packages.ai.nn.autobackend import AutoBackend


class Validator(BaseValidator, ABC):
    def __init__(
        self,
        cfg: YOLOConfig,
        dataloader: Optional[DataLoader] = None,
        save_dir: Optional[str | Path] = None,
        _callbacks: dict | None = None
    ):
        """Initialize a BaseValidator instance.

        Args:
            cfgs (YOLOConfig): Configuration for the validator.
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to be used for validation.
            save_dir (str | Path, optional): Directory to save results.
            _callbacks (dict, optional): Dictionary to store various callback functions.
        """
        super().__init__(cfg, dataloader, save_dir, _callbacks)
        self.save_dir = Path(save_dir) if save_dir else get_save_dir(cfg)
        (self.save_dir / "labels" if self.cfg.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.cfg.conf is None:
            self.cfg.conf = 0.01 if self.cfg.task == "obb" else 0.001  # reduce OBB val memory usage
        self.cfg.imgsz = check_imgsz(self.cfg.imgsz, max_dim=1)

        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Keep training validation read-only: inputs may be fp16, but EMA/model weights stay fp32 under autocast.
            self.args.quantize = 16 if (self.device.type != "cpu" and trainer.amp) else None
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod  # validate non-compiled original model to avoid issues
            model = model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            if hasattr(model, "end2end"):
                if self.args.end2end is not None:
                    model.end2end = self.args.end2end
                if model.end2end:
                    model.set_head_attr(max_det=self.args.max_det, agnostic_nms=self.args.agnostic_nms)
            with torch_distributed_zero_first(LOCAL_RANK):
                self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data)
            model = AutoBackend(
                model=model or self.args.model,
                # DDP ranks reuse the device assigned in trainer._setup_ddp() via torch.cuda.set_device()
                device=select_device(self.args.device)
                if RANK == -1
                else torch.device("cuda", torch.cuda.current_device()),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.quantize == 16,
            )
            self.device = model.device  # update device
            self.args.quantize = 16 if model.fp16 else None  # record actual inference precision
            stride, fmt = model.stride, model.format
            pt = fmt == "pt"
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if fmt not in {"pt", "torchscript"} and not getattr(model, "dynamic", False):
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            elif str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
                "semantic",
            }:
                self.data = check_det_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not (pt or (getattr(model, "dynamic", False) and fmt != "imx")):
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device, mode=self.args.compile)
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = tqdm(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            with autocast(self.training and self.args.quantize == 16, device=self.device.type):
                # Inference
                with dt[1]:
                    preds = model(batch["img"], augment=augment)

                # Loss
                with dt[2]:
                    if self.training:
                        self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = {}
        self.gather_stats()
        if RANK in {-1, 0}:
            stats = self.get_stats()
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
            self.finalize_metrics()
            self.print_results()
            self.run_callbacks("on_val_end")

        if self.training:
            # Reduce loss across all GPUs
            loss = self.loss.clone().detach()
            if trainer.world_size > 1:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
            if RANK > 0:
                return
            results = {**stats, **trainer.label_loss_items(loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            if RANK > 0:
                return stats
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """Match predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
            use_scipy (bool, optional): Whether to use Hungarian one-to-one matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = linear_sum_assignment(-cost_matrix)  # negate to maximize IoU
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)