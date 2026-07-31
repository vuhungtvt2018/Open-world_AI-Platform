from vision_ai_platform.packages.core.model import BaseTrainer
from vision_ai_platform.packages.core.config import TrainerConfig
from vision_ai_platform.packages.ai.data import YOLODataset
from vision_ai_platform.packages.utils import LOGGER, colorstr, emojis, RANK
from vision_ai_platform.packages.utils.files import get_latest_run
from vision_ai_platform.packages.utils.check import clean_url, normalize_platform_uri, check_imgsz, check_file
from vision_ai_platform.packages.utils.device_utils import (
    parse_device,
    select_device,
    autocast,
    unwrap_model,
    ModelEMA,
    EarlyStopping,
    TORCH_2_4,
)
from vision_ai_platform.packages.utils.check import check_amp
from vision_ai_platform.packages.ai.optim import MuSGD
from vision_ai_platform.packages.ai.nn.tasks import load_checkpoint
from vision_ai_platform.packages.ai.nn.distill_model import DistillationModel
from vision_ai_platform.packages.utils.device_utils import torch_distributed_zero_first

from typing import Any, Optional, Dict
from functools import partial
import time
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Return a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf.

    Args:
        y1 (float, optional): Initial value.
        y2 (float, optional): Final value.
        steps (int, optional): Number of steps.

    Returns:
        (function): Lambda function for computing the sinusoidal ramp.
    """
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1


class YOLOTrainer(BaseTrainer):
    optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSprop", "SGD", "MuSGD", "auto"}

    def build_dataset(self, data_path: str, mode: str = "train", imgsz: int = 640) -> Dataset:
        """Constructs and returns PyTorch DataLoader for training/validation splits."""
        is_train = mode == "train"
        dataset = YOLODataset(
            img_path=data_path,
            imgsz=imgsz,
            augment=is_train,
            hyp=self.cfg,
            rect=not is_train,
        )
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        data = YOLODataset(
            img_path=data_path,
            imgsz=imgsz,
            batch_size=self.cfg.batch_size,
            augment=is_train,
            hyp=self.cfg,
            rect=not is_train,
            single_cls=self.cfg.single_cls or False,
            stride=gs,
            pad=0.0 if is_train else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.cfg.task,
            classes=self.cfg.classes,
            data=data,
            fraction=self.cfg.fraction if is_train else 1.0,
        )

        return dataset

    def get_dataloader(self, dataset_path, imgsz=640, batch_size=16, rank=0, mode="train", workers: int=4):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, imgsz)
        is_train = mode == "train"
        if getattr(dataset, "rect", False) and is_train and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            is_train = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=workers if is_train else workers*2,
            rank=rank,
            pin_memory=True,
            collate_fn=YOLODataset.collate_fn,
            drop_last=self.cfg.compile and is_train,
        )

    def build_optimizer(self) -> optim.Optimizer:
        """Initialize optimizer based on self.cfg parameters."""
        g = [{}, {}, {}, {}]  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        
        name = {x.lower(): x for x in self.optimizers}.get(str(self.cfg.optimizer).lower(), str(self.cfg.optimizer))
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.cfg.lr0}' and 'momentum={self.cfg.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr fit equation to 6 decimal places
            name, lr, momentum = ("MuSGD", 0.01, 0.9) if self.epochs > 10000 else ("AdamW", lr_fit, 0.9)
            self.cfg.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        use_muon = name == "MuSGD"
        for module_name, module in unwrap_model(self.model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if param.ndim in {2, 4} and use_muon:  # muon only orthogonalizes matrices and conv filters
                    g[3][fullname] = param  # muon params
                elif "bias" in fullname:  # bias (no decay)
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1][fullname] = param
                else:  # weight (with decay)
                    g[0][fullname] = param
        if not use_muon:
            g = [x.values() for x in g[:3]]  # convert to list of params

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = {"lr": lr, "betas": (momentum, 0.999), "weight_decay": 0.0}
        elif name == "RMSprop":
            optim_args = {"lr": lr, "momentum": momentum}
        elif name == "SGD" or name == "MuSGD":
            optim_args = {"lr": lr, "momentum": momentum, "nesterov": True}
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {self.optimizers}. "
                "Request support for additional optimizers at https://github.com/ultralytics/ultralytics."
            )

        num_params = [len(g[0]), len(g[1]), len(g[2])]  # number of param groups
        g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
        g[0] = {"params": g[0], **optim_args, "weight_decay": self.cfg.weight_decay, "param_group": "weight"}
        g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
        muon, sgd = (0.2, 1.0)
        if use_muon:
            num_params[0] = len(g[3])  # update number of params
            g[3] = {"params": g[3], **optim_args, "weight_decay": self.cfg.weight_decay, "use_muon": True, "param_group": "muon"}
            import re

            # higher lr for certain parameters in MuSGD when finetuning
            # proto.semseg is the checkpoint parameter name for YOLO26 semantic auxiliary heads.
            pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg|SemanticSegment")
            g_ = []  # new param groups
            for x in g:
                p = x.pop("params")
                p1 = [v for k, v in p.items() if pattern.search(k)]
                p2 = [v for k, v in p.items() if not pattern.search(k)]
                g_.extend([{"params": p1, **x, "lr": lr * 3}, {"params": p2, **x}])
            g = g_
        optimizer = (partial(MuSGD, muon=muon, sgd=sgd) if use_muon else getattr(optim, name))(params=g)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={self.cfg.weight_decay}), {num_params[2]} bias(decay=0.0)"
        )
        return optimizer

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.cfg.cos_lr:
            self.lf = one_cycle(1, self.cfg.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.cfg.epochs, 0) * (1.0 - self.cfg.lrf) + self.cfg.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _get_warmup_iterations(self, num_batches):
        """Return warmup iterations, leaving at least the final epoch for regular training."""
        warmup_epochs = min(self.cfg.warmup_epochs, max(self.epochs - 1, 0))
        return round(warmup_epochs * num_batches) if warmup_epochs > 0 else 0

    def train_one_epoch(self, dataloader: Any) -> None:
        """Run a single epoch training loop over the dataloader."""
        self.model.train()
        self.model.to(self.device)

        num_batches = len(dataloader)
        nw = self._get_warmup_iterations(num_batches)

        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )

        # Initialize loss trackers
        running_losses = None

        for i, batch in enumerate(dataloader):
            ni = i + num_batches * self.epoch  # Total iterations accumulated

            # Warmup learning rate schedule
            if ni <= nw:
                xi = [0, nw]  # x interpolation points
                for j, x in enumerate(self.optimizer.param_groups):
                    # Bias LR warms up differently from weights
                    is_bias = x.get("param_group") == "bias"
                    lr_target = (
                        self.cfg.warmup_bias_lr
                        if is_bias
                        else x["initial_lr"] if "initial_lr" in x else self.cfg.lr0
                    )
                    x["lr"] = np.interp(ni, xi, [lr_target if is_bias else 0.0, x.get("lr", self.cfg.lr0)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(
                            ni, xi, [getattr(self.cfg, "warmup_momentum", 0.8), self.cfg.momentum]
                        )

            # Prepare data inputs
            imgs = batch["img"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)

            # Forward pass with Automatic Mixed Precision (AMP)
            self.optimizer.zero_grad()
            with autocast(enabled=(self.device.type != "cpu")):
                preds = self.model(imgs)
                
                # Assume loss_fn returns (total_loss, loss_items_tensor_or_dict)
                if hasattr(self.model, "criterion") and self.model.criterion is not None:
                    loss, loss_items = self.model.criterion(preds, targets)
                else:
                    # Generic dummy fallback if criterion attached directly
                    loss, loss_items = preds.sum(), torch.tensor([preds.sum().item()], device=self.device)

            # Backward pass & Gradient Scaling
            self.scaler.scale(loss).backward()

            # Clip gradients if required
            if hasattr(self.cfg, "max_grad_norm"):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update Exponential Moving Average (EMA) of model weights
            if self.ema:
                self.ema.update(self.model)

            # Track average loss statistics
            if running_losses is None:
                running_losses = (
                    loss_items.detach().clone()
                    if isinstance(loss_items, torch.Tensor)
                    else torch.tensor(list(loss_items.values()), device=self.device)
                )
            else:
                current_items = (
                    loss_items.detach()
                    if isinstance(loss_items, torch.Tensor)
                    else torch.tensor(list(loss_items.values()), device=self.device)
                )
                running_losses = (running_losses * i + current_items) / (i + 1)

            # Log step metrics
            if i % getattr(self.cfg, "log_interval", 10) == 0:
                mem = f"{torch.cuda.memory_reserved() / 1E9:.2f}G" if torch.cuda.is_available() else "0G"
                LOGGER.info(
                    f"Epoch {self.epoch + 1}/{self.epochs} | Batch {i}/{num_batches} | "
                    f"Loss: {loss.item():.4f} | GPU Mem: {mem}"
                )

        # Update learning rate scheduler after completing epoch
        if self.scheduler:
            self.scheduler.step()

        self.loss = loss.item()
        self.tloss = running_losses

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Saves current training state and model weights to disk."""
        self.weight_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": (self.ema.ema if self.ema else unwrap_model(self.model)).state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save last checkpoint
        torch.save(ckpt, self.last)

        # Save best checkpoint
        if is_best:
            torch.save(ckpt, self.best)
            LOGGER.info(f"{colorstr('Save:')} Updated best checkpoint to {self.best}")

    def train(self, data_path: str) -> Dict[str, Any]:
        """Runs full training pipeline with dataset building, evaluation, and checkpointing."""
        self.data_path = data_path
        self.weight_dir.mkdir(parents=True, exist_ok=True)
        train_loader = self.build_dataset(data_path, mode="train")
        self.optimizer = self.build_optimizer()
        self._setup_scheduler()

        self.ema = ModelEMA(self.model)

        self.best_fitness = 0.0

        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_one_epoch(train_loader)

            # Compute fitness / Validation step
            current_fitness = 1.0 / (self.loss + 1e-6)  # Basic loss-based fitness fallback
            self.fitness = current_fitness

            is_best = current_fitness > self.best_fitness
            if is_best:
                self.best_fitness = current_fitness

            # Save checkpoints
            self.save_checkpoint(is_best=is_best)

        return {
            "status": "success",
            "epochs_completed": self.epochs,
            "best_fitness": self.best_fitness,
        }

    def _setup_train(self):
        """Configure model, optimizer, dataloaders, and training utilities before the training loop."""
        self.weight_dir.mkdir(parents=True, exist_ok=True)
        train_loader = self.get_dataloader(self.data_path, batch_size=self.cfg.batch_size, rank=RANK, mode="train")
        self.optimizer = self.build_optimizer()
        self._setup_scheduler()

        # Freeze layers
        freeze_list = (
            self.cfg.freeze if isinstance(self.cfg.freeze, list)
            else range(self.cfg.freeze) if isinstance(self.cfg.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        if isinstance(unwrap_model(self.model), DistillationModel):
            freeze_layer_names.append("teacher_model.")
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True
        if not any(v.requires_grad for v in self.model.parameters()):
            raise RuntimeError(
                f"'freeze={self.cfg.freeze}' froze the entire model with no trainable parameters left. "
                f"Reduce 'freeze' or pass a list of specific layer indices."
            )

        # Check AMP
        self.amp = torch.tensor(self.cfg.amp).to(self.device)  # True or False
        if RANK > -1 and self.world_size > 1:  # DDP
            self.amp = self.amp.int()  # gloo errors with boolean
            dist.broadcast(self.amp, src=0)  # broadcast from rank 0 to all other ranks
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # resume training would directly load DistillationModel so check here
        if self.args.distill_model is not None and not isinstance(unwrap_model(self.model), DistillationModel):
            self.model = DistillationModel(student_model=self.model, teacher_model=self.args.distill_model)

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()
        self._build_train_pipeline()
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        self.set_class_weights()  # compute class weights after dataloader is ready
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        self.stopper, self.stop = EarlyStopping(patience=self.cfg.patience), False
        self.scheduler.last_epoch = max(self.epoch - 1, 0)  # do not move

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.cfg.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())
                ckpt_args = load_checkpoint(last)[0].args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.cfg = get_cfg(ckpt_args)
                self.cfg.model = self.cfg.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                    "augmentations",
                    "save_period",
                    "workers",
                    "cache",
                    "patience",
                    "time",
                    "freeze",
                    "val",
                    "plots",
                    "distill_model",
                    "save_dir",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.cfg, k, overrides[k])

                # Handle augmentations parameter for resume: check if user provided custom augmentations
                if ckpt_args.get("augmentations") is not None:
                    # Augmentations were saved in checkpoint as reprs but can't be restored automatically
                    LOGGER.warning(
                        "Custom Albumentations transforms were used in the original training run but are not "
                        "being restored. To preserve custom augmentations when resuming, you need to pass the "
                        "'augmentations' parameter again to get expected results. Example: \n"
                        f"model.train(resume=True, augmentations={ckpt_args['augmentations']})"
                    )

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume