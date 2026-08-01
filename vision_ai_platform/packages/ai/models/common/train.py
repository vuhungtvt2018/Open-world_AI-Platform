from pathlib import Path
from abc import ABC
from copy import deepcopy, copy
import subprocess
import math
import time
import warnings
from tqdm import tqdm
from datetime import datetime
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from vision_ai_platform import __version__
from vision_ai_platform.packages.core import BaseTrainer, YOLOConfig
from vision_ai_platform.packages.utils import (
    callbacks,
    LOGGER,
    RANK,
    YAML,
    LOCAL_RANK,
    colorstr,
    GIT,
    emojis,
    clean_url
)
from vision_ai_platform.packages.utils.converter import convert_ndjson_to_yolo_if_needed
from vision_ai_platform.packages.utils.device_utils import (
    parse_device,
    select_device,
    init_seeds,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
    autocast,
    check_train_batch_size,
    convert_optimizer_state_dict_to_fp16,
    ModelEMA,
)
from vision_ai_platform.packages.utils.check import (
    print_args,
    check_model_file_from_stem,
    check_file,    
)
from vision_ai_platform.packages.utils.dist import generate_ddp_command, ddp_cleanup
from vision_ai_platform.packages.utils.plotting import plot_results
from vision_ai_platform.packages.utils.files import get_latest_run, get_save_dir
from vision_ai_platform.packages.ai.nn import load_checkpoint
from vision_ai_platform.packages.ai.optim.utils import strip_optimizer
from vision_ai_platform.packages.ai.data.utils import check_cls_dataset, check_det_dataset
from vision_ai_platform.packages.ai.nn.distill_model import DistillationModel
from vision_ai_platform.packages.ai.optim import MuSGD


class Trainer(BaseTrainer, ABC):
    def __init__(
        self,
        cfg: YOLOConfig,
        save_dir: Optional[str | Path] = None,
        _callbacks: dict | None = None
    ):
        """Initialize the BaseTrainer class.

        Args:
            cfg (YOLOConfig): Configuration for the trainer.
            save_dir (str | Path, optional): Directory path to save training results.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        super().__init__(cfg, save_dir, _callbacks)
        self.cfg.device = parse_device(self.cfg.device)
        self.device = select_device(self.cfg.device)
        init_seeds(self.cfg.seed + 1 + RANK, deterministic=self.cfg.deterministic)

        # Dirs
        if not self.save_dir:
            self.save_dir = get_save_dir(self.cfg)
        self.cfg.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.cfg.save_dir = str(self.save_dir)
            # Save run args, serializing augmentations as reprs for resume compatibility
            args_dict = vars(self.cfg).copy()
            if args_dict.get("augmentations") is not None:
                # Serialize Albumentations transforms as their repr strings for checkpoint compatibility
                args_dict["augmentations"] = [repr(t) for t in args_dict["augmentations"]]
            YAML.save(self.save_dir / "cfg.yaml", args_dict)  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.cfg.save_period

        if RANK == -1:
            print_args(vars(self.cfg))

        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        # Run on_pretrain_routine_start before get_dataset() to capture original args.data (e.g., ul:// URIs)
        if RANK in {-1, 0} and not self.ddp:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_pretrain_routine_start")

        self.model = check_model_file_from_stem(self.cfg.model)  # add suffix, i.e. yolo26n -> yolo26n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.data = self.get_dataset()

        # CSV results file
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.cfg.resume:
            self.csv.unlink()

    def train(self):
        """Execute the training process, using DDP subprocess for multi-GPU or direct training for single-GPU."""
        # Run subprocess if DDP training, else train normally
        if self.ddp:
            # Argument checks
            if self.cfg.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.cfg.rect = False
            if self.cfg.batch < 1.0:
                raise ValueError(
                    "AutoBatch with batch<1 not supported for Multi-GPU training, "
                    f"please specify a valid batch size multiple of GPU count {self.world_size}, i.e. batch={self.world_size * 8}."
                )

            # Command
            cmd, file = None, None
            try:
                cmd, file = generate_ddp_command(self)
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                if file is not None:
                    ddp_cleanup(self, str(file))

        else:
            self._do_train()

    def _do_train(self):
        """Perform the full training loop including setup, epoch iteration, validation, and final evaluation."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.cfg.warmup_epochs * nb), 100) if self.cfg.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.cfg.imgsz} train, {self.cfg.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.cfg.time} hours..." if self.cfg.time else f"{self.epochs} epochs...")
        )
        if self.cfg.close_mosaic:
            base_idx = (self.epochs - self.cfg.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        self._oom_retries = 0  # OOM auto-reduce counter for first epoch
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.cfg.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.cfg.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = float(
                            np.interp(
                                ni,
                                xi,
                                [
                                    self.cfg.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                    x["initial_lr"] * self.lf(epoch),
                                ],
                            )
                        )
                        if "momentum" in x:
                            x["momentum"] = float(np.interp(ni, xi, [self.cfg.warmup_momentum, self.cfg.momentum]))

                # Forward
                try:
                    with autocast(self.amp):
                        batch = self.preprocess_batch(batch)
                        if self.cfg.compile:
                            # Decouple inference and loss calculations for improved compile performance
                            preds = self.model(batch["img"])
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                        else:
                            loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= self.world_size
                        self.tloss = (
                            self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                        )

                    # Backward
                    self.scaler.scale(self.loss).backward()
                except RuntimeError as e:
                    is_oom = "out of memory" in str(e).lower()  # torch.cuda.OutOfMemoryError requires torch>=1.13
                    if not is_oom and not any(
                        s in str(e) for s in ("CUDNN_STATUS_INTERNAL_ERROR", "unable to find an engine")
                    ):
                        raise
                    if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                        raise  # only auto-reduce during first epoch on single GPU, max 3 retries
                    self._oom_retries += 1
                    old_batch = self.batch_size
                    self.cfg.batch = self.batch_size = max(self.batch_size // 2, 1)
                    LOGGER.warning(
                        f"{'CUDA out of memory' if is_oom else 'CUDA backend memory error'} with batch={old_batch}. "
                        f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                    )
                    batch = loss = preds = None
                    self.loss = self.loss_items = self.tloss = None
                    self._clear_memory()
                    self._build_train_pipeline()  # rebuild dataloaders, optimizer, scheduler
                    self.scheduler.last_epoch = self.start_epoch - 1
                    nb = len(self.train_loader)
                    nw = max(round(self.cfg.warmup_epochs * nb), 100) if self.cfg.warmup_epochs > 0 else -1
                    last_opt_step = -1
                    self.optimizer.zero_grad()
                    break  # restart epoch loop with reduced batch size
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.cfg.time:
                        self.stop = (time.time() - self.train_time_start) > (self.cfg.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch.get("cls", batch["img"]).shape[0],  # no. of instances
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.cfg.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break  # allow external stop (e.g. platform cancellation) between batches
            else:
                # for/else: this block runs only when the for loop completes without break (no OOM retry)
                self._oom_retries = 0  # reset OOM counter after successful first epoch

            if self._oom_retries and not self.stop:
                continue  # OOM recovery broke the for loop, restart with reduced batch size

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            if self.cfg.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(None if self.device.type == "mps" else 0.5)  # prevent VRAM spike
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                continue

            self.nan_recovery_attempts = 0
            if RANK in {-1, 0}:
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.cfg.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.cfg.time * 3600)

                # Save model
                if (self.cfg.save or final_epoch) and self.save_model():
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.cfg.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.cfg.epochs = math.ceil(self.cfg.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            # clear if memory utilization > 50%; always clear on MPS due to leak https://github.com/ultralytics/ultralytics/issues/22621
            self._clear_memory(None if self.device.type == "mps" else 0.5)

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.cfg.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        for loader in (self.train_loader, self.test_loader):
            if hasattr(loader, "close"):
                loader.close()  # shut down persistent dataloader workers so none survive to interpreter exit
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0, dataset_size=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        max_imgsz = int(self.cfg.imgsz * (1 + self.cfg.multi_scale))  # need not be stride-aligned
        return check_train_batch_size(
            model=self.model,
            imgsz=max_imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
            dataset_size=dataset_size,
        )  # returns batch size

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # A transient NaN/Inf permanently poisons the EMA running average (ema = decay*ema + (1-decay)*model), so
        # save_model would otherwise skip every epoch and the run would finish with no checkpoint on valid input.
        # Resync each poisoned EMA tensor from the live model where finite; any tensor that is non-finite in both is
        # left for the nan_to_num_ pass below, so a usable checkpoint is always written.
        ema = unwrap_model(self.ema.ema)
        if not all(torch.isfinite(v).all() for v in ema.state_dict().values() if isinstance(v, torch.Tensor)):
            model_sd = unwrap_model(self.model).state_dict()
            for k, v in ema.state_dict().items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all() and torch.isfinite(model_sd[k]).all():
                    v.copy_(model_sd[k])
        ema = deepcopy(ema).half()
        if hasattr(ema, "criterion"):
            ema.criterion = None  # strip training-only state from the serialization snapshot
        # Clamp fp16 serialization overflow without mutating the live EMA.
        for v in ema.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v)

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.cfg),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "message": GIT.message,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.wdir.mkdir(parents=True, exist_ok=True)  # ensure weights directory exists
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        return True

    def get_dataset(self):
        """Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            self.cfg.data = convert_ndjson_to_yolo_if_needed(self.cfg.data)

            # Task-specific dataset checking
            if self.cfg.task == "classify":
                data = check_cls_dataset(self.cfg.data)
            elif str(self.cfg.data).rsplit(".", 1)[-1] in {"yaml", "yml"} or self.cfg.task in {
                "detect",
                "segment",
                "pose",
                "obb",
                "semantic",
            }:
                data = check_det_dataset(self.cfg.data)
                if "yaml_file" in data:
                    self.cfg.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.cfg.data)}' error ❌ {e}")) from e
        if self.cfg.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """Load, create, or download model for any task.

        Returns:
            (dict | None): Checkpoint to resume training from, or None if no checkpoint is loaded.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)
            cfg = weights.yaml
        if isinstance(self.cfg.pretrained, (str, Path)):
            weights, _ = load_checkpoint(self.cfg.pretrained)
        elif self.cfg.pretrained is False and not self.resume:
            weights = None

        # rebuild DistillationModel from resuming checkpoint
        if isinstance(weights, DistillationModel):
            if RANK in {-1, 0}:
                LOGGER.info("Resuming training DistillationModel from checkpoint weights")
            student_model = self.get_model(cfg=cfg, weights=weights.student_model, verbose=RANK in {-1, 0})
            student_model.args = self.cfg
            # teacher is stripped from the checkpoint to save memory/disk; rebuild it from the distill_model path
            teacher_model = weights.teacher_model if weights.teacher_model is not None else self.cfg.distill_model
            model = DistillationModel(student_model=student_model, teacher_model=teacher_model)
            if getattr(weights, "projector", None) is not None:
                model.projector.load_state_dict(weights.projector.state_dict())  # restore the trained projector
            model.criterion = None
            self.model = model
        else:
            self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK in {-1, 0})  # calls Model(cfg, weights)
        return ckpt

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def final_eval(self):
        """Perform final evaluation and validation for the YOLO model."""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):  # strip only on GPU 0; other GPUs should wait
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    # update best.pt train_metrics from last.pt
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model}...")
            self.validator.args.plots = self.cfg.plots
            self.validator.args.compile = False  # disable final val compile as too slow
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.epoch += 1  # log best metrics at step epochs+1, not overwriting last epoch
            self.run_callbacks("on_fit_epoch_end")
            self.epoch -= 1  # restore epoch

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.cfg.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())
                ckpt_args = load_checkpoint(last)[0].args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.cfg.data

                resume = True
                self.cfg = YOLOConfig(**ckpt_args)
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

    def _load_checkpoint_state(self, ckpt):
        """Load optimizer, scaler, EMA, and best_fitness from checkpoint."""
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema and ckpt.get("ema"):
            self.ema = ModelEMA(self.model)  # validation with EMA creates inference tensors that can't be updated
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]
        self.best_fitness = ckpt.get("best_fitness")

    def _handle_nan_recovery(self, epoch):
        """Detect and recover from NaN/Inf loss and fitness collapse by loading last checkpoint."""
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        fitness_collapse = self.best_fitness and self.best_fitness > 0 and self.fitness == 0
        corrupted = RANK in {-1, 0} and (loss_nan or fitness_nan or fitness_collapse)
        reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf" if fitness_nan else "Fitness collapse"
        if RANK != -1:  # DDP: broadcast to all ranks
            broadcast_list = [corrupted if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            corrupted = broadcast_list[0]
        if not corrupted:
            return False
        if epoch == self.start_epoch:
            LOGGER.warning(f"{reason} detected but can not recover from last.pt...")
            return False  # Cannot recover on first epoch, let training continue
        if not self.last.exists():
            raise RuntimeError(f"{reason} detected but no valid last.pt is available for recovery")
        self.nan_recovery_attempts += 1
        if self.nan_recovery_attempts > 3:
            raise RuntimeError(f"Training failed: NaN persisted for {self.nan_recovery_attempts} epochs")
        LOGGER.warning(f"{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt...")
        self._model_train()  # set model to train mode before loading checkpoint to avoid inference tensor errors
        _, ckpt = load_checkpoint(self.last)
        ema = ckpt["ema"].float()
        ema_state = ema.state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
        model = unwrap_model(self.model)
        if hasattr(model, "student_model"):
            # Distillation: the EMA is stripped of the teacher (rebuilt from the distill_model path), so only the
            # student and projector are restored; loading them separately keeps a strict key match.
            model.student_model.load_state_dict(ema.student_model.state_dict())
            model.projector.load_state_dict(ema.projector.state_dict())
        else:
            model.load_state_dict(ema_state)  # Load EMA weights into model
        self._load_checkpoint_state(ckpt)  # Load optimizer/scaler/EMA/best_fitness
        del ckpt, ema, ema_state
        self.scheduler.last_epoch = epoch - 1
        return True

    def resume_training(self, ckpt):
        """Resume YOLO training from a given checkpoint."""
        if ckpt is None or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        assert 0 < start_epoch < self.epochs, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self._load_checkpoint_state(ckpt)
        if getattr(unwrap_model(self.model), "end2end", False):
            # initialize loss and resume o2o and o2m args
            unwrap_model(self.model).criterion = unwrap_model(self.model).init_criterion()
            unwrap_model(self.model).criterion.updates = start_epoch - 1
            unwrap_model(self.model).criterion.update()
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected based on the
                number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if name is 'auto'.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [{}, {}, {}, {}]  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        use_muon = name == "MuSGD"
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if param.ndim >= 2 and use_muon:
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

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(str(name).lower(), str(name))
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optim_args = dict(lr=lr, momentum=momentum)
        elif name == "SGD" or name == "MuSGD":
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for additional optimizers at https://github.com/ultralytics/ultralytics."
            )

        num_params = [len(g[0]), len(g[1]), len(g[2])]  # number of param groups
        g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
        g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
        g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
        muon, sgd = (0.2, 1.0)
        if use_muon:
            num_params[0] = len(g[3])  # update number of params
            g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
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
        optimizer = getattr(optim, name, partial(MuSGD, muon=muon, sgd=sgd))(params=g)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)"
        )
        return optimizer