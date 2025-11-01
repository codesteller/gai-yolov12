'''
 #  Copyright (c) 2025 Gahan AI Private Limited
 #  Author: Pallab Maji
 #  Create Time: 2025-11-01 09:10:00
 #  Modified time: 2025-11-01 09:10:00
 #  Description: Training engine for the GAI-YOLOv12 detection model.
 #  Description (Legacy): Provides configuration utilities, training orchestration,
 #       checkpoint management, and metric logging for experiments.
'''

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from models import ModelBundle, create_model
from utils.dataloader import create_dataloaders
from utils.utils import ensure_dir

LOGGER = logging.getLogger("gai_yolov12.train")


@dataclass(frozen=True)
class TrainerConfig:
    """Normalized configuration for the training loop."""

    num_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    gradient_clip_norm: Optional[float] = None
    log_interval: int = 10
    checkpoint_interval: int = 5
    optimizer: str = "adamw"
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    artifact_dir: Path = Path("./experiments/default/")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrainerConfig":
        experiment_cfg = config.get("experiment", {})
        artifact_dir = Path(experiment_cfg.get("artifact_dir", "./experiments/default/")).expanduser()
        gradient_clip = experiment_cfg.get("gradient_clip_norm")
        clip_value = float(gradient_clip) if gradient_clip is not None else None
        log_interval = int(experiment_cfg.get("log_interval", 10))
        checkpoint_interval = int(experiment_cfg.get("checkpoint_interval", 5))
        device = str(experiment_cfg.get("device", "auto"))
        return cls(
            num_epochs=int(experiment_cfg.get("num_epochs", 1)),
            learning_rate=float(experiment_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(experiment_cfg.get("weight_decay", 0.0)),
            device=device,
            gradient_clip_norm=clip_value,
            log_interval=max(1, log_interval),
            checkpoint_interval=max(1, checkpoint_interval),
            optimizer=str(experiment_cfg.get("optimizer", "adamw")),
            scheduler=experiment_cfg.get("scheduler"),
            scheduler_params=dict(experiment_cfg.get("scheduler_params", {})),
            artifact_dir=artifact_dir,
        )

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None


@dataclass
class TrainingResult:
    metrics: List[EpochMetrics]
    checkpoints: List[Path]


class YoloGridAssigner:
    """Assigns targets to anchors using a grid-based heuristic similar to YOLO."""

    def __call__(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        num_anchors = int(outputs.get("pred_logits", torch.empty(0, 1, 1)).shape[1])
        grid_h, grid_w = _resolve_feature_map_shape(outputs)
        assigned_targets: List[Dict[str, Any]] = []

        for target in targets:
            assigned_targets.append(self._assign_single(target, num_anchors, grid_h, grid_w))
        return assigned_targets

    def _assign_single(
        self,
        target: Dict[str, Any],
        num_anchors: int,
        grid_h: int,
        grid_w: int,
    ) -> Dict[str, Any]:
        boxes = target.get("boxes")
        labels = target.get("labels")
        assigned = dict(target)

        if boxes is None or labels is None or grid_h == 0 or grid_w == 0:
            assigned.update(
                {
                    "assigned_anchors": [],
                    "assigned_labels": [],
                    "assigned_boxes": [],
                }
            )
            return assigned

        boxes_tensor = _to_float_tensor(boxes)
        labels_tensor = _to_long_tensor(labels)
        if boxes_tensor.numel() == 0 or labels_tensor.numel() == 0:
            assigned.update(
                {
                    "assigned_anchors": [],
                    "assigned_labels": [],
                    "assigned_boxes": [],
                }
            )
            return assigned

        img_h, img_w = _resolve_image_size(target)
        total_cells = max(1, grid_h * grid_w)
        occupied: set[int] = set()
        anchor_indices: List[int] = []
        assigned_boxes: List[List[float]] = []
        assigned_labels: List[int] = []

        order = torch.argsort(_box_area(boxes_tensor), descending=True)
        for raw_idx in order.tolist():
            if raw_idx >= boxes_tensor.shape[0]:
                continue
            center_x, center_y = _box_center(boxes_tensor[raw_idx])
            col = int(center_x / max(img_w, 1e-6) * grid_w)
            row = int(center_y / max(img_h, 1e-6) * grid_h)
            col = max(0, min(grid_w - 1, col))
            row = max(0, min(grid_h - 1, row))
            primary_idx = row * grid_w + col
            anchor_idx = self._find_available_slot(primary_idx, occupied, total_cells)
            if anchor_idx is None or anchor_idx >= num_anchors:
                continue
            occupied.add(anchor_idx)
            anchor_indices.append(anchor_idx)
            assigned_labels.append(int(labels_tensor[raw_idx].item()))
            assigned_boxes.append(boxes_tensor[raw_idx].tolist())

        assigned.update(
            {
                "assigned_anchors": anchor_indices,
                "assigned_labels": assigned_labels,
                "assigned_boxes": assigned_boxes,
            }
        )
        return assigned

    @staticmethod
    def _find_available_slot(primary_idx: int, occupied: set[int], total_cells: int) -> Optional[int]:
        if primary_idx not in occupied:
            return primary_idx
        for offset in range(total_cells):
            candidate = (primary_idx + offset) % total_cells
            if candidate not in occupied:
                return candidate
        return None


def _resolve_feature_map_shape(outputs: Dict[str, torch.Tensor]) -> tuple[int, int]:
    shape = outputs.get("feature_map_shape")
    if isinstance(shape, torch.Tensor):
        shape_values = shape.detach().cpu().view(-1).tolist()
        if len(shape_values) >= 2:
            return int(shape_values[0]), int(shape_values[1])
    elif isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return int(shape[0]), int(shape[1])

    logits = outputs.get("pred_logits")
    if logits is None:
        return 0, 0
    num_anchors = int(logits.shape[1])
    if num_anchors == 0:
        return 0, 0
    grid_size = int(math.sqrt(num_anchors))
    if grid_size * grid_size == num_anchors:
        return grid_size, grid_size
    return num_anchors, 1


def _to_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _to_long_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.int64)
    return torch.as_tensor(value, dtype=torch.int64)


def _resolve_image_size(target: Dict[str, Any]) -> tuple[float, float]:
    size = target.get("size")
    if size is None:
        size = target.get("orig_size")
    if isinstance(size, torch.Tensor):
        flat = size.detach().cpu().view(-1)
        if flat.numel() >= 2:
            return float(flat[0].item()), float(flat[1].item())
    elif isinstance(size, (list, tuple)) and len(size) >= 2:
        return float(size[0]), float(size[1])
    return 1.0, 1.0


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.ndim == 1:
        boxes = boxes.view(1, -1)
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
    return widths * heights


def _box_center(box: torch.Tensor) -> tuple[float, float]:
    if box.ndim != 1 or box.numel() < 4:
        raise ValueError("Bounding box tensor must be 1-D with four elements")
    x1, y1, x2, y2 = box.tolist()[:4]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return float(cx), float(cy)


class Trainer:
    """Orchestrates the training and evaluation loops."""

    def __init__(
        self,
        model_bundle: ModelBundle,
        dataloaders: Dict[str, DataLoader],
        config: TrainerConfig,
    ) -> None:
        self.config = config
        self.device = config.resolve_device()
        self.model = model_bundle.model.to(self.device)
        self.loss_fn = model_bundle.loss.to(self.device)
        self.dataloaders = dataloaders
        self.optimizer = _build_optimizer(self.model.parameters(), config)
        self.scheduler = _build_scheduler(self.optimizer, config)
        self.assigner = YoloGridAssigner()
        self.checkpoint_dir = ensure_dir(config.artifact_dir / "checkpoints")
        LOGGER.info("Trainer initialized on device: %s", self.device)

    def train(self) -> TrainingResult:
        train_loader = self.dataloaders.get("train")
        if train_loader is None:
            LOGGER.warning("No training dataloader available; aborting training")
            return TrainingResult(metrics=[], checkpoints=[])

        metrics: List[EpochMetrics] = []
        checkpoint_paths: List[Path] = []

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._validate_epoch(epoch)
            metrics.append(EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % self.config.checkpoint_interval == 0:
                checkpoint_paths.append(self._save_checkpoint(epoch, train_loss, val_loss))

        return TrainingResult(metrics=metrics, checkpoints=checkpoint_paths)

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = images.to(self.device)
            targets_on_device = self._move_targets_to_device(targets)
            outputs = self.model(images)
            assigned_targets = self.assigner(outputs, targets_on_device)
            loss = self.loss_fn(outputs, assigned_targets)
            loss.backward()
            if self.config.gradient_clip_norm is not None and self.config.gradient_clip_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            total_batches += 1

            if batch_index % self.config.log_interval == 0:
                LOGGER.info("Epoch %d | Batch %d/%d | loss=%.4f", epoch, batch_index, len(loader), loss.item())

        if total_batches == 0:
            return 0.0
        return total_loss / total_batches

    def _validate_epoch(self, epoch: int) -> Optional[float]:
        val_loader = self.dataloaders.get("val")
        if val_loader is None:
            return None
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets_on_device = self._move_targets_to_device(targets)
                outputs = self.model(images)
                assigned_targets = self.assigner(outputs, targets_on_device)
                loss = self.loss_fn(outputs, assigned_targets)
                total_loss += float(loss.item())
                total_batches += 1
        if total_batches == 0:
            return None
        LOGGER.info("Epoch %d | Validation loss=%.4f", epoch, total_loss / total_batches)
        return total_loss / total_batches

    def _move_targets_to_device(self, targets: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        moved: List[Dict[str, Any]] = []
        for target in targets:
            updated: Dict[str, Any] = {}
            for key, value in target.items():
                if isinstance(value, torch.Tensor):
                    updated[key] = value.to(self.device)
                else:
                    updated[key] = value
            moved.append(updated)
        return moved

    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float]) -> Path:
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        LOGGER.info("Saved checkpoint to %s", checkpoint_path)
        return checkpoint_path


def _build_optimizer(parameters: Iterable[torch.Tensor], config: TrainerConfig) -> torch.optim.Optimizer:
    name = config.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer '{config.optimizer}'")


def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainerConfig) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if config.scheduler is None:
        return None
    name = str(config.scheduler).lower()
    params = dict(config.scheduler_params)
    if name == "steplr":
        step_size = int(params.get("step_size", max(1, config.num_epochs // 3)))
        gamma = float(params.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosineannealing":
        t_max = int(params.get("t_max", config.num_epochs))
        eta_min = float(params.get("eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    raise ValueError(f"Unsupported scheduler '{config.scheduler}'")


def run_training(config: Dict[str, Any]) -> TrainingResult:
    trainer_config = TrainerConfig.from_config(config)
    dataloaders = create_dataloaders(config)
    model_bundle = create_model(config)
    trainer = Trainer(model_bundle, dataloaders, trainer_config)
    return trainer.train()
