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

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    summary_path: Optional[Path] = None


class YoloGridAssigner:
    """Anchor-aware assigner that mirrors YOLO's multi-scale target encoding."""

    def __init__(self, iou_threshold: float = 0.25) -> None:
        self.iou_threshold = float(iou_threshold)

    def __call__(self, outputs: Dict[str, Any], targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        predictions = outputs.get("predictions")
        anchors = outputs.get("anchors")
        feature_shapes = outputs.get("feature_shapes")
        strides = outputs.get("strides")
        if predictions is None or anchors is None or feature_shapes is None or strides is None:
            raise ValueError("Model outputs must include predictions, anchors, feature_shapes, and strides")

        anchor_groups = [torch.as_tensor(anchor, dtype=torch.float32).cpu() for anchor in anchors]
        feature_dims: List[Tuple[int, int]] = [
            (int(shape[0]), int(shape[1])) for shape in feature_shapes
        ]
        strides_tensor = torch.as_tensor(list(strides), dtype=torch.float32)

        if len(predictions) != len(anchor_groups):
            raise ValueError("Mismatch between predictions and anchor groups")

        assigned_targets: List[Dict[str, Any]] = []
        for target in targets:
            assigned_targets.append(
                self._assign_single(target, anchor_groups, feature_dims, strides_tensor)
            )
        return assigned_targets

    def _assign_single(
        self,
        target: Dict[str, Any],
        anchors: Sequence[torch.Tensor],
        feature_shapes: Sequence[Tuple[int, int]],
        strides: torch.Tensor,
    ) -> Dict[str, Any]:
        assigned = dict(target)
        defaults = {
            "assigned_scales": [],
            "assigned_anchors": [],
            "assigned_grid_xy": [],
            "assigned_boxes": [],
            "assigned_labels": [],
        }

        boxes = target.get("boxes")
        labels = target.get("labels")
        if boxes is None or labels is None:
            assigned.update(defaults)
            return assigned

        boxes_tensor = _to_float_tensor(boxes)
        labels_tensor = _to_long_tensor(labels)
        limit = min(boxes_tensor.shape[0], labels_tensor.numel())
        if limit == 0:
            assigned.update(defaults)
            return assigned

        image_h, image_w = _resolve_image_size(target)
        image_h = max(image_h, 1.0)
        image_w = max(image_w, 1.0)

        positives: set[Tuple[int, int, int, int]] = set()
        assigned_scales: List[int] = []
        assigned_anchors: List[int] = []
        assigned_grid_xy: List[List[int]] = []
        assigned_boxes: List[List[float]] = []
        assigned_labels: List[int] = []

        for idx in range(limit):
            x1, y1, x2, y2 = [float(value) for value in boxes_tensor[idx].tolist()[:4]]
            label = int(labels_tensor[idx].item())
            width = max(x2 - x1, 1e-6)
            height = max(y2 - y1, 1e-6)
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5

            anchor_candidates = self._select_anchor_candidates(width, height, anchors)
            if not anchor_candidates:
                continue

            for scale_idx, anchor_idx in anchor_candidates:
                if scale_idx >= len(feature_shapes):
                    continue
                grid_h, grid_w = feature_shapes[scale_idx]
                if grid_h <= 0 or grid_w <= 0:
                    continue
                gx = int(center_x / image_w * grid_w)
                gy = int(center_y / image_h * grid_h)
                gx = max(0, min(grid_w - 1, gx))
                gy = max(0, min(grid_h - 1, gy))

                key = (scale_idx, anchor_idx, gy, gx)
                if key in positives:
                    continue
                positives.add(key)

                assigned_scales.append(int(scale_idx))
                assigned_anchors.append(int(anchor_idx))
                assigned_grid_xy.append([int(gy), int(gx)])
                assigned_boxes.append([x1, y1, x2, y2])
                assigned_labels.append(label)

        defaults.update(
            {
                "assigned_scales": assigned_scales,
                "assigned_anchors": assigned_anchors,
                "assigned_grid_xy": assigned_grid_xy,
                "assigned_boxes": assigned_boxes,
                "assigned_labels": assigned_labels,
            }
        )
        assigned.update(defaults)
        return assigned

    def _select_anchor_candidates(
        self,
        width: float,
        height: float,
        anchors: Sequence[torch.Tensor],
    ) -> List[Tuple[int, int]]:
        candidates: List[Tuple[float, int, int]] = []
        for scale_idx, anchor_group in enumerate(anchors):
            if anchor_group.numel() == 0:
                continue
            ious = self._compute_anchor_iou(width, height, anchor_group)
            for anchor_idx, iou_value in enumerate(ious.tolist()):
                candidates.append((float(iou_value), scale_idx, anchor_idx))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected: List[Tuple[int, int]] = []
        for iou_value, scale_idx, anchor_idx in candidates:
            if iou_value >= self.iou_threshold or not selected:
                selected.append((scale_idx, anchor_idx))
            else:
                break
        return selected

    @staticmethod
    def _compute_anchor_iou(width: float, height: float, anchor_group: torch.Tensor) -> torch.Tensor:
        if width <= 0.0 or height <= 0.0 or anchor_group.numel() == 0:
            return torch.zeros(anchor_group.shape[0], dtype=torch.float32)
        box = anchor_group.new_tensor([width, height])
        inter = torch.minimum(box, anchor_group)
        inter_area = inter[:, 0] * inter[:, 1]
        box_area = width * height
        anchor_area = anchor_group[:, 0] * anchor_group[:, 1]
        union = box_area + anchor_area - inter_area + 1e-9
        return inter_area / union


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
            return TrainingResult(metrics=[], checkpoints=[], summary_path=None)

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

            summary_path = self._persist_epoch_metrics(metrics)
            return TrainingResult(metrics=metrics, checkpoints=checkpoint_paths, summary_path=summary_path)

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

    def _persist_epoch_metrics(self, metrics: List[EpochMetrics]) -> Path:
        history_dir = ensure_dir(self.config.artifact_dir / "metadata")
        summary_file = history_dir / "training_history.json"
        payload = [
            {
                "epoch": entry.epoch,
                "train_loss": entry.train_loss,
                "val_loss": entry.val_loss,
            }
            for entry in metrics
        ]
        ensure_dir(summary_file.parent)
        summary_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Persisted training history to %s", summary_file)
        return summary_file


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
