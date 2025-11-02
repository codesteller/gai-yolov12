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
import torchvision.transforms.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import cv2
from torchinfo import summary as torchinfo_summary
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import ModelBundle, create_model
from models.targets import YoloScaleTargets, YoloTargets
from models.yolov12 import DetectionScaleOutput
from utils.dataloader import create_dataloaders
from utils.utils import ensure_dir

LOGGER = logging.getLogger("gai_yolov12.train")


def generate_model_summary(model: nn.Module, input_size: Tuple[int, int] = (640, 640)) -> Dict[str, Any]:
    """Generate comprehensive model summary including parameters and layer details."""
    model.eval()
    
    # Basic model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Layer-wise parameter count
    layer_details = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                layer_details.append({
                    "name": name,
                    "type": module.__class__.__name__,
                    "parameters": module_params,
                    "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)
                })
    
    # Try to get model output shape with dummy input
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
            if next(model.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            outputs = model(dummy_input)
            
            if isinstance(outputs, (list, tuple)):
                output_shapes = [tuple(out.raw.shape) if hasattr(out, 'raw') else tuple(out.shape) for out in outputs]
            else:
                output_shapes = [tuple(outputs.shape)]
    except Exception as e:
        LOGGER.warning(f"Could not determine output shapes: {e}")
        output_shapes = ["Unable to determine"]
    
    # Model memory estimation (rough)
    param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    return {
        "model_architecture": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "parameter_size_mb": round(param_size_mb, 2),
        "input_shape": [1, 3, input_size[0], input_size[1]],
        "output_shapes": output_shapes,
        "layer_details": layer_details,
        "backbone_type": model.backbone.__class__.__name__ if hasattr(model, 'backbone') else 'Unknown'
    }


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
    assigner_params: Dict[str, Any] = field(default_factory=dict)
    enable_tensorboard: bool = True
    tensorboard_log_interval: int = 10
    tensorboard_log_images: bool = False
    tensorboard_images_per_batch: int = 2
    tensorboard_image_log_interval: int = 50
    tensorboard_show_ground_truth: bool = True  # Show GT boxes in visualizations
    tensorboard_show_predictions: bool = True   # Show prediction boxes in visualizations
    enable_coco_eval: bool = False
    coco_eval_interval: int = 5
    eval_on_validation: bool = True
    max_batches_per_epoch: Optional[int] = None
    visualization_conf_threshold: float = 0.01  # Confidence threshold for visualization
    use_amp: bool = False  # Enable Automatic Mixed Precision training

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrainerConfig":
        experiment_cfg = config.get("experiment", {})
        
        # Generate artifact directory based on experiment name
        experiment_name = experiment_cfg.get("name", "default-experiment")
        # Convert experiment name to filesystem-safe format
        safe_name = experiment_name.lower().replace(" ", "-").replace("_", "-")
        
        # Use artifact_dir if explicitly provided, otherwise generate from experiment name
        if "artifact_dir" in experiment_cfg:
            artifact_dir = Path(experiment_cfg["artifact_dir"]).expanduser()
        else:
            artifact_dir = Path(f"./experiments/{safe_name}/").expanduser()
        
        gradient_clip = experiment_cfg.get("gradient_clip_norm")
        clip_value = float(gradient_clip) if gradient_clip is not None else None
        log_interval = int(experiment_cfg.get("log_interval", 10))
        checkpoint_interval = int(experiment_cfg.get("checkpoint_interval", 5))
        device = str(experiment_cfg.get("device", "auto"))
        tensorboard_cfg = experiment_cfg.get("tensorboard", {})
        enable_tensorboard = bool(tensorboard_cfg.get("enabled", True))
        tb_log_interval = int(tensorboard_cfg.get("log_interval", log_interval))
        tb_log_images = bool(tensorboard_cfg.get("log_images", False))
        tb_images_per_batch = int(tensorboard_cfg.get("images_per_batch", 2))
        tb_image_log_interval = int(tensorboard_cfg.get("image_log_interval", 50))
        tb_show_ground_truth = bool(tensorboard_cfg.get("show_ground_truth", True))
        tb_show_predictions = bool(tensorboard_cfg.get("show_predictions", True))
        
        # Parse evaluation configuration
        eval_cfg = experiment_cfg.get("evaluation", {})
        enable_coco_eval = bool(eval_cfg.get("enabled", False))
        coco_eval_interval = int(eval_cfg.get("eval_interval", 5))
        eval_on_validation = bool(eval_cfg.get("eval_on_validation", True))
        visualization_conf_threshold = float(eval_cfg.get("visualization_conf_threshold", 0.01))
        
        # Parse max batches per epoch
        max_batches = experiment_cfg.get("max_batches_per_epoch")
        max_batches_per_epoch = int(max_batches) if max_batches is not None else None
        
        # Parse AMP (Automatic Mixed Precision) setting
        use_amp = bool(experiment_cfg.get("use_amp", False))
        
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
            assigner_params=dict(experiment_cfg.get("assigner", {})),
            enable_tensorboard=enable_tensorboard,
            tensorboard_log_interval=max(1, tb_log_interval),
            tensorboard_log_images=tb_log_images,
            tensorboard_images_per_batch=max(1, tb_images_per_batch),
            tensorboard_image_log_interval=max(1, tb_image_log_interval),
            tensorboard_show_ground_truth=tb_show_ground_truth,
            tensorboard_show_predictions=tb_show_predictions,
            enable_coco_eval=enable_coco_eval,
            coco_eval_interval=max(1, coco_eval_interval),
            eval_on_validation=eval_on_validation,
            max_batches_per_epoch=max_batches_per_epoch,
            visualization_conf_threshold=max(0.001, min(1.0, visualization_conf_threshold)),
            use_amp=use_amp,
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
    coco_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    metrics: List[EpochMetrics]
    checkpoints: List[Path]
    summary_path: Optional[Path] = None


class YoloTargetEncoder:
    """Encodes ground-truth boxes into YOLO-style multi-scale tensors."""

    def __init__(
        self,
        *,
        positive_iou_threshold: float = 0.0,
        ignore_iou_threshold: float = 0.6,
    ) -> None:
        if ignore_iou_threshold < positive_iou_threshold:
            raise ValueError("ignore_iou_threshold must be >= positive_iou_threshold")
        self.positive_iou_threshold = float(positive_iou_threshold)
        self.ignore_iou_threshold = float(ignore_iou_threshold)

    @torch.no_grad()
    def __call__(
        self,
        outputs: Sequence[DetectionScaleOutput],
        batch_targets: Sequence[Dict[str, Any]],
    ) -> YoloTargets:
        if not outputs:
            raise ValueError("Model must emit at least one detection scale")
        batch_size = outputs[0].raw.shape[0]
        encoded = self._build_empty_targets(outputs, batch_size)

        if not batch_targets:
            return encoded

        anchor_counts = [scale.anchors.shape[0] for scale in outputs]
        device = outputs[0].raw.device
        dtype = outputs[0].raw.dtype
        anchors_per_scale = [scale.anchors.to(device) for scale in outputs]

        for batch_index, target in enumerate(batch_targets):
            boxes = target.get("boxes")
            labels = target.get("labels")
            if boxes is None or labels is None:
                continue

            boxes_tensor = boxes.to(device=device, dtype=dtype)
            labels_tensor = labels.to(device=device, dtype=torch.long)
            num_entries = min(boxes_tensor.shape[0], labels_tensor.shape[0])
            if num_entries == 0:
                continue

            image_h, image_w = self._resolve_image_size(target, outputs)
            if image_h <= 0.0 or image_w <= 0.0:
                continue

            centers = 0.5 * (boxes_tensor[:num_entries, 0:2] + boxes_tensor[:num_entries, 2:4])
            wh = (boxes_tensor[:num_entries, 2:4] - boxes_tensor[:num_entries, 0:2]).clamp_min(1e-6)

            for obj_idx in range(num_entries):
                center = centers[obj_idx]
                width_height = wh[obj_idx]
                label = int(labels_tensor[obj_idx].item())
                center_x = float(center[0].item())
                center_y = float(center[1].item())

                all_ious: List[torch.Tensor] = []
                for anchor_tensor in anchors_per_scale:
                    ious = self._wh_iou(width_height, anchor_tensor)
                    all_ious.append(ious)

                flat_ious = torch.cat(all_ious)
                best_iou, best_flat_idx = torch.max(flat_ious, dim=0)
                if best_iou.item() < self.positive_iou_threshold:
                    continue

                scale_idx, anchor_idx = self._unflatten_index(int(best_flat_idx.item()), anchor_counts)
                scale_output = outputs[scale_idx]
                targets_for_scale = encoded[scale_idx]

                stride = float(scale_output.stride)
                grid_h, grid_w = scale_output.grid_size
                gx_float = center_x / max(stride, 1e-6)
                gy_float = center_y / max(stride, 1e-6)
                gx = self._clamp_index(int(gx_float), grid_w)
                gy = self._clamp_index(int(gy_float), grid_h)
                offset_x = float(gx_float - gx)
                offset_y = float(gy_float - gy)

                anchor_dims = anchors_per_scale[scale_idx][anchor_idx]
                tw = torch.log(width_height[0] / anchor_dims[0].clamp(min=1e-6) + 1e-9)
                th = torch.log(width_height[1] / anchor_dims[1].clamp(min=1e-6) + 1e-9)

                targets_for_scale.objectness[batch_index, anchor_idx, gy, gx] = 1.0
                targets_for_scale.objectness_iou[batch_index, anchor_idx, gy, gx] = best_iou.clamp(min=0.0, max=1.0)
                targets_for_scale.box[batch_index, anchor_idx, gy, gx, 0] = float(max(min(offset_x, 1.0), 0.0))
                targets_for_scale.box[batch_index, anchor_idx, gy, gx, 1] = float(max(min(offset_y, 1.0), 0.0))
                targets_for_scale.box[batch_index, anchor_idx, gy, gx, 2] = tw
                targets_for_scale.box[batch_index, anchor_idx, gy, gx, 3] = th
                targets_for_scale.class_id[batch_index, anchor_idx, gy, gx] = label
                targets_for_scale.ignore_mask[batch_index, anchor_idx, gy, gx] = 0.0

                for candidate_scale, ious in enumerate(all_ious):
                    stride_candidate = float(outputs[candidate_scale].stride)
                    grid_h_cand, grid_w_cand = outputs[candidate_scale].grid_size
                    gx_candidate = self._clamp_index(int(center_x / max(stride_candidate, 1e-6)), grid_w_cand)
                    gy_candidate = self._clamp_index(int(center_y / max(stride_candidate, 1e-6)), grid_h_cand)

                    for candidate_anchor, iou_value in enumerate(ious):
                        if candidate_scale == scale_idx and candidate_anchor == anchor_idx:
                            continue
                        if float(iou_value.item()) >= self.ignore_iou_threshold:
                            encoded[candidate_scale].ignore_mask[batch_index, candidate_anchor, gy_candidate, gx_candidate] = 1.0

        return encoded

    @staticmethod
    def _build_empty_targets(outputs: Sequence[DetectionScaleOutput], batch_size: int) -> YoloTargets:
        encoded: List[YoloScaleTargets] = []
        for scale_output in outputs:
            _, num_anchors, height, width, _ = scale_output.raw.shape
            device = scale_output.raw.device
            dtype = scale_output.raw.dtype
            encoded.append(
                YoloScaleTargets(
                    objectness=scale_output.raw.new_zeros((batch_size, num_anchors, height, width)),
                    objectness_iou=scale_output.raw.new_zeros((batch_size, num_anchors, height, width)),
                    box=scale_output.raw.new_zeros((batch_size, num_anchors, height, width, 4)),
                    class_id=torch.full((batch_size, num_anchors, height, width), -1, dtype=torch.long, device=device),
                    ignore_mask=scale_output.raw.new_zeros((batch_size, num_anchors, height, width)),
                )
            )
        return encoded

    @staticmethod
    def _resolve_image_size(target: Dict[str, Any], outputs: Sequence[DetectionScaleOutput]) -> Tuple[float, float]:
        size = target.get("size")
        if size is None:
            size = target.get("orig_size")
        if isinstance(size, torch.Tensor) and size.numel() >= 2:
            flat = size.view(-1)
            return float(flat[0].item()), float(flat[1].item())
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return float(size[0]), float(size[1])
        reference = outputs[0]
        ref_h = float(reference.grid_size[0] * reference.stride)
        ref_w = float(reference.grid_size[1] * reference.stride)
        return ref_h, ref_w

    @staticmethod
    def _wh_iou(box_wh: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        box_wh_clamped = box_wh.clamp_min(1e-6)
        inter = torch.minimum(box_wh_clamped, anchors.clamp_min(1e-6))
        inter_area = inter[:, 0] * inter[:, 1]
        box_area = box_wh_clamped[0] * box_wh_clamped[1]
        anchor_area = anchors[:, 0] * anchors[:, 1]
        union = box_area + anchor_area - inter_area + 1e-9
        return inter_area / union

    @staticmethod
    def _unflatten_index(index: int, counts: Sequence[int]) -> Tuple[int, int]:
        offset = 0
        for scale_idx, count in enumerate(counts):
            if index < offset + count:
                return scale_idx, index - offset
            offset += count
        raise IndexError("Anchor index out of bounds for provided scale counts")

    @staticmethod
    def _clamp_index(index: int, max_size: int) -> int:
        if max_size <= 0:
            return 0
        return max(0, min(max_size - 1, index))


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
        assigner_params = dict(config.assigner_params)
        positive_thr = float(assigner_params.get("positive_iou_threshold", 0.0))
        ignore_thr = float(assigner_params.get("ignore_iou_threshold", 0.6))
        self.assigner = YoloTargetEncoder(
            positive_iou_threshold=positive_thr,
            ignore_iou_threshold=ignore_thr,
        )
        self.checkpoint_dir = ensure_dir(config.artifact_dir / "checkpoints")
        
        # Initialize AMP GradScaler if enabled
        self.scaler: Optional[GradScaler] = None
        if config.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler()
            LOGGER.info("Automatic Mixed Precision (AMP) enabled")
        elif config.use_amp and self.device.type != 'cuda':
            LOGGER.warning("AMP requested but CUDA not available. Running in FP32 mode.")
        
        # Initialize TensorBoard logging
        self.writer: Optional[SummaryWriter] = None
        if config.enable_tensorboard:
            tensorboard_dir = ensure_dir(config.artifact_dir / "tensorboard")
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
            LOGGER.info("TensorBoard logging enabled at %s", tensorboard_dir)
        
        # Generate and save model summary
        self._save_model_summary(model_bundle)
        
        # Log model graph to TensorBoard
        if self.writer is not None:
            self._log_model_graph()
        
        LOGGER.info("Trainer initialized on device: %s", self.device)

    def train(self, resume: bool = False) -> TrainingResult:
        train_loader = self.dataloaders.get("train")
        if train_loader is None:
            LOGGER.warning("No training dataloader available; aborting training")
            return TrainingResult(metrics=[], checkpoints=[], summary_path=None)

        metrics: List[EpochMetrics] = []
        checkpoint_paths: List[Path] = []
        summary_path: Optional[Path] = None
        start_epoch = 1

        # Handle resume functionality
        if resume:
            start_epoch, metrics, checkpoint_paths = self._resume_from_checkpoint()

        for epoch in range(start_epoch, self.config.num_epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._validate_epoch(epoch)
            
            # Run COCO evaluation if enabled
            coco_metrics = {}
            if self.config.enable_coco_eval and epoch % self.config.coco_eval_interval == 0:
                coco_metrics = self._evaluate_coco_metrics(epoch)

            metrics.append(EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss, coco_metrics=coco_metrics))

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                if val_loss is not None:
                    self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                
                # Log COCO metrics
                for metric_name, metric_value in coco_metrics.items():
                    self.writer.add_scalar(f'COCO/{metric_name}', metric_value, epoch)
                
                # Log learning rate
                if self.scheduler is not None:
                    current_lr = self.scheduler.get_last_lr()[0]
                else:
                    current_lr = self.config.learning_rate
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % self.config.checkpoint_interval == 0:
                checkpoint_paths.append(self._save_checkpoint(epoch, train_loss, val_loss))
                # Save metrics for resume functionality
                self._save_metrics(metrics)

            summary_path = self._persist_epoch_metrics(metrics)

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            LOGGER.info("TensorBoard logging completed")

        return TrainingResult(metrics=metrics, checkpoints=checkpoint_paths, summary_path=summary_path)

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        
        # Determine effective loader length (limited by max_batches_per_epoch if set)
        effective_loader_length = len(loader)
        if self.config.max_batches_per_epoch is not None:
            effective_loader_length = min(len(loader), self.config.max_batches_per_epoch)
        
        for batch_index, (images, targets) in enumerate(loader, start=1):
            # Break if we've reached the batch limit
            if self.config.max_batches_per_epoch is not None and batch_index > self.config.max_batches_per_epoch:
                LOGGER.info(f"Reached batch limit of {self.config.max_batches_per_epoch} for epoch {epoch}")
                break
                
            images = images.to(self.device)
            targets_on_device = self._move_targets_to_device(targets)
            
            # Use automatic mixed precision if enabled
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    assigned_targets = self.assigner(outputs, targets_on_device)
                    loss = self.loss_fn(outputs, assigned_targets)
                
                self.scaler.scale(loss).backward()
                
                if self.config.gradient_clip_norm is not None and self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            else:
                # Standard FP32 training
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
                LOGGER.info("Epoch %d | Batch %d/%d | loss=%.4f", epoch, batch_index, effective_loader_length, loss.item())
                
                # Log to TensorBoard
                if self.writer is not None and batch_index % self.config.tensorboard_log_interval == 0:
                    global_step = (epoch - 1) * effective_loader_length + batch_index
                    self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
                    
                    # Log images with predictions
                    if (self.config.tensorboard_log_images and 
                        batch_index % self.config.tensorboard_image_log_interval == 0):
                        try:
                            # Get model predictions for visualization
                            self.model.eval()
                            with torch.no_grad():
                                pred_outputs = self.model(images)
                                # Convert outputs to list of dicts format for visualization
                                predictions = self._format_outputs_for_visualization(pred_outputs)
                            self.model.train()
                            
                            # Log images
                            self._log_images_to_tensorboard(
                                images, predictions, targets_on_device, global_step, "train"
                            )
                        except Exception as e:
                            LOGGER.debug(f"Failed to log training images: {e}")

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
        
        # Determine effective validation loader length
        effective_val_length = len(val_loader)
        if self.config.max_batches_per_epoch is not None:
            effective_val_length = min(len(val_loader), self.config.max_batches_per_epoch)
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader, start=1):
                # Break if we've reached the batch limit for validation too
                if self.config.max_batches_per_epoch is not None and batch_idx > self.config.max_batches_per_epoch:
                    break
                    
                images = images.to(self.device)
                targets_on_device = self._move_targets_to_device(targets)
                outputs = self.model(images)
                assigned_targets = self.assigner(outputs, targets_on_device)
                loss = self.loss_fn(outputs, assigned_targets)
                total_loss += float(loss.item())
                total_batches += 1
                
                # Log validation images (only for first batch of each epoch)
                if (batch_idx == 1 and self.writer is not None and 
                    self.config.tensorboard_log_images):
                    try:
                        predictions = self._format_outputs_for_visualization(outputs)
                        global_step = epoch * effective_val_length  # Use epoch-based step for validation
                        self._log_images_to_tensorboard(
                            images, predictions, targets_on_device, global_step, "validation"
                        )
                    except Exception as e:
                        LOGGER.debug(f"Failed to log validation images: {e}")
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

    def _resume_from_checkpoint(self) -> Tuple[int, List[EpochMetrics], List[Path]]:
        """Resume training from the latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("epoch_*.pt"))
        
        if not checkpoints:
            LOGGER.info("No checkpoints found for resume. Starting from scratch.")
            return 1, [], []
        
        # Find latest checkpoint by epoch number
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
        
        LOGGER.info(f"Resuming from checkpoint: {latest_checkpoint}")
        
        # Load checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        
        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore scheduler state if available
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Get starting epoch (next epoch after the checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        
        # Try to load existing metrics and checkpoint paths
        metrics_file = self.checkpoint_dir / "training_metrics.json"
        checkpoint_paths = []
        metrics = []
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    saved_metrics = json.load(f)
                    # Convert back to EpochMetrics objects
                    for metric_data in saved_metrics:
                        metrics.append(EpochMetrics(**metric_data))
                LOGGER.info(f"Loaded {len(metrics)} previous epoch metrics")
            except Exception as e:
                LOGGER.warning(f"Could not load previous metrics: {e}")
        
        # Find all existing checkpoints
        checkpoint_paths = [Path(cp) for cp in sorted(checkpoints)]
        
        LOGGER.info(f"Resuming training from epoch {start_epoch}")
        return start_epoch, metrics, checkpoint_paths

    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float]) -> Path:
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        
        # Save scheduler state if available
        if self.scheduler:
            checkpoint_data["scheduler_state_dict"] = self.scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        LOGGER.info("Saved checkpoint to %s", checkpoint_path)
        return checkpoint_path

    def _save_metrics(self, metrics: List[EpochMetrics]) -> None:
        """Save training metrics to JSON for resume functionality."""
        metrics_file = self.checkpoint_dir / "training_metrics.json"
        try:
            # Convert EpochMetrics to dict for JSON serialization
            metrics_data = []
            for metric in metrics:
                if hasattr(metric, '__dict__'):
                    metrics_data.append(metric.__dict__)
                else:
                    # Handle case where metric might be a dict already
                    metrics_data.append(dict(metric))
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            LOGGER.warning(f"Could not save metrics: {e}")

    def _persist_epoch_metrics(self, metrics: List[EpochMetrics]) -> Path:
        history_dir = ensure_dir(self.config.artifact_dir / "metadata")
        summary_file = history_dir / "training_history.json"
        payload = []
        for entry in metrics:
            metric_entry = {
                "epoch": entry.epoch,
                "train_loss": entry.train_loss,
                "val_loss": entry.val_loss,
            }
            # Add COCO metrics if available
            if entry.coco_metrics:
                metric_entry["coco_metrics"] = entry.coco_metrics
            payload.append(metric_entry)
        ensure_dir(summary_file.parent)
        summary_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Persisted training history to %s", summary_file)
        return summary_file

    def _save_model_summary(self, model_bundle: ModelBundle) -> None:
        """Generate and save comprehensive model summary and parameter details."""
        try:
            metadata_dir = ensure_dir(self.config.artifact_dir / "metadata")
            
            # Generate model summary
            model_summary = generate_model_summary(
                self.model, 
                input_size=getattr(self.model, 'input_size', (640, 640))
            )
            
            # Add model bundle metadata
            model_summary.update({
                "model_bundle_metadata": model_bundle.metadata,
                "device": str(self.device),
                "optimizer": self.config.optimizer,
                "scheduler": self.config.scheduler,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay
            })
            
            # Save model summary
            summary_file = metadata_dir / "model_summary.json"
            summary_file.write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
            LOGGER.info("Saved model summary to %s", summary_file)
            
            # Save detailed parameter breakdown
            param_details = self._generate_parameter_details()
            param_file = metadata_dir / "model_parameters.json"
            param_file.write_text(json.dumps(param_details, indent=2), encoding="utf-8")
            LOGGER.info("Saved parameter details to %s", param_file)
            
            # Save experiment configuration
            self._save_experiment_config(metadata_dir)
            
        except Exception as e:
            LOGGER.warning(f"Failed to save model summary: {e}")

    def _generate_parameter_details(self) -> Dict[str, Any]:
        """Generate detailed parameter breakdown by module."""
        param_details = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "modules": {},
            "parameter_groups": {}
        }
        
        # Group parameters by module type
        module_groups = {}
        
        for name, module in self.model.named_modules():
            module_type = module.__class__.__name__
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if module_params > 0:
                param_details["modules"][name] = {
                    "type": module_type,
                    "total_params": module_params,
                    "trainable_params": module_trainable,
                    "shape_info": []
                }
                
                # Add parameter shapes
                for param_name, param in module.named_parameters(recurse=False):
                    param_details["modules"][name]["shape_info"].append({
                        "name": param_name,
                        "shape": list(param.shape),
                        "numel": param.numel(),
                        "requires_grad": param.requires_grad,
                        "dtype": str(param.dtype)
                    })
                
                # Group by module type
                if module_type not in module_groups:
                    module_groups[module_type] = {
                        "count": 0,
                        "total_params": 0,
                        "trainable_params": 0
                    }
                
                module_groups[module_type]["count"] += 1
                module_groups[module_type]["total_params"] += module_params
                module_groups[module_type]["trainable_params"] += module_trainable
        
        param_details["parameter_groups"] = module_groups
        param_details["total_parameters"] = sum(p.numel() for p in self.model.parameters())
        param_details["trainable_parameters"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return param_details

    def _save_experiment_config(self, metadata_dir: Path) -> None:
        """Save the complete experiment configuration."""
        try:
            config_data = {
                "trainer_config": {
                    "num_epochs": self.config.num_epochs,
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "device": self.config.device,
                    "gradient_clip_norm": self.config.gradient_clip_norm,
                    "log_interval": self.config.log_interval,
                    "checkpoint_interval": self.config.checkpoint_interval,
                    "optimizer": self.config.optimizer,
                    "scheduler": self.config.scheduler,
                    "scheduler_params": self.config.scheduler_params,
                    "assigner_params": self.config.assigner_params,
                    "artifact_dir": str(self.config.artifact_dir),
                    "enable_tensorboard": self.config.enable_tensorboard,
                    "tensorboard_log_interval": self.config.tensorboard_log_interval
                },
                "runtime_info": {
                    "device_used": str(self.device),
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
                }
            }
            
            config_file = metadata_dir / "experiment_config.json"
            config_file.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
            LOGGER.info("Saved experiment configuration to %s", config_file)
            
        except Exception as e:
            LOGGER.warning(f"Failed to save experiment configuration: {e}")

    def _log_model_graph(self) -> None:
        """Log model architecture using torchinfo and attempt TensorBoard graph."""
        try:
            if self.writer is None:
                return
                
            # Create dummy input based on model's expected input size
            input_size = getattr(self.model, 'input_size', (640, 640))
            dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=self.device)
            
            # Set model to eval mode for graph logging
            self.model.eval()
            
            # Generate detailed architecture summary using torchinfo
            try:
                LOGGER.info("Generating model architecture summary with torchinfo...")
                model_stats = torchinfo_summary(
                    self.model,
                    input_size=(1, 3, input_size[0], input_size[1]),
                    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                    verbose=0,  # Don't print to console
                    device=self.device
                )
                
                # Save the summary to a text file
                summary_text = str(model_stats)
                summary_path = self.config.artifact_dir / "metadata" / "model_architecture.txt"
                with open(summary_path, 'w') as f:
                    f.write(summary_text)
                LOGGER.info(f"Model architecture summary saved to {summary_path}")
                
                # Log the summary as text to TensorBoard
                self.writer.add_text('Model/Architecture', f"```\n{summary_text}\n```", 0)
                
            except Exception as torchinfo_error:
                LOGGER.warning(f"Could not generate torchinfo summary: {torchinfo_error}")
            
            # Try to log the model graph to TensorBoard
            try:
                with torch.no_grad():
                    self.writer.add_graph(self.model, dummy_input)
                    self.writer.flush()
                    LOGGER.info("Model graph logged to TensorBoard")
            except Exception as graph_error:
                LOGGER.info(f"TensorBoard graph visualization not available: {graph_error}")
                LOGGER.info("This is normal for YOLO models - check model_architecture.txt instead")
            
            # Set model back to train mode
            self.model.train()
            
        except Exception as e:
            LOGGER.warning(f"Error in model graph logging: {e}")
            # This is not critical, so we continue without the graph

    def _evaluate_coco_metrics(self, epoch: int) -> Dict[str, float]:
        """Evaluate COCO metrics on validation set."""
        if not self.config.enable_coco_eval:
            return {}
        
        val_loader = self.dataloaders.get("val")
        if val_loader is None:
            LOGGER.warning("No validation dataloader available for COCO evaluation")
            return {}
        
        LOGGER.info(f"Starting COCO evaluation on validation set for epoch {epoch}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Collect predictions and ground truth
        predictions = []
        ground_truth_annotations = []
        ground_truth_images = []
        
        prediction_id = 1
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                if batch_idx % 50 == 0:  # Log progress every 50 batches
                    LOGGER.info(f"COCO eval progress: batch {batch_idx}/{len(val_loader)}")
                
                images = images.to(self.device)
                targets_on_device = self._move_targets_to_device(targets)
                
                # Get model predictions
                outputs = self.model(images)
                
                # Process each image in the batch
                for img_idx in range(len(images)):
                    image_id = batch_idx * val_loader.batch_size + img_idx + 1
                    
                    # Get predictions for this image
                    img_predictions = self._extract_yolo_predictions(
                        outputs, img_idx, conf_threshold=self.config.visualization_conf_threshold
                    )
                    
                    # Convert predictions to COCO format
                    if len(img_predictions['boxes']) > 0:
                        boxes = img_predictions['boxes'].cpu().numpy()
                        scores = img_predictions['scores'].cpu().numpy()
                        labels = img_predictions['labels'].cpu().numpy()
                        
                        # Convert from x1,y1,x2,y2 to x,y,w,h format (COCO format)
                        for box, score, label in zip(boxes, scores, labels):
                            x1, y1, x2, y2 = box
                            w, h = x2 - x1, y2 - y1
                            
                            predictions.append({
                                "id": prediction_id,
                                "image_id": image_id,
                                "category_id": int(label) + 1,  # COCO categories start from 1
                                "bbox": [float(x1), float(y1), float(w), float(h)],
                                "score": float(score)
                            })
                            prediction_id += 1
                    
                    # Get ground truth for this image
                    if img_idx < len(targets_on_device):
                        target = targets_on_device[img_idx]
                        
                        # Add image info
                        ground_truth_images.append({
                            "id": image_id,
                            "width": images.shape[3],  # W
                            "height": images.shape[2],  # H
                            "file_name": f"image_{image_id}.jpg"
                        })
                        
                        # Add annotations
                        if 'boxes' in target and len(target['boxes']) > 0:
                            boxes = target['boxes'].cpu().numpy()
                            labels = target['labels'].cpu().numpy()
                            
                            for box, label in zip(boxes, labels):
                                x1, y1, x2, y2 = box
                                w, h = x2 - x1, y2 - y1
                                area = w * h
                                
                                ground_truth_annotations.append({
                                    "id": len(ground_truth_annotations) + 1,
                                    "image_id": image_id,
                                    "category_id": int(label) + 1,
                                    "bbox": [float(x1), float(y1), float(w), float(h)],
                                    "area": float(area),
                                    "iscrowd": 0
                                })
                
                # Limit evaluation to avoid memory issues (evaluate on subset)
                if batch_idx >= 100:  # Limit to ~6400 images
                    break
        
        # Set model back to training mode
        self.model.train()
        
        if not predictions:
            LOGGER.warning("No predictions generated for COCO evaluation")
            return {}
        
        if not ground_truth_annotations:
            LOGGER.warning("No ground truth annotations found for COCO evaluation")
            return {}
        
        # Create COCO ground truth format
        categories = [{"id": i + 1, "name": f"class_{i}"} for i in range(10)]  # Assuming 10 classes
        
        coco_gt_dict = {
            "info": {
                "description": "GAI-YOLOv12 Validation Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "GAI-YOLOv12",
                "date_created": "2025-11-02"
            },
            "licenses": [],
            "images": ground_truth_images,
            "annotations": ground_truth_annotations,
            "categories": categories
        }
        
        # Create temporary files for COCO evaluation
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
                json.dump(coco_gt_dict, gt_file)
                gt_file_path = gt_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_file:
                json.dump(predictions, pred_file)
                pred_file_path = pred_file.name
            
            # Load COCO ground truth and predictions
            coco_gt = COCO(gt_file_path)
            coco_dt = coco_gt.loadRes(pred_file_path)
            
            # Run COCO evaluation
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            metrics = {
                'mAP': float(coco_eval.stats[0]),  # AP@0.5:0.95
                'mAP_50': float(coco_eval.stats[1]),  # AP@0.5
                'mAP_75': float(coco_eval.stats[2]),  # AP@0.75
                'mAP_small': float(coco_eval.stats[3]),  # AP for small objects
                'mAP_medium': float(coco_eval.stats[4]),  # AP for medium objects
                'mAP_large': float(coco_eval.stats[5]),  # AP for large objects
                'mAR_1': float(coco_eval.stats[6]),  # AR@1
                'mAR_10': float(coco_eval.stats[7]),  # AR@10
                'mAR_100': float(coco_eval.stats[8]),  # AR@100
                'mAR_small': float(coco_eval.stats[9]),  # AR for small objects
                'mAR_medium': float(coco_eval.stats[10]),  # AR for medium objects
                'mAR_large': float(coco_eval.stats[11])  # AR for large objects
            }
            
            LOGGER.info(f"COCO Evaluation Results for Epoch {epoch}:")
            LOGGER.info(f"  mAP (0.5:0.95): {metrics['mAP']:.4f}")
            LOGGER.info(f"  mAP@0.5:       {metrics['mAP_50']:.4f}")
            LOGGER.info(f"  mAP@0.75:      {metrics['mAP_75']:.4f}")
            
            return metrics
            
        except Exception as e:
            LOGGER.error(f"Failed to run COCO evaluation: {e}")
            return {}
        
        finally:
            # Clean up temporary files
            try:
                import os
                if 'gt_file_path' in locals():
                    os.unlink(gt_file_path)
                if 'pred_file_path' in locals():
                    os.unlink(pred_file_path)
            except:
                pass

    def _draw_predictions_on_image(self, image: torch.Tensor, predictions: Dict[str, torch.Tensor], 
                                 targets: Optional[Dict[str, torch.Tensor]] = None) -> np.ndarray:
        """Draw predictions and targets on image for visualization."""
        try:
            # Convert tensor to numpy array (C, H, W) -> (H, W, C)
            if image.dim() == 4:  # Remove batch dimension if present
                image = image.squeeze(0)
            
            # Denormalize image (assuming ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            
            # Convert to numpy and scale to 0-255
            img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Draw ground truth boxes in green (if enabled)
            if self.config.tensorboard_show_ground_truth and targets is not None and 'boxes' in targets and len(targets['boxes']) > 0:
                boxes = targets['boxes'].cpu().numpy()
                labels = targets.get('labels', torch.zeros(len(boxes))).cpu().numpy()
                
                LOGGER.info(f"GROUND TRUTH: Drawing {len(boxes)} GT boxes")
                LOGGER.info(f"GT box shapes: {boxes.shape}, label shape: {labels.shape}")
                LOGGER.info(f"Sample GT boxes: {boxes[:3] if len(boxes) > 0 else 'No boxes'}")
                
                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = box.astype(int)
                    # Ensure coordinates are within image bounds
                    h, w = img_np.shape[:2]
                    x1, x2 = max(0, x1), min(w-1, x2)
                    y1, y2 = max(0, y1), min(h-1, y2)
                    
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for GT (thicker)
                    cv2.putText(img_np, f'GT:{int(label)}', (x1, max(y1-10, 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw predictions in red (if enabled)
            if self.config.tensorboard_show_predictions and predictions is not None and 'boxes' in predictions and len(predictions['boxes']) > 0:
                boxes = predictions['boxes'].cpu().numpy()
                scores = predictions.get('scores', torch.ones(len(boxes))).cpu().numpy()
                labels = predictions.get('labels', torch.zeros(len(boxes))).cpu().numpy()
                
                LOGGER.info(f"PREDICTIONS: Total predictions: {len(boxes)}")
                LOGGER.info(f"Pred box shapes: {boxes.shape}, scores shape: {scores.shape}")
                LOGGER.info(f"Sample pred boxes: {boxes[:3] if len(boxes) > 0 else 'No boxes'}")
                LOGGER.info(f"Sample scores: {scores[:3] if len(scores) > 0 else 'No scores'}")
                
                # Use configured visualization threshold for drawing
                conf_threshold = self.config.visualization_conf_threshold
                high_conf_mask = scores > conf_threshold
                
                LOGGER.info(f"Drawing {high_conf_mask.sum()} predictions above threshold: {conf_threshold}")
                for box, score, label in zip(boxes[high_conf_mask], 
                                           scores[high_conf_mask], 
                                           labels[high_conf_mask]):
                    x1, y1, x2, y2 = box.astype(int)
                    # Ensure coordinates are within image bounds
                    h, w = img_np.shape[:2]
                    x1, x2 = max(0, x1), min(w-1, x2)
                    y1, y2 = max(0, y1), min(h-1, y2)
                    
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for predictions
                    cv2.putText(img_np, f'PRED:{int(label)}({score:.2f})', (x1, min(y2+15, h-5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                LOGGER.info("PREDICTIONS: No predictions to draw (empty or None)")
            
            # Convert back to RGB for TensorBoard
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            return img_np
            
        except Exception as e:
            LOGGER.warning(f"Error drawing predictions on image: {e}")
            # Return a simple image if drawing fails
            if image.dim() == 4:
                image = image.squeeze(0)
            img_simple = torch.clamp(image, 0, 1)
            return (img_simple.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    def _log_images_to_tensorboard(self, images: torch.Tensor, predictions: List[Dict[str, torch.Tensor]], 
                                  targets: List[Dict[str, torch.Tensor]], step: int, split: str = "train") -> None:
        """Log images with predictions to TensorBoard."""
        if self.writer is None or not self.config.tensorboard_log_images:
            return
        
        try:
            num_images = min(self.config.tensorboard_images_per_batch, len(images))
            
            vis_images = []
            for i in range(num_images):
                img_with_preds = self._draw_predictions_on_image(
                    images[i], 
                    predictions[i] if i < len(predictions) else None,
                    targets[i] if i < len(targets) else None
                )
                # Convert to tensor format expected by TensorBoard (C, H, W)
                img_tensor = torch.from_numpy(img_with_preds).permute(2, 0, 1).float() / 255.0
                vis_images.append(img_tensor)
            
            if vis_images:
                # Stack images into a grid
                image_grid = torch.stack(vis_images)
                self.writer.add_images(f'{split}/predictions', image_grid, step)
                self.writer.flush()
                LOGGER.debug(f"Logged {len(vis_images)} images to TensorBoard for step {step}")
                
        except Exception as e:
            LOGGER.warning(f"Could not log images to TensorBoard: {e}")
            import traceback
            LOGGER.debug(f"Full traceback: {traceback.format_exc()}")

    def _format_outputs_for_visualization(self, outputs) -> List[Dict[str, torch.Tensor]]:
        """Convert model outputs to list of per-image prediction dictionaries."""
        try:
            # Handle YOLO output format: List[DetectionScaleOutput]
            if isinstance(outputs, list) and hasattr(outputs[0], 'raw'):
                # This is YOLO output format - convert raw detections to boxes/scores/labels
                batch_size = outputs[0].raw.shape[0]
                predictions = []
                
                LOGGER.info(f"YOLO OUTPUT: Processing {len(outputs)} detection scales")
                LOGGER.info(f"Batch size: {batch_size}")
                for i, scale_output in enumerate(outputs):
                    LOGGER.info(f"Scale {i}: raw shape = {scale_output.raw.shape}, stride = {scale_output.stride}")
                
                for batch_idx in range(batch_size):
                    # Extract predictions from YOLO raw outputs (basic implementation for visualization)
                    # Use configured threshold for visualization during training
                    pred_dict = self._extract_yolo_predictions(
                        outputs, batch_idx, conf_threshold=self.config.visualization_conf_threshold
                    )
                    predictions.append(pred_dict)
                
                LOGGER.info(f"Created {len(predictions)} prediction dictionaries")
                return predictions
                
            # Handle standard dictionary format
            elif isinstance(outputs, dict):
                batch_size = outputs.get('pred_boxes', outputs.get('boxes', torch.tensor([]))).shape[0]
                predictions = []
                
                for i in range(batch_size):
                    pred_dict = {}
                    
                    # Extract boxes, scores, and labels for this image
                    if 'pred_boxes' in outputs:
                        pred_dict['boxes'] = outputs['pred_boxes'][i]
                    elif 'boxes' in outputs:
                        pred_dict['boxes'] = outputs['boxes'][i]
                    
                    if 'pred_scores' in outputs:
                        pred_dict['scores'] = outputs['pred_scores'][i]
                    elif 'scores' in outputs:
                        pred_dict['scores'] = outputs['scores'][i]
                    
                    if 'pred_labels' in outputs:
                        pred_dict['labels'] = outputs['pred_labels'][i]
                    elif 'labels' in outputs:
                        pred_dict['labels'] = outputs['labels'][i]
                    
                    predictions.append(pred_dict)
                
                return predictions
            else:
                LOGGER.debug(f"Unsupported output format: {type(outputs)}")
                return []
            
        except Exception as e:
            LOGGER.debug(f"Failed to format outputs for visualization: {e}")
            # Return empty predictions 
            return [{'boxes': torch.empty(0, 4), 'scores': torch.empty(0), 'labels': torch.empty(0)}]

    def _extract_yolo_predictions(self, outputs, batch_idx: int, conf_threshold: float = 0.5, max_predictions: int = 50):
        """Extract predictions from YOLO raw outputs for a specific batch item."""
        try:
            device = outputs[0].raw.device
            all_boxes = []
            all_scores = []
            all_labels = []
            
            for scale_output in outputs:
                raw = scale_output.raw[batch_idx]  # Shape: [A, H, W, num_classes + 5]
                anchors = scale_output.anchors
                stride = scale_output.stride
                
                A, H, W, _ = raw.shape
                
                # Extract objectness, box coordinates, and class scores
                obj_scores = torch.sigmoid(raw[..., 4])  # Objectness
                box_coords = raw[..., :4]  # Raw box coordinates
                class_scores = torch.sigmoid(raw[..., 5:])  # Class probabilities
                
                # Apply confidence threshold on objectness
                obj_mask = obj_scores > conf_threshold
                
                if obj_mask.sum() == 0:
                    continue
                
                # Get indices of valid predictions
                anchor_idx, h_idx, w_idx = torch.where(obj_mask)
                
                # Extract valid predictions
                valid_obj_scores = obj_scores[anchor_idx, h_idx, w_idx]
                valid_box_coords = box_coords[anchor_idx, h_idx, w_idx]
                valid_class_scores = class_scores[anchor_idx, h_idx, w_idx]
                
                # Get class predictions
                class_confidences, class_predictions = torch.max(valid_class_scores, dim=1)
                final_scores = valid_obj_scores * class_confidences
                
                # Simple box decoding (basic implementation)
                # Convert relative coordinates to absolute coordinates
                grid_x = w_idx.float()
                grid_y = h_idx.float()
                
                # Basic coordinate transformation (simplified)
                cx = (torch.sigmoid(valid_box_coords[:, 0]) + grid_x) * stride
                cy = (torch.sigmoid(valid_box_coords[:, 1]) + grid_y) * stride
                w = torch.exp(valid_box_coords[:, 2]) * anchors[anchor_idx, 0] * stride
                h = torch.exp(valid_box_coords[:, 3]) * anchors[anchor_idx, 1] * stride
                
                # Convert to x1, y1, x2, y2 format
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                
                # Clamp coordinates to reasonable bounds (image size is typically 640x640)
                x1 = torch.clamp(x1, 0, 640)
                y1 = torch.clamp(y1, 0, 640)
                x2 = torch.clamp(x2, 0, 640)
                y2 = torch.clamp(y2, 0, 640)
                
                # Filter out degenerate boxes
                valid_mask = (x2 > x1) & (y2 > y1)
                if valid_mask.sum() == 0:
                    continue
                    
                x1, y1, x2, y2 = x1[valid_mask], y1[valid_mask], x2[valid_mask], y2[valid_mask]
                final_scores = final_scores[valid_mask]
                class_predictions = class_predictions[valid_mask]
                
                boxes = torch.stack([x1, y1, x2, y2], dim=1)
                
                all_boxes.append(boxes)
                all_scores.append(final_scores)
                all_labels.append(class_predictions)
            
            if all_boxes:
                # Concatenate all predictions
                final_boxes = torch.cat(all_boxes, dim=0)
                final_scores = torch.cat(all_scores, dim=0)
                final_labels = torch.cat(all_labels, dim=0)
                
                # Simple NMS alternative: take top predictions by score
                if len(final_scores) > max_predictions:
                    top_k_indices = torch.topk(final_scores, max_predictions).indices
                    final_boxes = final_boxes[top_k_indices]
                    final_scores = final_scores[top_k_indices]
                    final_labels = final_labels[top_k_indices]
                
                return {
                    'boxes': final_boxes,
                    'scores': final_scores,
                    'labels': final_labels
                }
            else:
                return {
                    'boxes': torch.empty(0, 4, device=device),
                    'scores': torch.empty(0, device=device),
                    'labels': torch.empty(0, dtype=torch.long, device=device)
                }
                
        except Exception as e:
            LOGGER.debug(f"Failed to extract YOLO predictions: {e}")
            device = outputs[0].raw.device
            return {
                'boxes': torch.empty(0, 4, device=device),
                'scores': torch.empty(0, device=device),
                'labels': torch.empty(0, dtype=torch.long, device=device)
            }


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


def run_training(config: Dict[str, Any], resume: bool = False) -> TrainingResult:
    trainer_config = TrainerConfig.from_config(config)
    dataloaders = create_dataloaders(config)
    model_bundle = create_model(config)
    trainer = Trainer(model_bundle, dataloaders, trainer_config)
    return trainer.train(resume=resume)
