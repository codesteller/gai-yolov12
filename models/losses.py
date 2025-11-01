'''
 #  Copyright (c) 2025 Gahan AI Private Limited
 #  Author: Pallab Maji
 #  Create Time: 2025-10-31 11:20:00
 #  Modified time: 2025-10-31 11:20:00
 #  Description: Loss builders for the GAI-YOLOv12 detection model.
 #  Description (Legacy): Provides baseline detection loss computation with
 #       support for configurable weighting of classification, box, and
 #       objectness components.
'''

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_LOSS_CONFIG: Dict[str, Any] = {
    "type": "simple",
    "cls_weight": 1.0,
    "bbox_weight": 2.0,
    "obj_weight": 1.0,
    "eps": 1e-6,
}


class YoloDetectionLoss(nn.Module):
    """Baseline detection loss expecting pre-assigned targets per anchor."""

    def __init__(self, *, num_classes: int, cls_weight: float, bbox_weight: float, obj_weight: float, eps: float) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = float(cls_weight)
        self.bbox_weight = float(bbox_weight)
        self.obj_weight = float(obj_weight)
        self.eps = float(eps)

    def forward(self, predictions: Dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:  # noqa: D401
        logits = predictions.get("pred_logits")
        boxes = predictions.get("pred_boxes")
        objectness = predictions.get("pred_objectness")
        if logits is None or boxes is None or objectness is None:
            raise ValueError("Predictions must contain 'pred_logits', 'pred_boxes', and 'pred_objectness'")

        batch_size, num_anchors, num_classes = logits.shape
        device = logits.device

        cls_targets = torch.zeros_like(logits, device=device)
        box_targets = torch.zeros_like(boxes, device=device)
        obj_targets = torch.zeros(batch_size, num_anchors, 1, device=device)

        for batch_index, target in enumerate(targets):
            assigned = target.get("assigned_anchors")
            labels = target.get("assigned_labels")
            assigned_boxes = target.get("assigned_boxes")
            if assigned is None or labels is None or assigned_boxes is None:
                continue
            anchor_idx = torch.as_tensor(assigned, device=device, dtype=torch.long)
            label_tensor = torch.as_tensor(labels, device=device, dtype=torch.long)
            box_tensor = torch.as_tensor(assigned_boxes, device=device, dtype=torch.float32)
            limit = min(anchor_idx.numel(), label_tensor.numel(), box_tensor.shape[0])
            if limit == 0:
                continue
            anchor_idx = anchor_idx[:limit].clamp_(0, num_anchors - 1)
            label_tensor = label_tensor[:limit].clamp_(0, num_classes - 1)
            box_tensor = box_tensor[:limit]
            cls_targets[batch_index, anchor_idx, :] = 0.0
            cls_targets[batch_index, anchor_idx, label_tensor] = 1.0
            box_targets[batch_index, anchor_idx] = box_tensor
            obj_targets[batch_index, anchor_idx] = 1.0

        cls_loss = _binary_cross_entropy(logits, cls_targets, self.eps)
        box_loss = _smooth_l1_loss(boxes, box_targets, obj_targets, self.eps)
        obj_loss = _binary_cross_entropy(objectness, obj_targets, self.eps)

        total_loss = (
            self.cls_weight * cls_loss
            + self.bbox_weight * box_loss
            + self.obj_weight * obj_loss
        )
        return total_loss


def _binary_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, eps: float) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.mean(dim=-1)
    return loss.mean().clamp_min(eps)


def _smooth_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    obj_targets: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    pointwise = F.smooth_l1_loss(predictions, targets, reduction="none")
    pointwise = pointwise.sum(dim=-1)
    weights = obj_targets.squeeze(-1)
    if weights.sum() <= 0:
        return predictions.new_tensor(eps)
    weighted = pointwise * weights
    return (weighted.sum() / weights.sum()).clamp_min(eps)


def build_loss(config: Dict[str, Any], num_classes: int) -> nn.Module:
    merged_config: Dict[str, Any] = {**DEFAULT_LOSS_CONFIG, **(config or {})}
    loss_type = merged_config.get("type", "simple")
    if loss_type != "simple":  # pragma: no cover - reserved for future extensions
        raise ValueError(f"Unsupported loss type '{loss_type}'")
    return YoloDetectionLoss(
        num_classes=num_classes,
        cls_weight=float(merged_config.get("cls_weight", 1.0)),
        bbox_weight=float(merged_config.get("bbox_weight", 2.0)),
        obj_weight=float(merged_config.get("obj_weight", 1.0)),
        eps=float(merged_config.get("eps", 1e-6)),
    )
