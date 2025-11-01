'''
 #  Copyright (c) 2025 Gahan AI Private Limited
 #  Author: Pallab Maji
 #  Create Time: 2025-11-01 11:55:00
 #  Modified time: 2025-11-01 11:55:00
 #  Description: Loss builders for the GAI-YOLOv12 detection model.
 #  Description (Legacy): Implements multi-scale YOLO loss with anchor-aware
 #       assignment, IoU-based objectness targets, and label smoothing.
'''

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .targets import YoloTargets
from .yolov12 import DetectionScaleOutput

DEFAULT_LOSS_CONFIG: Dict[str, Any] = {
    "type": "yolo",
    "cls_weight": 1.0,
    "box_weight": 2.5,
    "obj_weight": 1.0,
    "label_smoothing": 0.1,
}


class YoloDetectionLoss(nn.Module):
    """Anchor-based YOLO loss supporting label smoothing and IoU-weighted objectness."""

    def __init__(
        self,
        *,
        num_classes: int,
        cls_weight: float,
        box_weight: float,
        obj_weight: float,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = float(cls_weight)
        self.box_weight = float(box_weight)
        self.obj_weight = float(obj_weight)
        self.label_smoothing = max(0.0, float(label_smoothing))

    def forward(self, outputs: Sequence[DetectionScaleOutput], targets: YoloTargets) -> torch.Tensor:  # noqa: D401
        if len(outputs) != len(targets):
            raise ValueError("Number of model outputs must match number of target scales")
        if not outputs:
            raise ValueError("At least one detection scale is required for YOLO loss")

        device = outputs[0].raw.device
        dtype = outputs[0].raw.dtype

        total_box_loss = outputs[0].raw.new_tensor(0.0)
        total_obj_loss = outputs[0].raw.new_tensor(0.0)
        total_cls_loss = outputs[0].raw.new_tensor(0.0)
        box_denominator = outputs[0].raw.new_tensor(0.0)
        obj_denominator = outputs[0].raw.new_tensor(0.0)
        cls_denominator = outputs[0].raw.new_tensor(0.0)

        for scale_output, scale_target in zip(outputs, targets, strict=True):
            raw = scale_output.raw
            obj_target = scale_target.objectness.to(device=device, dtype=dtype)
            obj_iou = scale_target.objectness_iou.to(device=device, dtype=dtype)
            box_target = scale_target.box.to(device=device, dtype=dtype)
            class_target = scale_target.class_id.to(device=device)
            ignore_mask = scale_target.ignore_mask.to(device=device, dtype=dtype)

            positive_mask = obj_target.unsqueeze(-1)

            xy_loss = F.binary_cross_entropy_with_logits(raw[..., 0:2], box_target[..., 0:2], reduction="none")
            xy_loss = (xy_loss * positive_mask).sum()

            wh_loss = F.smooth_l1_loss(raw[..., 2:4], box_target[..., 2:4], reduction="none")
            wh_loss = (wh_loss * positive_mask).sum()

            total_box_loss = total_box_loss + xy_loss + wh_loss
            box_denominator = box_denominator + obj_target.sum()

            obj_target_values = obj_target * obj_iou.clamp(min=0.0, max=1.0)
            valid_mask = (1.0 - ignore_mask)
            obj_loss_tensor = F.binary_cross_entropy_with_logits(raw[..., 4], obj_target_values, reduction="none")
            obj_loss_tensor = obj_loss_tensor * valid_mask
            total_obj_loss = total_obj_loss + obj_loss_tensor.sum()
            obj_denominator = obj_denominator + valid_mask.sum()

            if self.num_classes > 0 and raw.size(-1) > 5:
                positive_indices = class_target >= 0
                if positive_indices.any():
                    cls_logits = raw[..., 5:][positive_indices]
                    target_labels = class_target[positive_indices]
                    smoothing = self.label_smoothing if self.label_smoothing > 0 else 0.0
                    total_cls_loss = total_cls_loss + F.cross_entropy(
                        cls_logits,
                        target_labels,
                        reduction="sum",
                        label_smoothing=smoothing,
                    )
                    cls_denominator = cls_denominator + target_labels.numel()

        box_normalizer = torch.clamp(box_denominator, min=1.0)
        obj_normalizer = torch.clamp(obj_denominator, min=1.0)
        cls_normalizer = torch.clamp(cls_denominator, min=1.0)

        loss = self.box_weight * (total_box_loss / box_normalizer)
        loss = loss + self.obj_weight * (total_obj_loss / obj_normalizer)
        if self.num_classes > 0:
            loss = loss + self.cls_weight * (total_cls_loss / cls_normalizer)
        return loss


def build_loss(
    config: Dict[str, Any],
    num_classes: int,
    *,
    anchors: Sequence[Sequence[Sequence[float]]] | None = None,
    strides: Sequence[int] | None = None,
) -> nn.Module:
    del anchors, strides
    merged_config: Dict[str, Any] = {**DEFAULT_LOSS_CONFIG, **(config or {})}
    return YoloDetectionLoss(
        num_classes=num_classes,
        cls_weight=float(merged_config.get("cls_weight", 1.0)),
        box_weight=float(merged_config.get("box_weight", 2.5)),
        obj_weight=float(merged_config.get("obj_weight", 1.0)),
        label_smoothing=float(merged_config.get("label_smoothing", 0.0)),
    )
