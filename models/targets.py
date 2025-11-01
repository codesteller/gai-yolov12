'''
 #  Copyright (c) 2025 Gahan AI Private Limited
 #  Author: Pallab Maji
 #  Create Time: 2025-11-01 11:55:00
 #  Modified time: 2025-11-01 11:55:00
 #  Description: Shared dataclasses for YOLO detection targets.
 #  Description (Legacy): Provides structured containers for multi-scale
 #       target tensors used by the anchor-based training pipeline.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class YoloScaleTargets:
    """Targets for a single detection scale.

    Attributes:
        objectness: Tensor of shape (B, A, H, W) indicating positive anchors.
        objectness_iou: Tensor of shape (B, A, H, W) storing IoU-based objectness targets.
        box: Tensor of shape (B, A, H, W, 4) containing [tx, ty, tw, th] regression targets.
        class_id: Tensor of shape (B, A, H, W) with integer class assignments (-1 for background).
        ignore_mask: Tensor of shape (B, A, H, W) where 1.0 marks anchors to ignore for objectness loss.
    """

    objectness: torch.Tensor
    objectness_iou: torch.Tensor
    box: torch.Tensor
    class_id: torch.Tensor
    ignore_mask: torch.Tensor


YoloTargets = List[YoloScaleTargets]

__all__ = ["YoloScaleTargets", "YoloTargets"]
