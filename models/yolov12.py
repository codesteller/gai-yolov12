'''
 #  Copyright (c) 2025 Gahan AI Private Limited
 #  Author: Pallab Maji
 #  Create Time: 2025-10-31 11:20:00
 #  Modified time: 2025-10-31 11:20:00
 #  Description: Core network definition for the GAI-YOLOv12 detector.
 #  Description (Legacy): Implements a lightweight detection head that can be
 #       extended for experiments with additional features and loss functions.
'''

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class DetectionHead(nn.Module):
    """Single-scale detection head producing class, box, and objectness logits."""

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 128) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        self.box_pred = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        self.obj_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:  # noqa: D401
        hidden = self.stem(features)
        cls_logits = self.cls_pred(hidden)
        box_reg = self.box_pred(hidden)
        obj_logits = self.obj_pred(hidden)
        return {
            "cls_logits": cls_logits,
            "box_reg": box_reg,
            "obj_logits": obj_logits,
        }


class GAIYoloV12(nn.Module):
    """GAI-YOLOv12 detection network supporting lightweight backbones."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        num_classes: int,
        hidden_channels: int = 128,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        backbone_out_channels = self._infer_backbone_channels(backbone, input_channels)
        self.head = DetectionHead(backbone_out_channels, num_classes, hidden_channels)

    @staticmethod
    def _infer_backbone_channels(backbone: nn.Module, input_channels: int) -> int:
        if hasattr(backbone, "out_channels"):
            return int(getattr(backbone, "out_channels"))
        raise AttributeError("Backbone must define an 'out_channels' attribute")

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:  # noqa: D401
        features = self.backbone(images)

        if isinstance(features, dict):
            feature_maps = list(features.values())
        elif isinstance(features, (list, tuple)):
            feature_maps = list(features)
        else:
            feature_maps = [features]

        if not feature_maps:
            raise ValueError("Backbone returned no feature maps")

        pyramid_feature = feature_maps[-1]
        head_outputs = self.head(pyramid_feature)
        cls_logits = head_outputs["cls_logits"]
        box_reg = head_outputs["box_reg"]
        obj_logits = head_outputs["obj_logits"]

        batch_size, _, height, width = cls_logits.shape
        cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
        box_reg = box_reg.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        obj_logits = obj_logits.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)

        return {
            "pred_logits": cls_logits,
            "pred_boxes": box_reg,
            "pred_objectness": obj_logits,
            "feature_map_shape": (height, width),
        }