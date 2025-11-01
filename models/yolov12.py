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

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import nn


@dataclass
class DetectionScaleOutput:
    """Container for raw detection logits and metadata per feature scale."""

    raw: torch.Tensor  # shape: (B, A, H, W, num_classes + 5)
    stride: int
    anchors: torch.Tensor  # shape: (A, 2)
    grid_size: Tuple[int, int]


class GAIYoloV12(nn.Module):
    """GAI-YOLOv12 detection network supporting lightweight backbones."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        num_classes: int,
        hidden_channels: int = 128,
        input_channels: int = 3,
        anchors: Sequence[Sequence[Sequence[float]]] | None = None,
        strides: Sequence[int] | None = None,
        input_size: Tuple[int, int] = (640, 640),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.input_size = tuple(input_size)

        if anchors is None or len(anchors) == 0:
            raise ValueError("Anchors configuration must be provided for YOLO head")
        self.anchors: List[torch.Tensor] = []
        for idx, scale in enumerate(anchors):
            anchor_tensor = torch.tensor(scale, dtype=torch.float32)
            self.register_buffer(f"_anchors_{idx}", anchor_tensor, persistent=True)
            self.anchors.append(getattr(self, f"_anchors_{idx}"))

        self.strides: List[int] = list(strides) if strides else [8, 16, 32][: len(self.anchors)]
        if len(self.strides) != len(self.anchors):
            raise ValueError("Number of strides must match number of anchor groups")

        feature_channels = self._resolve_feature_channels(backbone)
        if len(feature_channels) < len(self.anchors):
            raise ValueError("Backbone does not expose enough feature maps for detection scales")

        selected_channels = feature_channels[-len(self.anchors) :]
        self.heads = nn.ModuleList(
            [
                DetectionHeadBlock(in_ch, len(anchor_group), num_classes, hidden_channels)
                for in_ch, anchor_group in zip(selected_channels, self.anchors, strict=True)
            ]
        )

    @staticmethod
    def _resolve_feature_channels(backbone: nn.Module) -> List[int]:
        if hasattr(backbone, "feature_channels"):
            channels = getattr(backbone, "feature_channels")
            return list(channels)
        if hasattr(backbone, "out_channels"):
            return [int(getattr(backbone, "out_channels"))]
        raise AttributeError("Backbone must expose feature_channels or out_channels attribute")

    def forward(self, images: torch.Tensor) -> List[DetectionScaleOutput]:  # noqa: D401
        features = self.backbone(images)

        if isinstance(features, dict):
            sorted_features = [features[key] for key in sorted(features.keys())]
        elif isinstance(features, (list, tuple)):
            sorted_features = list(features)
        else:
            sorted_features = [features]

        if not sorted_features:
            raise ValueError("Backbone returned no feature maps")

        feature_maps = sorted_features[-len(self.heads) :]
        outputs: List[DetectionScaleOutput] = []
        _, _, img_h, img_w = images.shape

        for scale_idx, (head, feature, anchor_group, stride) in enumerate(
            zip(self.heads, feature_maps, self.anchors, self.strides, strict=True)
        ):
            raw = head(feature)
            _, _, height, width, _ = raw.shape
            stride_h = max(int(round(img_h / height)), 1)
            stride_w = max(int(round(img_w / width)), 1)
            stride_value = int(stride) if stride else int(stride_w)
            # Ensure consistent stride estimation
            if stride_value != stride_h or stride_value != stride_w:
                stride_value = int(stride_w)
            outputs.append(
                DetectionScaleOutput(
                    raw=raw,
                    stride=stride_value,
                    anchors=anchor_group.to(images.device),
                    grid_size=(height, width),
                )
            )

        return outputs


class DetectionHeadBlock(nn.Module):
    """Multi-anchor detection head producing raw logits for YOLO decoding."""

    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, hidden_channels: int) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.num_outputs = num_classes + 5
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, num_anchors * self.num_outputs, kernel_size=1),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        raw = self.block(tensor)
        batch_size, _, height, width = raw.shape
        raw = raw.view(batch_size, self.num_anchors, self.num_outputs, height, width)
        raw = raw.permute(0, 1, 3, 4, 2).contiguous()
        return raw