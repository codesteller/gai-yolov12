'''
 #  Copyright (c) 2025 Gahan AI Private Limited
 #  Author: Pallab Maji
 #  Create Time: 2025-10-31 11:20:00
 #  Modified time: 2025-10-31 11:20:00
 #  Description: Backbone implementations and registry for GAI-YOLOv12 models.
 #  Description (Legacy): Provides lightweight feature extractor backbones used by
 #       the detection head. Additional backbones can be registered for experiments.
'''

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn

LOGGER = logging.getLogger("gai_yolov12.models")


@dataclass
class BackboneSpec:
    """Descriptor holding a backbone module and its output channel dimension."""

    module: nn.Module
    out_channels: int


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # noqa: D401 - inherits docs
        return self.block(tensor)


class ResidualBlock(nn.Module):
    """Residual block composed of two convolutional blocks."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=1)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # noqa: D401
        identity = tensor
        out = self.conv1(tensor)
        out = self.conv2(out)
        return out + identity


class TinyCSPBackbone(nn.Module):
    """Lightweight CSP-like backbone for quick experiments and tests."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64, depth: int = 3) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Backbone depth must be positive")
        channels = in_channels
        layers = []
        outputs = []
        for stage_index in range(depth):
            hidden_channels = base_channels * (2 ** stage_index)
            downsample = ConvBlock(channels, hidden_channels, stride=2)
            residual = ResidualBlock(hidden_channels)
            stage = nn.Sequential(downsample, residual)
            layers.append(stage)
            outputs.append(hidden_channels)
            channels = hidden_channels
        self.stages = nn.ModuleList(layers)
        self._out_channels = outputs[-1]

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:  # noqa: D401
        features: Dict[str, torch.Tensor] = {}
        x = tensor
        for index, stage in enumerate(self.stages):
            x = stage(x)
            features[f"stage_{index + 1}"] = x
        return features


_BACKBONE_REGISTRY: Dict[str, Any] = {
    "tiny_csp": TinyCSPBackbone,
}


def register_backbone(name: str, constructor: Any) -> None:
    if name in _BACKBONE_REGISTRY:
        raise ValueError(f"Backbone '{name}' already registered")
    _BACKBONE_REGISTRY[name] = constructor
    LOGGER.info("Registered backbone '%s'", name)


def build_backbone(
    *,
    name: str,
    input_channels: int,
    pretrained: bool = False,
    **kwargs: Any,
) -> BackboneSpec:
    constructor = _BACKBONE_REGISTRY.get(name)
    if constructor is None:
        raise ValueError(f"Unknown backbone '{name}'")

    if name == "tiny_csp":
        base_channels = int(kwargs.get("base_channels", 64))
        depth = int(kwargs.get("depth", 3))
        module = constructor(in_channels=input_channels, base_channels=base_channels, depth=depth)
    else:  # pragma: no cover - future backbones
        module = constructor(in_channels=input_channels, pretrained=pretrained, **kwargs)

    if not hasattr(module, "out_channels"):
        raise AttributeError(f"Backbone '{name}' must expose an 'out_channels' attribute")
    out_channels = int(getattr(module, "out_channels"))
    return BackboneSpec(module=module, out_channels=out_channels)