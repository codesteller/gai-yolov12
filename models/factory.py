'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-10-31 11:20:00
 # @ Modified time: 2025-10-31 11:20:00
 # @ Description: Factory utilities to construct the GAI-YOLOv12 model stack.
 # @ Description (Legacy): This module defines configuration data structures and
 #      helpers to build the model, backbone, and loss objects for training.
'''

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from .backbones import BackboneSpec, build_backbone
from .losses import build_loss
from .yolov12 import GAIYoloV12

LOGGER = logging.getLogger("gai_yolov12.models")


@dataclass(frozen=True)
class ModelConfig:
    """Normalized configuration for constructing the GAI-YOLOv12 model."""

    name: str
    num_classes: int
    input_channels: int = 3
    backbone: str = "tiny_csp"
    backbone_params: Dict[str, Any] = field(default_factory=dict)
    head_params: Dict[str, Any] = field(default_factory=dict)
    loss_config: Dict[str, Any] = field(default_factory=dict)
    pretrained: bool = False
    checkpoint_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if self.input_channels <= 0:
            raise ValueError("input_channels must be a positive integer")
        if self.checkpoint_path is not None and not isinstance(self.checkpoint_path, Path):
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))

    @classmethod
    def from_config(cls, raw_config: Dict[str, Any]) -> "ModelConfig":
        model_section = raw_config.get("model", {}) if "model" in raw_config else raw_config
        name = str(model_section.get("name", "gai-yolov12"))
        num_classes = int(model_section.get("num_classes", 0))
        input_channels = int(model_section.get("input_channels", 3))
        backbone = str(model_section.get("backbone", "tiny_csp"))
        pretrained = bool(model_section.get("pretrained", False))
        checkpoint = model_section.get("checkpoint_path")
        backbone_params = dict(model_section.get("backbone_params", {}))
        head_params = dict(model_section.get("head", {}))
        loss_config = dict(model_section.get("loss", {}))
        return cls(
            name=name,
            num_classes=num_classes,
            input_channels=input_channels,
            backbone=backbone,
            backbone_params=backbone_params,
            head_params=head_params,
            loss_config=loss_config,
            pretrained=pretrained,
            checkpoint_path=Path(checkpoint).expanduser() if checkpoint else None,
        )


@dataclass
class ModelBundle:
    """Container for the assembled model components."""

    model: nn.Module
    loss: nn.Module
    metadata: Dict[str, Any]


def create_model(config: Dict[str, Any] | ModelConfig, *, load_checkpoint: bool = True) -> ModelBundle:
    """Build the model, backbone, and loss objects from configuration data.

    Args:
        config: Either a mapping representing the model configuration or an existing
            :class:`ModelConfig` instance.
        load_checkpoint: When ``True`` and a checkpoint path is provided, attempt to
            load weights into the instantiated model.

    Returns:
        A :class:`ModelBundle` containing the model, loss function, and metadata.
    """

    model_config = config if isinstance(config, ModelConfig) else ModelConfig.from_config(config)

    backbone_spec = _build_backbone(model_config)
    model = _build_model(model_config, backbone_spec)
    loss_fn = build_loss(model_config.loss_config, model_config.num_classes)

    metadata = {
        "model_name": model_config.name,
        "backbone": model_config.backbone,
        "num_classes": model_config.num_classes,
        "num_parameters": sum(param.numel() for param in model.parameters()),
    }

    if load_checkpoint and model_config.checkpoint_path is not None:
        _maybe_load_checkpoint(model, model_config.checkpoint_path)
        metadata["checkpoint"] = str(model_config.checkpoint_path)

    return ModelBundle(model=model, loss=loss_fn, metadata=metadata)


def _build_backbone(model_config: ModelConfig) -> BackboneSpec:
    return build_backbone(
        name=model_config.backbone,
        input_channels=model_config.input_channels,
        pretrained=model_config.pretrained,
        **model_config.backbone_params,
    )


def _build_model(model_config: ModelConfig, backbone_spec: BackboneSpec) -> GAIYoloV12:
    head_params = dict(model_config.head_params)
    hidden_channels = int(head_params.get("hidden_channels", 128))
    return GAIYoloV12(
        backbone=backbone_spec.module,
        num_classes=model_config.num_classes,
        hidden_channels=hidden_channels,
        input_channels=model_config.input_channels,
    )


def _maybe_load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        LOGGER.warning("Checkpoint path does not exist: %s", checkpoint_path)
        return
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except (OSError, RuntimeError) as exc:
        LOGGER.warning("Failed to load checkpoint %s: %s", checkpoint_path, exc)
        return

    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        LOGGER.warning("Invalid checkpoint format at %s", checkpoint_path)
        return

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.info("Missing keys while loading checkpoint: %s", sorted(missing))
    if unexpected:
        LOGGER.info("Unexpected keys while loading checkpoint: %s", sorted(unexpected))
    LOGGER.info("Loaded checkpoint from %s", checkpoint_path)
