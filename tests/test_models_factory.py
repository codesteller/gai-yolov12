'''
 #  Copyright (c) 2025 Gahan AI Private Limited
 #  Author: Pallab Maji
 #  Create Time: 2025-10-31 11:25:00
 #  Modified time: 2025-10-31 11:25:00
 #  Description: Tests for the GAI-YOLOv12 model factory and configuration helpers.
 #  Description (Legacy): Ensures the model creation workflow generates valid
 #       modules, performs forward passes, and computes losses without errors.
'''

from __future__ import annotations

import torch

from models import ModelConfig, create_model


def _base_model_dict() -> dict[str, object]:
    return {
        "model": {
            "name": "gai-yolov12",
            "num_classes": 4,
            "input_channels": 3,
            "backbone": "tiny_csp",
            "backbone_params": {"base_channels": 16, "depth": 2},
            "head": {"hidden_channels": 32},
            "loss": {"cls_weight": 1.0, "bbox_weight": 1.5, "obj_weight": 1.0},
        }
    }


def test_model_config_from_dict_defaults() -> None:
    config = ModelConfig.from_config({"model": {"num_classes": 3}})
    assert config.name == "gai-yolov12"
    assert config.backbone == "tiny_csp"
    assert config.num_classes == 3
    assert config.pretrained is False


def test_model_factory_forward_and_loss() -> None:
    bundle = create_model(_base_model_dict())
    assert bundle.metadata["num_parameters"] > 0

    model = bundle.model
    loss_fn = bundle.loss

    inputs = torch.randn(2, 3, 320, 320)
    outputs = model(inputs)

    predictions = outputs["predictions"]
    assert isinstance(predictions, list)
    assert predictions and predictions[0].shape[0] == 2
    assert predictions[0].shape[-1] == 5 + bundle.metadata["num_classes"]

    targets = []
    for _ in range(inputs.shape[0]):
        targets.append(
            {
                "assigned_scales": [0],
                "assigned_anchors": [0],
                "assigned_grid_xy": [[0, 0]],
                "assigned_boxes": [[16.0, 16.0, 48.0, 48.0]],
                "assigned_labels": [1],
            }
        )

    loss = loss_fn(outputs, targets)
    assert torch.isfinite(loss).item() == 1
    assert loss.item() >= 0.0