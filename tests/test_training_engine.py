'''
 #  Copyright (c) 2025 Gahan AI Private Limited
 #  Author: Pallab Maji
 #  Create Time: 2025-11-01 09:15:00
 #  Modified time: 2025-11-01 09:15:00
 #  Description: Unit tests for the GAI-YOLOv12 training engine.
 #  Description (Legacy): Ensures the trainer executes forward/backward passes,
 #       produces checkpoints, and records loss metrics for synthetic data.
'''

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import DetectionScaleOutput, create_model
from train import Trainer, TrainerConfig, YoloTargetEncoder


def _detection_collate(batch: List[tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


class DummyDetectionDataset(Dataset):
    def __init__(self, num_samples: int = 8, num_boxes: int = 2) -> None:
        self.num_samples = num_samples
        self.num_boxes = num_boxes

    def __len__(self) -> int:  # noqa: D401
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = torch.rand(3, 64, 64)
        boxes = torch.stack([torch.tensor([5.0 * (i + 1)] * 4) for i in range(self.num_boxes)], dim=0)
        labels = torch.arange(1, self.num_boxes + 1, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor([1.0] * self.num_boxes, dtype=torch.float32),
            "iscrowd": torch.zeros(self.num_boxes, dtype=torch.int64),
            "orig_size": torch.tensor([64, 64], dtype=torch.int64),
            "size": torch.tensor([64, 64], dtype=torch.int64),
        }
        return image, target


def test_trainer_runs_single_epoch(tmp_path) -> None:
    dataset = DummyDetectionDataset(num_samples=4, num_boxes=2)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=_detection_collate)

    config = {
        "model": {
            "name": "gai-yolov12",
            "num_classes": 4,
            "input_channels": 3,
            "backbone": "tiny_csp",
            "backbone_params": {"base_channels": 8, "depth": 1},
            "head": {"hidden_channels": 16},
            "loss": {"cls_weight": 1.0, "bbox_weight": 1.0, "obj_weight": 1.0},
                "anchors": [[[16.0, 16.0], [32.0, 32.0]]],
                "strides": [32],
        },
        "experiment": {
            "num_epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "artifact_dir": str(tmp_path),
            "log_interval": 1,
            "checkpoint_interval": 1,
            "device": "cpu",
        },
    }

    bundle = create_model(config)
    trainer_config = TrainerConfig.from_config(config)
    trainer = Trainer(bundle, {"train": dataloader}, trainer_config)

    result = trainer.train()
    assert len(result.metrics) == 1
    assert result.metrics[0].train_loss >= 0.0
    assert result.checkpoints
    assert result.checkpoints[0].is_file()
    assert result.summary_path is not None
    assert result.summary_path.is_file()
    history = result.summary_path.read_text(encoding="utf-8")
    assert "train_loss" in history


def test_yolo_target_encoder_assigns_anchor_targets() -> None:
    encoder = YoloTargetEncoder()
    raw = torch.zeros(1, 2, 2, 2, 9)
    anchors = torch.tensor([[32.0, 32.0], [64.0, 64.0]])
    outputs = [DetectionScaleOutput(raw=raw, stride=32, anchors=anchors, grid_size=(2, 2))]
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 32.0, 32.0], [32.0, 32.0, 64.0, 64.0]], dtype=torch.float32),
            "labels": torch.tensor([1, 2], dtype=torch.int64),
            "size": torch.tensor([64, 64], dtype=torch.float32),
        }
    ]

    scale_targets = encoder(outputs, targets)[0]

    assert torch.isclose(scale_targets.objectness.sum(), torch.tensor(2.0))
    assert scale_targets.class_id[0, 0, 0, 0].item() == 1
    assert scale_targets.class_id[0, 0, 1, 1].item() == 2
    assert scale_targets.class_id[0, 1].eq(-1).all()
    assert scale_targets.objectness_iou[0, 0, 0, 0] > 0.0
    assert scale_targets.ignore_mask.sum() >= 0.0