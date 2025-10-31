'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-10-31 10:45:00
 # @ Modified time: 2025-10-31 10:55:00
 # @ Description: Integration test to visualize dataloader outputs with configured augmentations on real data.
 # @ Description (Legacy): This is a object detection dataloader utility module. This module 
 #      1. loads data for training and evaluation of the model.
 #      2. applies data augmentation techniques to the input data.
 #      3. prepares data for feeding into the model during training and evaluation.
 #      4. Converts Datasets from various formats to COCO format and saves them.
'''

from __future__ import annotations

import copy
import random
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402

from utils.converter import run_conversion  # noqa: E402
from utils.dataloader import create_dataloaders  # noqa: E402
from utils.utils import load_yaml_config  # noqa: E402

CONFIG_PATH = ROOT_DIR / "config.yaml"
OUTPUT_ROOT = Path(__file__).resolve().parent / "test_output" / "train_augmented"


def _prepare_test_config(base_config: dict[str, object], augmentation_override: dict[str, object] | None = None) -> dict[str, object]:
    config = copy.deepcopy(base_config)
    dataloader_cfg = config.setdefault("dataloader", {})
    dataloader_cfg["batch_size"] = max(2, int(dataloader_cfg.get("batch_size", 2)))
    dataloader_cfg["num_workers"] = 0
    dataloader_cfg["pin_memory"] = False

    if augmentation_override is None:
        dataloader_cfg["augmentation"] = False
        dataloader_cfg["augmentations"] = []
        dataloader_cfg["augmentation_probabilities_per_batch"] = 0.0
    else:
        dataloader_cfg["augmentation"] = True
        dataloader_cfg["augmentation_probabilities_per_batch"] = 1.0
        dataloader_cfg["augmentations"] = [_normalize_augmentation_config(augmentation_override)]
    return config


def _normalize_augmentation_config(augmentation_cfg: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(augmentation_cfg)
    aug_type = str(normalized.get("type", "unspecified"))

    if aug_type in {"MixUp", "CutMix"}:
        normalized["p"] = 1.0
        normalized["alpha"] = float(normalized.get("alpha", 0.4 if aug_type == "MixUp" else 1.0))
    else:
        normalized["p"] = 1.0
        if "probability" in normalized:
            normalized["probability"] = 1.0

    if aug_type == "NoiseInjection":
        normalized["std"] = max(float(normalized.get("std", 0.05)), 0.01)
        normalized["mean"] = float(normalized.get("mean", 0.0))
    return normalized


def _collect_test_variants(base_config: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    dataloader_cfg = base_config.get("dataloader", {})
    variants: list[tuple[str, dict[str, object]]] = [("baseline", _prepare_test_config(base_config, None))]

    for index, augmentation_cfg in enumerate(dataloader_cfg.get("augmentations", [])):
        aug_type = str(augmentation_cfg.get("type", f"augmentation_{index}"))
        slug = f"{aug_type.lower()}_{index:02d}"
        variants.append((slug, _prepare_test_config(base_config, augmentation_cfg)))
    return variants


def _tensor_to_rgb(image_tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu()
    if mean.ndim == 4:
        mean = mean.view(mean.size(1), 1, 1)
    elif mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 4:
        std = std.view(std.size(1), 1, 1)
    elif std.ndim == 1:
        std = std.view(-1, 1, 1)
    image = image * std + mean
    image = image.clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def _save_augmented_batch(
    coco_api: COCO,
    dataset,
    images: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    output_dir: Path,
    max_visuals: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    inverse_mapping = getattr(dataset, "inverse_category_mapping", {})

    for idx in range(min(images.size(0), max_visuals)):
        rgb_image = _tensor_to_rgb(images[idx], mean, std)
        target = targets[idx]
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.imshow(rgb_image)
        boxes = target.get("boxes", torch.zeros((0, 4)))
        labels = target.get("labels", torch.zeros((0,), dtype=torch.int64))
        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle(
                (x1, y1),
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
                linewidth=2,
                edgecolor="cyan",
                facecolor="none",
            )
            axis.add_patch(rect)
            cat_name = "unknown"
            if inverse_mapping:
                label_id = int(labels[box_idx].item()) if box_idx < len(labels) else None
                if label_id is not None and label_id in inverse_mapping:
                    coco_id = inverse_mapping[label_id]
                    cat_name = coco_api.cats.get(coco_id, {}).get("name", cat_name)
            axis.text(
                x1,
                y1 - 4,
                cat_name,
                color="yellow",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.4, "pad": 1},
            )
        axis.axis("off")
        save_path = output_dir / f"augmented_sample_{idx:02d}.png"
        figure.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(figure)
        saved_paths.append(save_path)
    return saved_paths


@pytest.mark.integration
def test_dataloader_augmented_batch_visualization() -> None:
    if not CONFIG_PATH.is_file():
        pytest.skip("Configuration file missing; cannot run dataloader visualization test")

    base_config = load_yaml_config(CONFIG_PATH)
    dataset_cfg = base_config.get("dataset", {})
    if dataset_cfg.get("name", "").lower() != "bdd100k":
        pytest.skip("Config not targeting BDD100K dataset")

    dataset_root = Path(dataset_cfg.get("root_dir", "")).expanduser()
    export_dir = Path(dataset_cfg.get("export_coco_path", "")).expanduser()
    if not dataset_root.exists() or not export_dir.exists():
        pytest.skip("Dataset root or export directory missing; skipping visualization test")

    run_conversion(base_config)

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    variants = _collect_test_variants(base_config)
    saved_artifacts: dict[str, list[Path]] = {}

    for variant_index, (variant_name, variant_config) in enumerate(variants):
        seed = 1337 + variant_index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        dataloaders = create_dataloaders(variant_config)
        train_loader = dataloaders.get("train")
        if train_loader is None:
            pytest.skip("Train dataloader not available")

        images, targets = next(iter(train_loader))
        assert images.size(0) >= 2, "Augmentation batch should contain at least two samples"

        images = images.cpu()
        targets = [
            {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in target.items()}
            for target in targets
        ]

        dataset = getattr(train_loader, "dataset", None)
        assert dataset is not None, "Train dataloader missing dataset reference"

        coco_api = getattr(dataset, "coco", None)
        if coco_api is None:
            pytest.skip("COCO API not accessible from dataset; cannot visualize labels")

        mean = torch.tensor(
            variant_config["dataloader"].get("normalize_mean", [0.485, 0.456, 0.406]),
            dtype=torch.float32,
        ).view(-1, 1, 1)
        std = torch.tensor(
            variant_config["dataloader"].get("normalize_std", [0.229, 0.224, 0.225]),
            dtype=torch.float32,
        ).view(-1, 1, 1)

        output_dir = OUTPUT_ROOT / variant_name
        saved_paths = _save_augmented_batch(
            coco_api=coco_api,
            dataset=dataset,
            images=images,
            targets=targets,
            output_dir=output_dir,
            max_visuals=3,
            mean=mean,
            std=std,
        )

        saved_artifacts[variant_name] = saved_paths

    assert saved_artifacts, "No dataloader variants were processed"
    for variant_name, paths in saved_artifacts.items():
        assert paths, f"No samples saved for variant: {variant_name}"
        for path in paths:
            assert path.is_file()
            assert path.stat().st_size > 0