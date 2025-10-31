'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-10-30 18:05:00
 # @ Modified time: 2025-10-31 10:05:00
 # @ Description: Integration test that validates COCO conversion outputs using real BDD100K samples.
'''

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from pycocotools.coco import COCO

from utils.converter import run_conversion
from utils.utils import load_yaml_config


CONFIG_PATH = ROOT_DIR / "config.yaml"
OUTPUT_ROOT = Path(__file__).resolve().parent / "test_output"


def _resolve_split_dir(dataset_root: Path, split: str) -> Path:
    rel_path = Path(split.strip("/")) if split else Path()
    return (dataset_root / rel_path).resolve()


def _visualize_samples(
    coco: COCO,
    dataset_root: Path,
    split: str,
    output_dir: Path,
    max_images: int = 3,
) -> list[Path]:
    saved_paths: list[Path] = []
    split_dir = _resolve_split_dir(dataset_root, split)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_ids = coco.getImgIds()
    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        if not ann_ids:
            continue
        img_info = coco.loadImgs([image_id])[0]
        image_path = split_dir / img_info["file_name"]
        if not image_path.is_file():
            continue
        figure, axis = plt.subplots(figsize=(8, 5))
        with Image.open(image_path) as img:
            axis.imshow(img)
        annotations = coco.loadAnns(ann_ids)
        for ann in annotations:
            x, y, width, height = ann["bbox"]
            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            axis.add_patch(rect)
            category_name = coco.cats.get(ann["category_id"], {}).get("name", "unknown")
            axis.text(
                x,
                y - 4,
                category_name,
                color="yellow",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.4, "pad": 1},
            )
        axis.axis("off")
        save_path = output_dir / f"{Path(img_info['file_name']).stem}_viz.png"
        figure.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(figure)
        saved_paths.append(save_path)
        if len(saved_paths) >= max_images:
            break
    return saved_paths


@pytest.mark.integration
def test_bdd_conversion_visualizations_use_real_data() -> None:
    if not CONFIG_PATH.is_file():
        pytest.skip("Configuration file missing; cannot validate conversion")

    config = load_yaml_config(CONFIG_PATH)
    dataset_cfg = config.get("dataset", {})

    if dataset_cfg.get("name", "").lower() != "bdd100k":
        pytest.skip("Config not set for BDD100K dataset")

    dataset_root = Path(dataset_cfg.get("root_dir", "")).expanduser()
    if not dataset_root.exists():
        pytest.skip(f"Dataset root not found: {dataset_root}")

    train_split = dataset_cfg.get("train_split")
    if not train_split:
        pytest.skip("Train split path missing in dataset configuration")

    export_path = dataset_cfg.get("export_coco_path")
    if not export_path:
        pytest.skip("COCO export path missing in dataset configuration")

    export_dir = Path(export_path).expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)

    run_conversion(config)

    train_export = export_dir / "train.json"
    if not train_export.is_file():
        pytest.skip(f"Train COCO annotation not available at {train_export}")

    coco = COCO(str(train_export))
    output_dir = OUTPUT_ROOT / "train"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    split_dir = _resolve_split_dir(dataset_root, train_split)
    if not split_dir.exists():
        pytest.skip(f"Train split directory not found: {split_dir}")

    saved_paths = _visualize_samples(coco, dataset_root, train_split, output_dir, max_images=3)

    assert saved_paths, "No visualization images were generated from the converted annotations"
    for path in saved_paths:
        assert path.is_file()
        assert path.stat().st_size > 0