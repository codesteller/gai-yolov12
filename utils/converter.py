from __future__ import annotations
'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-10-30 17:20:00
 # @ Modified time: 2025-10-30 17:27:00
 # @ Description: Conversion framework for exporting supported datasets into COCO format artifacts.
'''

"""Dataset conversion utilities for exporting annotations to COCO format."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from .utils import ensure_dir, read_json, write_json

LOGGER = logging.getLogger("gai_yolov12.converter")


@dataclass
class ConversionMetadata:
    """Record describing a single dataset split conversion."""

    dataset_name: str
    split: str
    source_annotation: str
    source_mtime: float
    images_root: str
    class_map: Dict[str, int]
    converted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "source_annotation": self.source_annotation,
            "source_mtime": self.source_mtime,
            "images_root": self.images_root,
            "class_map": self.class_map,
            "converted_at": self.converted_at,
        }


class BaseDatasetToCocoConverter:
    """Base class encapsulating conversion logic common to object detection datasets."""

    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        export_dir: str,
        class_map: Dict[str, int],
        force: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.root_dir = Path(root_dir)
        self.export_dir = ensure_dir(export_dir)
        self.class_map = class_map
        self.force = force
        self._categories = self._build_categories()

    def _build_categories(self) -> List[Dict[str, object]]:
        categories = []
        for label, idx in sorted(self.class_map.items(), key=lambda item: item[1]):
            categories.append({"id": idx, "name": label, "supercategory": "object"})
        return categories

    def convert(self, splits: Dict[str, Tuple[str, str]]) -> Dict[str, Path]:
        """Convert each requested split into a COCO annotation file."""
        results: Dict[str, Path] = {}
        for split_name, (images_rel_path, annotation_rel_path) in splits.items():
            output_path = self.export_dir / f"{split_name}.json"
            meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
            source_annotation = self.root_dir / annotation_rel_path
            images_root = self.root_dir / images_rel_path
            ensure_dir(output_path.parent)
            if not source_annotation.is_file():
                LOGGER.warning("Annotation file missing for split %s: %s", split_name, source_annotation)
                continue
            if not images_root.exists():
                LOGGER.warning("Image directory missing for split %s: %s", split_name, images_root)
                continue
            if self._should_skip_conversion(output_path, meta_path, source_annotation.stat().st_mtime):
                LOGGER.info("Skipping conversion for %s split; up-to-date export found at %s", split_name, output_path)
                results[split_name] = output_path
                continue
            LOGGER.info("Converting %s split to COCO format", split_name)
            coco_payload = self._build_coco_for_split(split_name, images_root, source_annotation)
            write_json(output_path, coco_payload)
            meta = ConversionMetadata(
                dataset_name=self.dataset_name,
                split=split_name,
                source_annotation=str(source_annotation),
                source_mtime=source_annotation.stat().st_mtime,
                images_root=str(images_root),
                class_map=self.class_map,
            )
            write_json(meta_path, meta.to_dict())
            results[split_name] = output_path
            LOGGER.info("Finished conversion for %s split: %s", split_name, output_path)
        return results

    def _should_skip_conversion(self, output_path: Path, meta_path: Path, source_mtime: float) -> bool:
        if self.force:
            return False
        if not output_path.exists() or not meta_path.exists():
            return False
        try:
            metadata = read_json(meta_path)
        except (json.JSONDecodeError, OSError):
            return False
        recorded_mtime = metadata.get("source_mtime")
        return isinstance(recorded_mtime, (float, int)) and abs(float(recorded_mtime) - source_mtime) < 1e-6

    def _build_coco_for_split(self, split: str, images_root: Path, annotation_path: Path) -> Dict[str, object]:
        raise NotImplementedError


class BDD100KToCocoConverter(BaseDatasetToCocoConverter):
    """Converter for the BDD100K detection dataset."""

    CATEGORY_ALIAS: Dict[str, str] = {
        "person": "Person",
        "rider": "Person",
        "car": "Car",
        "bus": "Bus",
        "truck": "Truck",
        "bike": "Bicycle",
        "motor": "Motorcycle",
        "traffic light": "Traffic light",
        "traffic sign": "Traffic sign",
        "caravan": "Caravan",
        "trailer": "Trailer",
    }

    def _build_coco_for_split(self, split: str, images_root: Path, annotation_path: Path) -> Dict[str, object]:
        raw_entries = read_json(annotation_path)
        progress = tqdm(
            raw_entries,
            desc=f"{self.dataset_name} | {split}".strip(),
            unit="image",
            leave=False,
            dynamic_ncols=True,
            total=len(raw_entries),
            colour="cyan",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        images: List[Dict[str, object]] = []
        annotations: List[Dict[str, object]] = []
        image_id = 1
        annotation_id = 1
        for entry in progress:
            file_name = entry.get("name")
            if not file_name:
                continue
            image_path = images_root / file_name
            width, height = self._get_image_dimensions(image_path)
            images.append(
                {
                    "id": image_id,
                    "file_name": str(file_name),
                    "width": width,
                    "height": height,
                }
            )
            labels = entry.get("labels", []) or []
            for label in labels:
                category_name = label.get("category", "").lower()
                mapped_category = self.CATEGORY_ALIAS.get(category_name)
                if not mapped_category:
                    continue
                class_id = self.class_map.get(mapped_category)
                if not class_id:
                    continue
                box = label.get("box2d") or {}
                if not all(k in box for k in ("x1", "y1", "x2", "y2")):
                    continue
                x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
                width_box = max(0.0, x2 - x1)
                height_box = max(0.0, y2 - y1)
                if width_box <= 0 or height_box <= 0:
                    continue
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x1, y1, width_box, height_box],
                        "area": width_box * height_box,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                annotation_id += 1
            image_id += 1
        return {
            "info": {
                "description": f"{self.dataset_name} converted to COCO",
                "url": "https://www.bdd100k.com/",
                "version": "1.0",
                "year": datetime.utcnow().year,
                "contributor": "Gahan AI",
                "date_created": datetime.utcnow().isoformat() + "Z",
            },
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": self._categories,
        }

    @staticmethod
    def _get_image_dimensions(image_path: Path) -> Tuple[int, int]:
        if not image_path.exists():
            LOGGER.debug("Image missing while converting: %s", image_path)
            return 0, 0
        try:
            with Image.open(image_path) as img:
                width, height = img.size
            return int(width), int(height)
        except (OSError, ValueError):
            LOGGER.debug("Failed to read image dimensions for %s", image_path)
            return 0, 0


CONVERTER_REGISTRY: Dict[str, type[BaseDatasetToCocoConverter]] = {
    "bdd100k": BDD100KToCocoConverter,
}


def build_converter_from_config(config: Dict[str, object]) -> Optional[BaseDatasetToCocoConverter]:
    dataset_cfg = config.get("dataset") if isinstance(config, dict) else None
    if not isinstance(dataset_cfg, dict):
        raise ValueError("Config missing 'dataset' section")
    dataset_name = dataset_cfg.get("name", "").lower()
    converter_cls = CONVERTER_REGISTRY.get(dataset_name)
    if not converter_cls:
        LOGGER.warning("No converter available for dataset '%s'", dataset_name)
        return None
    export_dir = dataset_cfg.get("export_coco_path")
    if not export_dir:
        raise ValueError("'export_coco_path' must be provided in dataset config")
    root_dir = dataset_cfg.get("root_dir")
    if not root_dir:
        raise ValueError("'root_dir' must be provided in dataset config")
    class_map = _resolve_class_map(config)
    force = bool(dataset_cfg.get("force_reconvert", False))
    return converter_cls(
        dataset_name=dataset_cfg.get("name", ""),
        root_dir=root_dir,
        export_dir=export_dir,
        class_map=class_map,
        force=force,
    )


def run_conversion(config: Dict[str, object]) -> Dict[str, Path]:
    converter = build_converter_from_config(config)
    if converter is None:
        return {}
    dataset_cfg = config["dataset"]
    splits: Dict[str, Tuple[str, str]] = {}
    for split_key in ("train", "val", "test"):
        split_dir_key = f"{split_key}_split"
        ann_file_key = f"{split_key}_ann_file"
        images_rel = dataset_cfg.get(split_dir_key)
        ann_rel = dataset_cfg.get(ann_file_key)
        if images_rel and ann_rel:
            splits[split_key] = (images_rel, ann_rel)
    if not splits:
        LOGGER.warning("No dataset splits configured for conversion")
        return {}
    return converter.convert(splits)


def _resolve_class_map(config: Dict[str, object]) -> Dict[str, int]:
    # Default class ordering follows repository documentation.
    default_classes = [
        "Person",
        "Car",
        "Bus",
        "Truck",
        "Bicycle",
        "Motorcycle",
        "Caravan",
        "Trailer",
        "Traffic light",
        "Traffic sign",
    ]
    dataset_cfg = config.get("dataset", {})
    custom_map = dataset_cfg.get("class_map") if isinstance(dataset_cfg, dict) else None
    if isinstance(custom_map, dict) and custom_map:
        normalized = {str(k): int(v) for k, v in custom_map.items()}
        return normalized
    return {label: idx + 1 for idx, label in enumerate(default_classes)}