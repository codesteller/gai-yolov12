'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-10-30 15:48:40
 # @ Modified time: 2025-10-31 10:30:00
 # @ Description: Config-driven COCO dataloader utilities with augmentation support for GAI-YOLOv12 training.
 # @ Description (Legacy): This is a object detection dataloader utility module. This module 
 #      1. loads data for training and evaluation of the model.
 #      2. applies data augmentation techniques to the input data.
 #      3. prepares data for feeding into the model during training and evaluation.
 #      4. Converts Datasets from various formats to COCO format and saves them.
'''

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from .utils import ensure_dir, write_json

LOGGER = logging.getLogger("gai_yolov12.dataloader")


@dataclass
class DatasetSplit:
    """Configuration for a single dataset split."""

    name: str
    images_dir: Path
    annotation_path: Path
    apply_augmentation: bool = False


class AlbumentationsPipelines:
    """Wrapper that applies either the base or a randomly selected augmentation pipeline."""

    def __init__(
        self,
        base: A.BasicTransform,
        candidates: List[A.BasicTransform],
        augmentation_probability: float,
    ) -> None:
        self.base = base
        self.candidates = candidates
        self.augmentation_probability = max(0.0, min(augmentation_probability, 1.0))

    def select(self, use_aug: bool) -> A.BasicTransform:
        if use_aug and self.candidates and random.random() <= self.augmentation_probability:
            return random.choice(self.candidates)
        return self.base


class COCODataset(Dataset):
    """PyTorch dataset wrapping COCO-formatted annotations."""

    def __init__(
        self,
        split: DatasetSplit,
        pipelines: AlbumentationsPipelines,
        category_mapping: Dict[int, int],
    ) -> None:
        self.split = split
        self.pipelines = pipelines
        self.images_dir = split.images_dir
        self.annotation_path = split.annotation_path
        self.coco = COCO(str(self.annotation_path))
        self.image_ids = self.coco.getImgIds()
        self.category_mapping = category_mapping
        self.inverse_category_mapping = {v: k for k, v in category_mapping.items()}

    def __len__(self) -> int:  # noqa: D401 - standard Dataset contract
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.image_ids[index]
        img_info = self.coco.loadImgs([image_id])[0]
        image_path = self.images_dir / img_info["file_name"]
        image = self._load_image(image_path)
        annotations = self._load_annotations(image_id)

        bboxes = [ann["bbox"] for ann in annotations]
        pascal_bboxes = [self._convert_bbox_xywh_to_pascal(bbox) for bbox in bboxes]
        labels = [self.category_mapping[ann["category_id"]] for ann in annotations]
        iscrowd = [int(ann.get("iscrowd", 0)) for ann in annotations]

        transform = self.pipelines.select(self.split.apply_augmentation)
        transformed = transform(
            image=image,
            bboxes=pascal_bboxes,
            labels=labels,
            iscrowd=iscrowd,
        )

        image_tensor = transformed["image"]
        boxes_tensor = self._to_tensor(transformed["bboxes"], dtype=torch.float32, shape=(-1, 4))
        labels_tensor = self._to_tensor(transformed["labels"], dtype=torch.int64)
        iscrowd_tensor = self._to_tensor(transformed.get("iscrowd", []), dtype=torch.int64)

        area_tensor = self._compute_area(boxes_tensor)
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
            "orig_size": torch.tensor([img_info["height"], img_info["width"]], dtype=torch.int64),
            "size": torch.tensor(list(image_tensor.shape[1:]), dtype=torch.int64),
        }
        return image_tensor, target

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Image not found for dataset item: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_annotations(self, image_id: int) -> List[Dict[str, Any]]:
        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        annotations = self.coco.loadAnns(ann_ids)
        filtered: List[Dict[str, Any]] = []
        for ann in annotations:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            if ann["category_id"] not in self.category_mapping:
                continue
            filtered.append(ann)
        return filtered

    @staticmethod
    def _convert_bbox_xywh_to_pascal(bbox: Iterable[float]) -> Tuple[float, float, float, float]:
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        return float(x_min), float(y_min), float(x_max), float(y_max)

    @staticmethod
    def _to_tensor(values: Iterable[Any], dtype: torch.dtype, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            tensor = values.to(dtype)
            if shape is not None:
                return tensor.view(shape)
            return tensor
        if isinstance(values, np.ndarray):
            array = values.astype(np.float32 if dtype.is_floating_point else np.int64, copy=False)
        else:
            array = np.array(list(values), dtype=np.float32 if dtype.is_floating_point else np.int64)
        if array.size == 0:
            if shape is None:
                return torch.zeros((0,), dtype=dtype)
            adjusted_shape = tuple(0 if dim == -1 else dim for dim in shape)
            return torch.zeros(adjusted_shape, dtype=dtype)
        tensor = torch.as_tensor(array, dtype=dtype)
        if shape is not None:
            tensor = tensor.view(shape)
        return tensor

    @staticmethod
    def _compute_area(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32)
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        return widths * heights


class BatchAugmentor:
    """Applies batch-level augmentations such as MixUp and CutMix."""

    def __init__(self, mixup_cfg: Optional[Dict[str, Any]], cutmix_cfg: Optional[Dict[str, Any]]) -> None:
        self.mixup_cfg = mixup_cfg
        self.cutmix_cfg = cutmix_cfg

    def __call__(self, images: torch.Tensor, targets: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        if self.mixup_cfg and len(images) > 1 and random.random() <= float(self.mixup_cfg.get("p", 0.0)):
            images, targets = self._apply_mixup(images, targets, float(self.mixup_cfg.get("alpha", 0.4)))
        if self.cutmix_cfg and len(images) > 1 and random.random() <= float(self.cutmix_cfg.get("p", 0.0)):
            images, targets = self._apply_cutmix(images, targets, float(self.cutmix_cfg.get("alpha", 1.0)))
        return images, targets

    def _apply_mixup(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        alpha: float,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        lam = np.random.beta(alpha, alpha)
        perm = torch.randperm(images.size(0))
        mixed_images = lam * images + (1.0 - lam) * images[perm]
        mixed_targets: List[Dict[str, torch.Tensor]] = []
        for idx, perm_idx in enumerate(perm):
            primary = targets[idx]
            secondary = targets[int(perm_idx.item()) if hasattr(perm_idx, "item") else int(perm_idx)]
            boxes = torch.cat([primary["boxes"], secondary["boxes"]], dim=0)
            labels = torch.cat([primary["labels"], secondary["labels"]], dim=0)
            iscrowd = torch.cat([primary["iscrowd"], secondary["iscrowd"]], dim=0)
            area = COCODataset._compute_area(boxes)
            mixed_targets.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                    "image_id": primary["image_id"],
                    "area": area,
                    "iscrowd": iscrowd,
                    "orig_size": primary["orig_size"],
                    "size": primary["size"],
                }
            )
        return mixed_images, mixed_targets

    def _apply_cutmix(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        alpha: float,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        batch_size, _, height, width = images.shape
        perm = torch.randperm(batch_size)
        lam = np.random.beta(alpha, alpha)

        mixed_images = images.clone()
        mixed_targets: List[Dict[str, torch.Tensor]] = []
        for idx in range(batch_size):
            perm_idx = int(perm[idx].item()) if hasattr(perm[idx], "item") else int(perm[idx])
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(width, height, lam)
            mixed_images[idx, :, bby1:bby2, bbx1:bbx2] = images[perm_idx, :, bby1:bby2, bbx1:bbx2]
            primary = targets[idx]
            secondary = targets[perm_idx]
            boxes = torch.cat([primary["boxes"], secondary["boxes"]], dim=0)
            labels = torch.cat([primary["labels"], secondary["labels"]], dim=0)
            iscrowd = torch.cat([primary["iscrowd"], secondary["iscrowd"]], dim=0)
            area = COCODataset._compute_area(boxes)
            mixed_targets.append(
                {
                    "boxes": boxes,
                    "labels": labels,
                    "image_id": primary["image_id"],
                    "area": area,
                    "iscrowd": iscrowd,
                    "orig_size": primary["orig_size"],
                    "size": primary["size"],
                }
            )
        return mixed_images, mixed_targets

    @staticmethod
    def _rand_bbox(width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
        cut_ratio = math.sqrt(1.0 - lam)
        cut_w = max(1, int(width * cut_ratio)) if width > 1 else width
        cut_h = max(1, int(height * cut_ratio)) if height > 1 else height
        cx = random.randint(0, max(width - 1, 0)) if width > 0 else 0
        cy = random.randint(0, max(height - 1, 0)) if height > 0 else 0

        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bby2 = np.clip(cy + cut_h // 2, 0, height)
        return int(bbx1), int(bby1), int(bbx2), int(bby2)


class DetectionCollate:
    """Collate function that supports optional batch augmentations."""

    def __init__(self, batch_augmentor: Optional[BatchAugmentor] = None) -> None:
        self.batch_augmentor = batch_augmentor

    def __call__(self, batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        images, targets = zip(*batch)
        images_tensor = torch.stack(list(images), dim=0)
        targets_list = [dict(item) for item in targets]
        if self.batch_augmentor is not None:
            images_tensor, targets_list = self.batch_augmentor(images_tensor, targets_list)
        return images_tensor, targets_list


def create_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    dataset_cfg = config.get("dataset", {})
    dataloader_cfg = config.get("dataloader", {})
    experiment_cfg = config.get("experiment", {})
    model_cfg = config.get("model", {})

    dataset_root = Path(dataset_cfg.get("root_dir", "")).expanduser()
    export_dir = Path(dataset_cfg.get("export_coco_path", "")).expanduser()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}")
    if not export_dir.exists():
        raise FileNotFoundError(f"COCO export directory does not exist: {export_dir}")

    input_size = tuple(model_cfg.get("input_size", [640, 640]))
    base_transform, aug_candidates, aug_prob = build_transforms(dataloader_cfg, input_size)
    pipelines = AlbumentationsPipelines(base_transform, aug_candidates, aug_prob)

    category_mapping = _build_category_mapping(export_dir / "train.json")
    batch_augmentor = build_batch_augmentor(dataloader_cfg)

    dataloaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        split_ann = export_dir / f"{split}.json"
        images_rel = dataset_cfg.get(f"{split}_split")
        if not images_rel or not split_ann.is_file():
            LOGGER.warning("Skipping %s split: missing images path or annotation file", split)
            continue
        images_dir = _resolve_split_path(dataset_root, images_rel)
        apply_aug = bool(dataloader_cfg.get("augmentation", False)) and split == "train"
        split_config = DatasetSplit(split, images_dir, split_ann, apply_aug)
        dataset = COCODataset(split_config, pipelines, category_mapping)
        collate = DetectionCollate(batch_augmentor if split == "train" else None)
        
        # Build DataLoader with performance optimizations
        num_workers = int(dataloader_cfg.get("num_workers", 4))
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": int(dataloader_cfg.get("batch_size", 16)),
            "shuffle": split == "train",
            "num_workers": num_workers,
            "pin_memory": bool(dataloader_cfg.get("pin_memory", True)),
            "collate_fn": collate,
            "drop_last": split == "train",
        }
        
        # Add prefetch_factor and persistent_workers only if num_workers > 0
        if num_workers > 0:
            if "prefetch_factor" in dataloader_cfg:
                dataloader_kwargs["prefetch_factor"] = int(dataloader_cfg["prefetch_factor"])
            if "persistent_workers" in dataloader_cfg:
                dataloader_kwargs["persistent_workers"] = bool(dataloader_cfg["persistent_workers"])
        
        dataloaders[split] = DataLoader(**dataloader_kwargs)
        LOGGER.info(
            "Prepared %s dataloader: %d samples, batch_size=%s, augmentations=%s",
            split,
            len(dataset),
            dataloader_cfg.get("batch_size", 16),
            apply_aug,
        )

    artifact_dir = Path(experiment_cfg.get("artifact_dir", "./experiments/"))
    snapshot_dataloader_config(dataloader_cfg, artifact_dir)
    return dataloaders


def build_transforms(dataloader_cfg: Dict[str, Any], input_size: Tuple[int, int]) -> Tuple[A.BasicTransform, List[A.BasicTransform], float]:
    height, width = int(input_size[1]), int(input_size[0])
    mean = dataloader_cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std = dataloader_cfg.get("normalize_std", [0.229, 0.224, 0.225])
    per_sample_prob = float(dataloader_cfg.get("augmentation_probabilities_per_batch", 1.0))

    resize = A.Resize(height=height, width=width)
    normalize = A.Normalize(mean=mean, std=std)
    to_tensor = ToTensorV2()
    bbox_params = A.BboxParams(format="pascal_voc", label_fields=["labels", "iscrowd"], min_visibility=0.0)

    base_transform = A.Compose([resize, normalize, to_tensor], bbox_params=bbox_params)

    candidates: List[A.BasicTransform] = []
    if dataloader_cfg.get("augmentation", False):
        for aug_cfg in dataloader_cfg.get("augmentations", []):
            aug_type = aug_cfg.get("type")
            if aug_type in {"MixUp", "CutMix"}:
                continue
            built = build_augmentation(aug_cfg)
            if built is None:
                LOGGER.warning("Unknown or unsupported augmentation configuration: %s", aug_cfg)
                continue
            composed = A.Compose([built, resize, normalize, to_tensor], bbox_params=bbox_params)
            candidates.append(composed)

    return base_transform, candidates, per_sample_prob


def build_augmentation(cfg: Dict[str, Any]) -> Optional[A.BasicTransform]:
    aug_type = cfg.get("type")
    if aug_type == "RandomHorizontalFlip":
        probability = float(cfg.get("probability", 0.5))
        return A.HorizontalFlip(p=probability)
    if aug_type == "RandomCrop":
        size = cfg.get("size", [640, 640])
        return A.RandomCrop(height=int(size[0]), width=int(size[1]), p=float(cfg.get("p", 1.0)))
    if aug_type == "ColorJitter":
        return A.ColorJitter(
            brightness=float(cfg.get("brightness", 0.2)),
            contrast=float(cfg.get("contrast", 0.2)),
            saturation=float(cfg.get("saturation", 0.2)),
            hue=float(cfg.get("hue", 0.1)),
            p=float(cfg.get("p", 1.0)),
        )
    if aug_type == "NoiseInjection":
        std = float(cfg.get("std", 0.05))
        mean = float(cfg.get("mean", 0.0))
        std_range = (std, std)
        mean_range = (mean, mean)
        return A.GaussNoise(std_range=std_range, mean_range=mean_range, p=float(cfg.get("p", 1.0)))
    if aug_type == "RandomRotation":
        degrees = float(cfg.get("degrees", 10))
        fill_color = cfg.get("fill", 0)
        if isinstance(fill_color, str) and fill_color.lower() == "black":
            fill_color = 0
        return A.Rotate(limit=degrees, border_mode=cv2.BORDER_CONSTANT, fill=fill_color, p=float(cfg.get("p", 1.0)))
    if aug_type == "Cutout":
        max_holes = int(cfg.get("num_holes", 1))
        max_height = int(cfg.get("max_h_size", 50))
        max_width = int(cfg.get("max_w_size", 50))
        fill_value = float(cfg.get("fill_value", 0))
        return A.CoarseDropout(
            num_holes_range=(max_holes, max_holes),
            hole_height_range=(max_height, max_height),
            hole_width_range=(max_width, max_width),
            fill=fill_value,
            p=float(cfg.get("p", 1.0)),
        )
    if aug_type == "MotionBlur":
        kernel_size = int(cfg.get("kernel_size", 5))
        return A.MotionBlur(blur_limit=(kernel_size, kernel_size), p=float(cfg.get("p", 1.0)))
    return None


def build_batch_augmentor(dataloader_cfg: Dict[str, Any]) -> Optional[BatchAugmentor]:
    if not dataloader_cfg.get("augmentation", False):
        return None
    mixup_cfg = None
    cutmix_cfg = None
    for aug_cfg in dataloader_cfg.get("augmentations", []):
        if aug_cfg.get("type") == "MixUp":
            mixup_cfg = aug_cfg
        if aug_cfg.get("type") == "CutMix":
            cutmix_cfg = aug_cfg
    if mixup_cfg is None and cutmix_cfg is None:
        return None
    return BatchAugmentor(mixup_cfg, cutmix_cfg)


def snapshot_dataloader_config(dataloader_cfg: Dict[str, Any], artifact_dir: Path) -> None:
    try:
        target_dir = ensure_dir(artifact_dir / "metadata")
        snapshot_path = target_dir / "dataloader_config_snapshot.json"
        write_json(snapshot_path, dataloader_cfg)
        LOGGER.info("Saved dataloader configuration snapshot to %s", snapshot_path)
    except OSError as exc:
        LOGGER.warning("Failed to persist dataloader config snapshot: %s", exc)


def _build_category_mapping(train_annotation_path: Path) -> Dict[int, int]:
    if not train_annotation_path.is_file():
        raise FileNotFoundError(f"Training annotation file missing for category mapping: {train_annotation_path}")
    coco = COCO(str(train_annotation_path))
    cat_ids = sorted(coco.getCatIds())
    mapping = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    LOGGER.info("Resolved %d categories from training annotations", len(mapping))
    return mapping


def _resolve_split_path(root: Path, rel_path: str) -> Path:
    rel = Path(rel_path.strip("/"))
    images_dir = (root / rel).expanduser()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return images_dir


__all__ = [
    "create_dataloaders",
    "build_transforms",
    "build_augmentation",
    "build_batch_augmentor",
    "COCODataset",
]
