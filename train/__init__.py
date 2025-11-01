'''
 #  Copyright (c) 2025 Gahan AI Private Limited
 #  Author: Pallab Maji
 #  Create Time: 2025-11-01 09:10:00
 #  Modified time: 2025-11-01 09:10:00
 #  Description: Public API for the training engine utilities.
 #  Description (Legacy): Exposes helpers to construct trainer configuration,
 #       orchestrate experiment loops, and manage training artifacts.
'''

from .engine import (
    EpochMetrics,
    Trainer,
    TrainerConfig,
    TrainingResult,
    YoloGridAssigner,
    run_training,
)

__all__ = [
    "TrainerConfig",
    "Trainer",
    "EpochMetrics",
    "TrainingResult",
    "YoloGridAssigner",
    "run_training",
]
