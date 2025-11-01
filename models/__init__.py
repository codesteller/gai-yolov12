'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-10-31 11:20:00
 # @ Modified time: 2025-10-31 11:20:00
 # @ Description: Public interface for the GAI-YOLOv12 model factory utilities.
 # @ Description (Legacy): This module exposes helper functions to create and configure
 #      GAI-YOLOv12 model instances along with the associated loss functions.
'''

from .factory import ModelBundle, ModelConfig, create_model
from .targets import YoloScaleTargets
from .yolov12 import DetectionScaleOutput

__all__ = [
    "ModelBundle",
    "ModelConfig",
    "create_model",
    "DetectionScaleOutput",
    "YoloScaleTargets",
]
