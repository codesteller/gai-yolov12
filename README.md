# GAI-YOLOv12 Train and Evaluate
This repository contains the code and instructions to train and evaluate the GAI-YOLOv12 model, an advanced object detection model based on the custom YOLO v12 architecture.

Datasets supported:
- [ ] BDD
- [ ] Pascal VOC
- [ ] Waymo
- [ ] Appoloscape
- [ ] Cityscapes
- [ ] KITTI
- [ ] EuroCityPersons (Custom Dataset)
- [ ] Meteor Dataset (Custom Dataset)

Classes supported:
- Person [01]
- Car [02]
- Bus [03]
- Truck [04]
- Bicycle [05]
- Motorcycle [06]
- Caravan [07]
- Trailer [08]
- Traffic light [09]
- Traffic sign [10]


## Modules
- 'dataloaders': Contains data loading utilities and custom dataset classes. This is a object detection dataloader utility module. This module 
    1. loads data for training and evaluation of the model.
    2. Applies data augmentation techniques to the input data.
    3. Prepares data for feeding into the model during training and evaluation.
    4. Converts Datasets from various formats (BDD, Pascal VOC, Waymo, Appoloscape, Cityscapes, KITTI, Custom Dataset) to COCO format and saves them.
- 'models': Contains the GAI-YOLOv12 model architecture and related components. This module 
    1. Implements loading pretrained weights for transfer learning using popular backbones. 
    2. Defines the architecture of the GAI-YOLOv12 model.
    3. Includes layers, activation functions, and other building blocks of the model.
    4. Provides utilities for model initialization and weight loading.
    5. Includes custom loss functions specific to object detection tasks.
- 'train': Contains training scripts and utilities for training the GAI-YOLOv12 model. This module 
    1. Implements the training loop for the model.
    2. Handles optimization, learning rate scheduling, and checkpointing.
    3. Logs training metrics and progress.
    4. Provides options for hyperparameter tuning and configuration.
    5. Logs training progress and metrics using TensorBoard along with saving model checkpoints at regular intervals and 5 images with predicted bounding boxes after every epoch to visualize the model's performance and debug improvements.
- 'evaluate': Contains evaluation scripts and utilities for assessing the performance of the GAI-YOLOv12 model. This module 
    1. Implements evaluation metrics such as mAP (mean Average Precision).
    2. Loads trained model checkpoints for evaluation.
    3. Generates evaluation reports and visualizations.
    4. Provides options for evaluating on different datasets and configurations.
    5. Visualizes evaluation results by saving images with predicted bounding boxes and class labels to help assess model performance qualitatively.
- 'utils': Contains utility functions and helper scripts used across the repository. This module
    1. Implements common functions for data processing, logging, and configuration management.
    2. Provides utilities for file handling, visualization, and other repetitive tasks.
    3. Includes functions for non-maximum suppression (NMS) and bounding box manipulation.
    4. Implements functions for calculating evaluation metrics like IoU, precision, recall, and mAP.
    5. Provides visualization utilities to draw bounding boxes and labels on images for better understanding of model predictions.
- 'configs': Contains configuration files for training and evaluation settings. This module
    1. Implements YAML/JSON configuration files to specify hyperparameters, dataset paths, and model settings.
    2. Provides a structured way to manage different experiment setups.
    3. Includes default configurations for common use cases.
    4. Allows easy modification of training and evaluation parameters without changing the code.
    5. Supports command-line overrides for configuration parameters to facilitate quick experimentation.

## Requirements
- Python 3.12
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- matplotlib >= 3.3.0
- tqdm >= 4.62.0
- PyYAML >= 5.4.0
- tensorboard >= 2.7.0
- Pillow >= 8.3.0
- scipy >= 1.7.0
- ultralytics>=8.3.223

## Installation
Using UV:
```bash
uv sync
```

Or using pip with the provided pyproject.toml:
```bash
pip install .
```