# GAI-YOLOv12 Configuration Guide

Complete reference for configuring the GAI-YOLOv12 training pipeline.

## Table of Contents
- [Dataset Configuration](#dataset-configuration)
- [Model Configuration](#model-configuration)
- [Dataloader Configuration](#dataloader-configuration)
- [Experiment Configuration](#experiment-configuration)
- [TensorBoard Configuration](#tensorboard-configuration)
- [Evaluation Configuration](#evaluation-configuration)
- [Augmentation Configuration](#augmentation-configuration)
- [Complete Example](#complete-example)

---

## Dataset Configuration

Controls dataset paths and COCO format conversion.

```yaml
dataset:
  name: "BDD100K"                                      # Dataset name for logging
  root_dir: "/path/to/dataset/"                        # Root directory of dataset
  train_split: "images/100k/train/"                    # Training images subdirectory
  val_split: "images/100k/val/"                        # Validation images subdirectory
  test_split: "images/100k/test/"                      # Test images subdirectory (optional)
  train_ann_file: "labels/train.json"                  # Training annotations file
  val_ann_file: "labels/val.json"                      # Validation annotations file
  test_ann_file: "labels/test.json"                    # Test annotations file (optional)
  convert_to_coco: true                                # Convert annotations to COCO format
  export_coco_path: "/path/to/coco_format/"           # Output path for COCO conversion
```

**Parameters:**
- `name`: String identifier for your dataset
- `root_dir`: Absolute path to dataset root directory
- `*_split`: Relative paths from root_dir to image folders
- `*_ann_file`: Relative paths from root_dir to annotation files
- `convert_to_coco`: Boolean, whether to convert annotations to COCO format
- `export_coco_path`: Where to save converted COCO format annotations

---

## Model Configuration

Defines the model architecture and backbone selection.

```yaml
model:
  name: "gai-yolo12"                    # Model name
  pretrained: false                     # Load pretrained weights
  num_classes: 10                       # Number of object classes
  input_size: [640, 640]               # Input image size [height, width]
  input_channels: 3                     # Number of input channels (default: 3 for RGB)
  backbone: "tiny_csp"                  # Backbone architecture
  backbone_params:                      # Backbone-specific parameters
    base_channels: 64                   # Base channel width
    depth: 3                            # Number of stages
  head:                                 # Detection head parameters
    hidden_channels: 128                # Hidden layer channels
  checkpoint_path: null                 # Path to checkpoint for initialization
  anchors:                              # Custom anchor boxes (optional)
    - [[10, 13], [16, 30], [33, 23]]   # Small objects
    - [[30, 61], [62, 45], [59, 119]]  # Medium objects
    - [[116, 90], [156, 198], [373, 326]]  # Large objects
  strides: [8, 16, 32]                 # Feature map strides
```

### Backbone Options

**TinyCSPBackbone configurations** (ranked by size):

| Configuration | base_channels | depth | Parameters | Use Case |
|--------------|---------------|-------|------------|----------|
| **Nano** | 32 | 2 | 0.07M | Edge devices, fastest |
| **Tiny** | 64 | 2 | 0.28M | Mobile devices |
| **Small** | 64 | 3 | 1.23M | **Recommended default** |
| **Medium** | 128 | 3 | 4.92M | Higher accuracy |
| **Large** | 128 | 4 | 20.13M | Maximum accuracy |
| **XLarge** | 256 | 4 | 80.51M | Research/benchmarking |

**Example configurations:**

```yaml
# Nano - Fastest inference
model:
  backbone: "tiny_csp"
  backbone_params:
    base_channels: 32
    depth: 2

# Small - Recommended balance
model:
  backbone: "tiny_csp"
  backbone_params:
    base_channels: 64
    depth: 3

# Large - Maximum accuracy
model:
  backbone: "tiny_csp"
  backbone_params:
    base_channels: 128
    depth: 4
```

**Parameters:**
- `name`: Model identifier (currently only "gai-yolo12" supported)
- `pretrained`: Load pretrained weights if available
- `num_classes`: Number of object categories to detect
- `input_size`: Model input dimensions [H, W]
- `backbone`: Backbone architecture name ("tiny_csp")
- `backbone_params.base_channels`: Controls network width (32, 64, 128, 256)
- `backbone_params.depth`: Number of downsampling stages (2, 3, 4)
- `head.hidden_channels`: Detection head hidden layer size

---

## Dataloader Configuration

Controls data loading and augmentation.

```yaml
dataloader:
  batch_size: 64                        # Training batch size
  num_workers: 16                       # Number of data loading workers
  pin_memory: true                      # Pin memory for faster GPU transfer
  augmentation: true                    # Enable data augmentation
  augmentation_strategies:              # Available augmentation types
    - "RandomHorizontalFlip"
    - "RandomCrop"
    - "ColorJitter"
    - "NoiseInjection"
    - "RandomRotation"
    - "MotionBlur"
  augmentation_probabilities_per_batch: 0.2  # Fraction of batch to augment
  augmentations:                        # Detailed augmentation parameters
    - type: "RandomHorizontalFlip"
      probability: 0.5
    - type: "RandomCrop"
      size: [600, 600]
    - type: "ColorJitter"
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    - type: "NoiseInjection"
      mean: 0
      std: 0.05
      p: 0.5
    - type: "RandomRotation"
      degrees: 10
      fill: "black"
    - type: "MotionBlur"
      kernel_size: 5
      p: 0.3
```

**Parameter Recommendations:**

| Parameter | Small Dataset | Medium Dataset | Large Dataset |
|-----------|--------------|----------------|---------------|
| batch_size | 8-16 | 32-64 | 64-128 |
| num_workers | 2-4 | 8-16 | 16-32 |
| augmentation | true | true | true/false |
| augmentation_probabilities_per_batch | 0.3-0.5 | 0.2-0.3 | 0.1-0.2 |

**Parameters:**
- `batch_size`: Images per batch (adjust based on GPU memory)
- `num_workers`: Parallel data loading processes (typically 2-4x num_gpus)
- `pin_memory`: Set true for GPU training, false for CPU
- `augmentation`: Master switch for data augmentation
- `augmentation_strategies`: List of augmentation techniques to use
- `augmentation_probabilities_per_batch`: 0.0-1.0, fraction of batch to augment

---

## Experiment Configuration

Core training parameters and experiment management.

```yaml
experiment:
  name: "gai-yolov12-experiment-{timestamp}"  # Experiment name (supports {timestamp})
  description: "Training on BDD100K dataset"  # Human-readable description
  num_epochs: 50                              # Total training epochs
  learning_rate: 0.001                        # Initial learning rate
  weight_decay: 0.0001                        # L2 regularization weight
  max_batches_per_epoch: null                 # Limit batches per epoch (null = no limit)
  checkpoint_interval: 5                      # Save checkpoint every N epochs
  gradient_clip_norm: 10.0                    # Gradient clipping threshold (null to disable)
  log_interval: 10                            # Log metrics every N batches
  optimizer: "adamw"                          # Optimizer type
  scheduler: null                             # Learning rate scheduler (null to disable)
  scheduler_params: {}                        # Scheduler-specific parameters
  device: "auto"                              # Device: "auto", "cuda", "cpu"
```

### Learning Rate Recommendations

| Dataset Size | Initial LR | Weight Decay | Batch Size |
|-------------|-----------|--------------|------------|
| Small (<5k images) | 0.0001 - 0.0005 | 0.0001 | 8-16 |
| Medium (5k-50k) | 0.0005 - 0.001 | 0.0001 - 0.0005 | 16-64 |
| Large (>50k) | 0.001 - 0.01 | 0.0005 - 0.001 | 64-128 |

### Optimizer Options

```yaml
# AdamW (Recommended)
experiment:
  optimizer: "adamw"
  learning_rate: 0.001
  weight_decay: 0.0001

# SGD with Momentum
experiment:
  optimizer: "sgd"
  learning_rate: 0.01
  weight_decay: 0.0005
```

**Parameters:**
- `name`: Experiment identifier (use {timestamp} for auto-naming)
- `num_epochs`: Total training epochs
- `learning_rate`: Initial LR (typically 1e-4 to 1e-2)
- `weight_decay`: L2 regularization (typically 1e-5 to 1e-3)
- `max_batches_per_epoch`: Limit batches for faster debugging (null for full dataset)
- `checkpoint_interval`: Frequency of checkpoint saves
- `gradient_clip_norm`: Prevent gradient explosion (typically 1.0-10.0)

---

## TensorBoard Configuration

Configure TensorBoard logging and visualization.

```yaml
experiment:
  tensorboard:
    enabled: true                   # Enable TensorBoard logging
    log_interval: 10                # Log scalars every N batches
    log_images: true                # Enable image logging
    images_per_batch: 2             # Number of images to log per batch
    image_log_interval: 100         # Log images every N batches
    show_ground_truth: true         # Show GT boxes in visualizations
    show_predictions: true          # Show prediction boxes in visualizations
```

**Visualization Options:**
- `show_ground_truth`: Display ground truth bounding boxes (green)
- `show_predictions`: Display model predictions (red)

You can configure different visualization modes:

| show_ground_truth | show_predictions | Use Case |
|------------------|------------------|----------|
| true | true | **Default**: Compare GT vs predictions |
| false | true | Only predictions (cleaner for presentation) |
| true | false | Only GT (verify data quality) |
| false | false | No boxes (just images) |

**Visualization Features:**
- ✅ Training/validation loss curves
- ✅ Learning rate tracking
- ✅ COCO evaluation metrics (mAP, mAR)
- ✅ Image visualizations with bounding boxes
  - Green boxes: Ground truth annotations
  - Red boxes: Model predictions with confidence scores

**Parameter Recommendations:**

| Scenario | log_images | images_per_batch | image_log_interval |
|----------|-----------|------------------|-------------------|
| Quick debugging | true | 4-8 | 10-50 |
| Normal training | true | 2-4 | 100-200 |
| Production | false | 2 | 500-1000 |

**Parameters:**
- `enabled`: Turn TensorBoard logging on/off
- `log_interval`: Frequency of scalar logging (loss, LR, etc.)
- `log_images`: Enable image visualization (can be storage-intensive)
- `images_per_batch`: Images to visualize per logging interval
- `image_log_interval`: How often to log images (higher = less storage)
- `show_ground_truth`: Display ground truth boxes (green, thick lines)
- `show_predictions`: Display model predictions (red, thin lines with confidence scores)

**Starting TensorBoard:**
```bash
# Automatic (starts server and opens browser)
uv run main.py --tensorboard

# Manual
tensorboard --logdir ./experiments/your-experiment-name/tensorboard/
```

---

## Evaluation Configuration

Configure COCO metrics evaluation during training.

```yaml
experiment:
  evaluation:
    enabled: true                   # Enable COCO evaluation
    eval_interval: 5                # Evaluate every N epochs
    eval_on_validation: true        # Use validation set for evaluation
    visualization_conf_threshold: 0.01  # Confidence threshold for visualization (0.001-1.0)
```

**Visualization Confidence Threshold:**
Controls which predictions are displayed in TensorBoard images and used for COCO evaluation. Lower values show more predictions (including uncertain ones), higher values show only confident predictions.

| Threshold | Use Case |
|-----------|----------|
| 0.001 - 0.01 | Early training, debugging (see all predictions) |
| 0.1 - 0.3 | Mid training (filter obvious noise) |
| 0.5 - 0.9 | Final evaluation (high confidence only) |

**COCO Metrics Computed:**
- `mAP` - Mean Average Precision @ IoU 0.5:0.95
- `mAP_50` - Mean Average Precision @ IoU 0.5
- `mAP_75` - Mean Average Precision @ IoU 0.75
- `mAP_small` - mAP for small objects
- `mAP_medium` - mAP for medium objects
- `mAP_large` - mAP for large objects
- `mAR_1` - Mean Average Recall with 1 detection per image
- `mAR_10` - Mean Average Recall with 10 detections per image
- `mAR_100` - Mean Average Recall with 100 detections per image
- `mAR_small` - mAR for small objects
- `mAR_medium` - mAR for medium objects
- `mAR_large` - mAR for large objects

**Evaluation Frequency Recommendations:**

| Training Duration | eval_interval |
|------------------|---------------|
| Short (<20 epochs) | 1-2 |
| Medium (20-100 epochs) | 5-10 |
| Long (>100 epochs) | 10-20 |

**Parameters:**
- `enabled`: Turn COCO evaluation on/off
- `eval_interval`: How often to run evaluation (balance accuracy vs speed)
- `eval_on_validation`: Use validation set (true) or training set (false)
- `visualization_conf_threshold`: Minimum confidence to show predictions (0.001-1.0)
  - Lower values (0.001-0.01): Show all predictions, useful early in training
  - Medium values (0.1-0.3): Filter noise, typical mid-training
  - High values (0.5-0.9): Only confident predictions, final evaluation

---

## Augmentation Configuration

Data augmentation is configured under the `dataloader` section.

### Available Augmentations

#### 1. RandomHorizontalFlip
```yaml
- type: "RandomHorizontalFlip"
  probability: 0.5          # 50% chance to flip
```

#### 2. RandomCrop
```yaml
- type: "RandomCrop"
  size: [600, 600]         # Crop size [H, W]
```

#### 3. ColorJitter
```yaml
- type: "ColorJitter"
  brightness: 0.2          # ±20% brightness change
  contrast: 0.2            # ±20% contrast change
  saturation: 0.2          # ±20% saturation change
  hue: 0.1                 # ±10% hue shift
```

#### 4. NoiseInjection
```yaml
- type: "NoiseInjection"
  mean: 0                  # Gaussian noise mean
  std: 0.05                # Gaussian noise std
  p: 0.5                   # Probability to apply
```

#### 5. RandomRotation
```yaml
- type: "RandomRotation"
  degrees: 10              # Rotation range ±10°
  fill: "black"            # Fill color for empty areas
```

#### 6. MotionBlur
```yaml
- type: "MotionBlur"
  kernel_size: 5           # Blur kernel size
  p: 0.3                   # Probability to apply
```

### Augmentation Strategies by Task

**General Object Detection:**
```yaml
dataloader:
  augmentation: true
  augmentation_probabilities_per_batch: 0.2
  augmentations:
    - type: "RandomHorizontalFlip"
      probability: 0.5
    - type: "ColorJitter"
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
```

**Autonomous Driving (BDD100K):**
```yaml
dataloader:
  augmentation: true
  augmentation_probabilities_per_batch: 0.3
  augmentations:
    - type: "RandomHorizontalFlip"
      probability: 0.5
    - type: "ColorJitter"
      brightness: 0.3
      contrast: 0.3
      saturation: 0.3
      hue: 0.1
    - type: "MotionBlur"
      kernel_size: 5
      p: 0.2
    - type: "NoiseInjection"
      mean: 0
      std: 0.03
      p: 0.3
```

**Small Dataset (aggressive augmentation):**
```yaml
dataloader:
  augmentation: true
  augmentation_probabilities_per_batch: 0.5
  augmentations:
    - type: "RandomHorizontalFlip"
      probability: 0.5
    - type: "RandomRotation"
      degrees: 15
      fill: "black"
    - type: "ColorJitter"
      brightness: 0.3
      contrast: 0.3
      saturation: 0.3
      hue: 0.15
    - type: "RandomCrop"
      size: [600, 600]
```

---

## Complete Example

### Quick Start (Debugging)
```yaml
dataset:
  name: "BDD100K"
  root_dir: "/path/to/bdd100k/"
  train_split: "images/100k/train/"
  val_split: "images/100k/val/"
  train_ann_file: "labels/train.json"
  val_ann_file: "labels/val.json"
  convert_to_coco: true
  export_coco_path: "/path/to/coco_format/"

model:
  name: "gai-yolo12"
  pretrained: false
  num_classes: 10
  input_size: [640, 640]
  backbone: "tiny_csp"
  backbone_params:
    base_channels: 32      # Nano for fast debugging
    depth: 2

dataloader:
  batch_size: 8
  num_workers: 4
  pin_memory: true
  augmentation: false      # Disable for debugging

experiment:
  name: "debug-{timestamp}"
  num_epochs: 5
  learning_rate: 0.001
  weight_decay: 0.0001
  max_batches_per_epoch: 20  # Only 20 batches per epoch
  checkpoint_interval: 1
  gradient_clip_norm: 10.0
  log_interval: 5
  tensorboard:
    enabled: true
    log_interval: 5
    log_images: true
    images_per_batch: 4
    image_log_interval: 10
  evaluation:
    enabled: true
    eval_interval: 1
    eval_on_validation: true
```

### Production Training
```yaml
dataset:
  name: "BDD100K"
  root_dir: "/path/to/bdd100k/"
  train_split: "images/100k/train/"
  val_split: "images/100k/val/"
  train_ann_file: "labels/train.json"
  val_ann_file: "labels/val.json"
  convert_to_coco: true
  export_coco_path: "/path/to/coco_format/"

model:
  name: "gai-yolo12"
  pretrained: false
  num_classes: 10
  input_size: [640, 640]
  backbone: "tiny_csp"
  backbone_params:
    base_channels: 128     # Large for accuracy
    depth: 4

dataloader:
  batch_size: 64
  num_workers: 16
  pin_memory: true
  augmentation: true
  augmentation_probabilities_per_batch: 0.2
  augmentations:
    - type: "RandomHorizontalFlip"
      probability: 0.5
    - type: "ColorJitter"
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    - type: "NoiseInjection"
      mean: 0
      std: 0.05
      p: 0.5
    - type: "MotionBlur"
      kernel_size: 5
      p: 0.3

experiment:
  name: "production-large-{timestamp}"
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  max_batches_per_epoch: null  # Use full dataset
  checkpoint_interval: 10
  gradient_clip_norm: 10.0
  log_interval: 50
  tensorboard:
    enabled: true
    log_interval: 50
    log_images: true
    images_per_batch: 2
    image_log_interval: 500
  evaluation:
    enabled: true
    eval_interval: 10
    eval_on_validation: true
```

---

## Command Line Usage

### Basic Training
```bash
# Use default config.yaml
uv run main.py

# Use custom config
uv run main.py --config my_config.yaml

# Resume from checkpoint
uv run main.py --config my_config.yaml --resume
```

### TensorBoard
```bash
# Start TensorBoard server
uv run main.py --tensorboard

# Custom port
uv run main.py --tensorboard --tensorboard-port 8080

# Custom host
uv run main.py --tensorboard --tensorboard-host 0.0.0.0
```

### Dataset Conversion Only
```bash
# Convert dataset to COCO format and exit
uv run main.py --only-convert
```

### Debugging Options
```bash
# Set log level
uv run main.py --log-level DEBUG
```

---

## Tips and Best Practices

### 1. Starting a New Project
- Begin with **Nano** or **Tiny** backbone for quick iteration
- Set `max_batches_per_epoch: 20-50` for fast debugging
- Enable `tensorboard.log_images` to verify data pipeline
- Use `eval_interval: 1` to monitor metrics every epoch

### 2. Hyperparameter Tuning
- Start with learning rate: `1e-3` (AdamW) or `1e-2` (SGD)
- Adjust batch size based on GPU memory (larger is generally better)
- Increase model size gradually: Tiny → Small → Medium → Large
- Enable augmentation only after baseline is working

### 3. Production Training
- Use full dataset: `max_batches_per_epoch: null`
- Save checkpoints frequently: `checkpoint_interval: 5-10`
- Reduce image logging: `image_log_interval: 500-1000`
- Monitor validation metrics to prevent overfitting

### 4. GPU Memory Optimization
If you run out of GPU memory:
- Reduce `batch_size` (32 → 16 → 8)
- Use smaller backbone (Large → Medium → Small)
- Reduce `input_size` (640 → 512 → 416)
- Decrease `num_workers` if using pin_memory

### 5. Training Speed Optimization
- Use `pin_memory: true` for GPU training
- Set `num_workers` to 2-4x number of GPUs
- Disable image logging in production
- Use `max_batches_per_epoch` for quick experiments

---

## Troubleshooting

### Training is too slow
- Reduce `max_batches_per_epoch` for faster epochs
- Decrease `eval_interval` to evaluate less frequently
- Disable `tensorboard.log_images`
- Use smaller backbone configuration

### Loss is NaN or exploding
- Reduce `learning_rate` (try 1e-4 or 1e-5)
- Enable or decrease `gradient_clip_norm` (5.0 or 1.0)
- Check data preprocessing and normalization
- Reduce batch size to improve stability

### Validation mAP is very low
- Check `tensorboard.log_images` to verify predictions
- Increase model capacity (larger backbone)
- Train for more epochs
- Verify data annotations are correct
- Try different augmentation strategies

### Out of memory errors
- Reduce `batch_size`
- Use smaller backbone configuration
- Reduce `input_size`
- Set `num_workers` to lower value

---

## Advanced Configuration

### Custom Loss Configuration
```yaml
model:
  loss:
    type: "yolo_loss"
    lambda_coord: 5.0
    lambda_noobj: 0.5
    lambda_obj: 1.0
    lambda_class: 1.0
```

### Custom Anchor Boxes
```yaml
model:
  anchors:
    - [[10, 13], [16, 30], [33, 23]]      # Small objects
    - [[30, 61], [62, 45], [59, 119]]     # Medium objects  
    - [[116, 90], [156, 198], [373, 326]] # Large objects
  strides: [8, 16, 32]
```

### Learning Rate Scheduler
```yaml
experiment:
  scheduler: "cosine"
  scheduler_params:
    T_max: 100
    eta_min: 0.00001
```

---

## Output Files

After training, your experiment directory will contain:

```
experiments/
└── your-experiment-name/
    ├── checkpoints/
    │   ├── epoch_005.pt
    │   ├── epoch_010.pt
    │   └── training_metrics.json
    ├── metadata/
    │   ├── config.json
    │   ├── model_summary.json
    │   ├── experiment_info.json
    │   ├── training_history.json
    │   └── final_metrics.json
    └── tensorboard/
        └── events.out.tfevents.*
```

---

## Further Reading

- [Resume Training Guide](../RESUME_TRAINING.md)
- [TensorBoard Visualization](https://www.tensorflow.org/tensorboard)
- [COCO Evaluation Metrics](https://cocodataset.org/#detection-eval)
- [Data Augmentation Best Practices](https://arxiv.org/abs/1906.11172)
