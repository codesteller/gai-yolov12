# Model Metadata Generation

The GAI-YOLOv12 training system automatically generates comprehensive metadata for every experiment, providing detailed insights into model architecture, parameters, and training configuration.

## Generated Metadata Files

When training is started, the following metadata files are automatically created in the `{experiment_dir}/metadata/` directory:

### 1. `model_summary.json`
Comprehensive model architecture summary including:
- **Model Architecture**: Main model class name (e.g., "GAIYoloV12")
- **Backbone Type**: Backbone architecture used (e.g., "TinyCSPBackbone")
- **Parameter Counts**: Total, trainable, and non-trainable parameters
- **Model Size**: Estimated memory usage in MB
- **Input/Output Shapes**: Model input dimensions and output tensor shapes
- **Layer Details**: Per-layer parameter breakdown with names, types, and counts
- **Training Configuration**: Device, optimizer, scheduler, learning rate, etc.
- **Model Bundle Metadata**: Anchors, strides, input size, etc.

### 2. `model_parameters.json`
Detailed parameter breakdown including:
- **Module-wise Parameters**: Parameter count for each module/layer
- **Parameter Groups**: Statistics grouped by module type
- **Shape Information**: Detailed parameter tensor shapes and types
- **Trainable/Non-trainable Split**: Parameter counts by training status

### 3. `experiment_config.json`
Complete experiment configuration including:
- **Trainer Configuration**: Epochs, learning rate, optimizer settings
- **Runtime Information**: Device used, CUDA availability and device count
- **Scheduler Parameters**: Learning rate scheduler configuration
- **Assigner Parameters**: Target assignment configuration

### 4. `dataloader_config_snapshot.json`
Dataloader configuration snapshot including:
- **Batch Configuration**: Batch size, num_workers, pin_memory
- **Augmentation Settings**: Enabled augmentations and their parameters
- **Dataset Parameters**: Normalization, input preprocessing

### 5. `training_history.json`
Training progress history (generated during training):
- **Per-epoch Metrics**: Training and validation loss
- **Training Timeline**: Epoch-by-epoch progress tracking

## Example Usage

The metadata generation is automatic and requires no additional configuration. Simply run training:

```bash
python main.py --config config.yaml
```

## Example Model Summary Output

```json
{
  "model_architecture": "GAIYoloV12",
  "backbone_type": "TinyCSPBackbone",
  "total_parameters": 485340,
  "trainable_parameters": 485340,
  "parameter_size_mb": 1.85,
  "input_shape": [1, 3, 320, 320],
  "output_shapes": [[1, 2, 160, 160, 15], [1, 2, 80, 80, 15]],
  "device": "cuda",
  "optimizer": "adamw",
  "learning_rate": 0.001
}
```

## Benefits

1. **Experiment Tracking**: Complete record of model architecture and training configuration
2. **Reproducibility**: All settings saved for exact experiment reproduction
3. **Model Analysis**: Detailed parameter breakdown for optimization and analysis
4. **Debugging**: Runtime information for troubleshooting device and configuration issues
5. **Comparison**: Easy comparison between different model configurations and experiments

## Integration with Experiment Management

The metadata files integrate seamlessly with experiment management tools and can be used for:
- Automated experiment comparison
- Model selection and analysis
- Performance benchmarking
- Configuration optimization
- Research documentation

## File Locations

All metadata files are saved in the experiment artifact directory:
```
{experiment.artifact_dir}/metadata/
├── model_summary.json
├── model_parameters.json
├── experiment_config.json
├── dataloader_config_snapshot.json
└── training_history.json (generated during training)
```

This comprehensive metadata generation ensures that every experiment is fully documented and reproducible, supporting better model development and research workflows.