# Resume Training Feature

The GAI-YOLOv12 training pipeline now supports resuming interrupted training sessions from the latest checkpoint.

## Features

- **Automatic checkpoint discovery**: Finds the latest checkpoint in the experiment directory
- **Complete state restoration**: Restores model, optimizer, and scheduler states
- **Metrics continuation**: Loads previous training metrics and continues from the last epoch
- **Seamless integration**: Works with all existing training features (TensorBoard, COCO evaluation, etc.)

## Usage

### Basic Resume

To resume training from the latest checkpoint:

```bash
uv run main.py --resume
```

This will:
1. Look for checkpoints in the experiment's checkpoint directory
2. Load the latest checkpoint (by epoch number)
3. Restore model weights, optimizer state, and scheduler state
4. Continue training from the next epoch

### Resume with Custom Config

```bash
uv run main.py --config configs/config.yaml --resume
```

### First Time Training (No Resume)

```bash
uv run main.py --config config.yaml
```

## How It Works

### Checkpoint Structure

Checkpoints are saved with the following structure:
```python
{
    "model_state_dict": <model weights>,
    "optimizer_state_dict": <optimizer state>,
    "scheduler_state_dict": <scheduler state>,  # if scheduler exists
    "epoch": <epoch number>,
    "train_loss": <training loss>,
    "val_loss": <validation loss>
}
```

### Checkpoint Naming

Checkpoints are named as `epoch_XXX.pt` (e.g., `epoch_001.pt`, `epoch_010.pt`)

### Metrics Persistence

Training metrics are automatically saved to `training_metrics.json` in the checkpoint directory whenever a checkpoint is created. This allows metrics to be restored on resume.

## Example Workflow

1. **Start training**:
   ```bash
   uv run main.py --config configs/config.yaml
   ```

2. **Training interrupted** (e.g., Ctrl+C, system crash, etc.)

3. **Resume training**:
   ```bash
   uv run main.py --config configs/config.yaml --resume
   ```

The training will continue from where it left off, maintaining all training state.

## Configuration

Checkpoint frequency is controlled by the `checkpoint_interval` setting in your config:

```yaml
training:
  checkpoint_interval: 5  # Save checkpoint every 5 epochs
```

## Notes

- If no checkpoints are found when using `--resume`, training starts from scratch (epoch 1)
- The resume feature automatically finds the experiment directory based on the config
- All checkpoint files and metrics are preserved across resume operations
- TensorBoard logs continue seamlessly - you can see the full training history
- COCO evaluation metrics are also preserved and continued

## Testing

A test configuration is provided for quick testing:

```bash
# Run short training
uv run main.py --config configs/config_test_resume.yaml

# Then test resume
uv run main.py --config configs/config_test_resume.yaml --resume
```

The test config uses only 2 epochs and limited batches for quick validation.
