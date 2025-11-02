'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-10-30 17:25:00
 # @ Modified time: 2025-10-30 17:25:00
 # @ Description: Command-line entry point for configuration-driven conversion and training workflows.
'''
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from train import run_training
from utils.converter import run_conversion
from utils.utils import load_yaml_config


def _setup_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def start_tensorboard_server(host: str = "localhost", port: int = 6006, logdir: str = "./experiments") -> None:
    """Start TensorBoard server for experiment visualization."""
    logger = logging.getLogger(__name__)
    
    # Check if experiments directory exists
    experiments_path = Path(logdir)
    if not experiments_path.exists():
        logger.error("Experiments directory not found: %s", experiments_path.absolute())
        logger.info("Run some training first to generate experiment data")
        return
    
    # Check if there are any experiment directories
    experiment_dirs = [d for d in experiments_path.iterdir() if d.is_dir()]
    if not experiment_dirs:
        logger.error("No experiment directories found in: %s", experiments_path.absolute())
        logger.info("Run some training first to generate experiment data")
        return
    
    logger.info("Found %d experiment directories:", len(experiment_dirs))
    for exp_dir in experiment_dirs:
        logger.info("  - %s", exp_dir.name)
    
    # Start TensorBoard server
    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(experiments_path.absolute()),
        "--host", host,
        "--port", str(port),
        "--reload_interval", "30",
    ]
    
    logger.info("Starting TensorBoard server...")
    logger.info("Command: %s", " ".join(cmd))
    logger.info("TensorBoard will be available at: http://%s:%d", host, port)
    logger.info("Press Ctrl+C to stop the server")
    logger.info("")
    logger.info("📊 TensorBoard Features Available:")
    logger.info("  • SCALARS: Training/validation metrics and losses")
    logger.info("  • GRAPHS: Model architecture visualization")
    logger.info("  • IMAGES: Input data and model predictions (if logged)")
    logger.info("  • HISTOGRAMS: Weight and gradient distributions")
    logger.info("")
    
    time.sleep(1)  # Brief pause for user to read the info
    
    try:
        # Start TensorBoard process
        process = subprocess.run(cmd, check=False)
        if process.returncode != 0:
            logger.error("TensorBoard server exited with code: %d", process.returncode)
    except KeyboardInterrupt:
        logger.info("TensorBoard server stopped by user")
    except FileNotFoundError:
        logger.error("TensorBoard not found. Install it with: pip install tensorboard")
    except Exception as e:
        logger.error("Failed to start TensorBoard server: %s", e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GAI-YOLOv12 training entry point",
        epilog="""
Examples:
  python main.py                                    # Run training with default config
  python main.py --config my_config.yaml          # Run training with custom config
  python main.py --only-convert                    # Only convert dataset to COCO format
  python main.py --tensorboard                     # Start TensorBoard server
  python main.py --tensorboard --tensorboard-port 8080  # Start TensorBoard on port 8080
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--only-convert",
        action="store_true",
        help="Only run dataset conversion and exit",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Start TensorBoard server and exit",
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="Port for TensorBoard server (default: 6006)",
    )
    parser.add_argument(
        "--tensorboard-host",
        type=str,
        default="localhost",
        help="Host for TensorBoard server (default: localhost)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    
    # Handle TensorBoard server mode
    if args.tensorboard:
        start_tensorboard_server(
            host=args.tensorboard_host,
            port=args.tensorboard_port,
            logdir="./experiments"
        )
        return
    
    # Normal training/conversion workflow
    config_path = Path(args.config)
    logging.getLogger(__name__).info("Loading configuration from %s", config_path)
    config = load_yaml_config(config_path)

    dataset_cfg = config.get("dataset", {})
    if dataset_cfg.get("convert_to_coco", False):
        run_conversion(config)
    else:
        logging.getLogger(__name__).info("Dataset conversion disabled via configuration")

    if args.only_convert:
        logging.getLogger(__name__).info("Conversion finished; exiting due to --only-convert")
        return

    if args.resume:
        logging.getLogger(__name__).info("Starting training pipeline (resume mode)")
    else:
        logging.getLogger(__name__).info("Starting training pipeline")
    training_result = run_training(config, resume=args.resume)
    if not training_result.metrics:
        logging.getLogger(__name__).warning("Training did not run; see logs for details")
    else:
        last_epoch = training_result.metrics[-1]
        logging.getLogger(__name__).info(
            "Training complete | epochs=%d | final_train_loss=%.4f | final_val_loss=%s",
            last_epoch.epoch,
            last_epoch.train_loss,
            f"{last_epoch.val_loss:.4f}" if last_epoch.val_loss is not None else "n/a",
        )
        if training_result.summary_path is not None:
            logging.getLogger(__name__).info("Training history saved to %s", training_result.summary_path)


if __name__ == "__main__":
    main()
