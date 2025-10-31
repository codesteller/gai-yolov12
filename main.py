'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-10-30 17:25:00
 # @ Modified time: 2025-10-30 17:25:00
 # @ Description: Command-line entry point for configuration-driven conversion and training workflows.
'''
import argparse
import logging
from pathlib import Path

from utils.converter import run_conversion
from utils.utils import load_yaml_config


def _setup_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAI-YOLOv12 training entry point")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
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

    # Placeholder for upcoming training workflow.
    logging.getLogger(__name__).info("Training pipeline not yet implemented. Coming soon.")


if __name__ == "__main__":
    main()
