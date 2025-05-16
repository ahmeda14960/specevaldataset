"""
Logging utilities for SpecEval framework.

This module provides utilities for setting up logging across the SpecEval framework,
with support for both console and file-based logging.
"""

import logging
import datetime
from pathlib import Path


def setup_logging(verbose: bool = False, folder_name: str = "logs") -> logging.Logger:
    """
    Configure and return a logger with console and optional file handlers.

    This function creates a logger with appropriate formatting and handlers.
    When verbose mode is enabled, the logging level is set to DEBUG (instead of INFO)
    and logs are also written to a timestamped file in the specified folder.

    Args:
        verbose: Whether to enable DEBUG level logging (otherwise INFO) and file output
        folder_name: Name of the folder to store log files within data/logs/
                    (default: "logs")

    Returns:
        Configured logger
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler if verbose is enabled
    if verbose:
        # Create logs directory if it doesn't exist
        logs_dir = Path(f"data/logs/{folder_name}")
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create a timestamped log filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"evaluation_{timestamp}.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to {log_file}")

    return logger
