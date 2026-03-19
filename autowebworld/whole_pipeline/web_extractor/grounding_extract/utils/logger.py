import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

class Logger:
    """Simple logger with configurable verbosity and file output."""

    def __init__(
        self,
        name: str = "grounding",
        verbose: bool = True,
        log_file: Optional[Path] = None,
        console_output: bool = True
    ):
        """
        Initialize logger.

        Args:
            name: Logger name
            verbose: Enable verbose logging
            log_file: Path to log file (if None, no file logging)
            console_output: Whether to output to console (default: True)
        """
        # Create unique logger name to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_name = f"{name}_{timestamp}"
        self.logger = logging.getLogger(unique_name)
        self.verbose = verbose
        self.log_file = log_file

        # Clear any existing handlers
        self.logger.handlers.clear()
        # Always set to INFO level, control verbosity through methods
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(message)s')

        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Add file handler if log_file is provided
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str):
        if self.verbose:
            self.logger.info(msg)

    def always_info(self, msg: str):
        """Always log info message, regardless of verbose setting."""
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        if self.verbose:
            self.logger.debug(msg)

