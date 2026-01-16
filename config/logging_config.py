"""
Logging configuration for RAG Brain
Configures file and console logging with rotation
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from config.settings import settings


def setup_logging(
    name: str = "rag_brain",
    log_level: str = None,
    log_file: str = None
) -> logging.Logger:
    """
    Set up logging with file and console handlers

    Args:
        name: Logger name
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file path (defaults to logs/rag_brain.log)

    Returns:
        Configured logger instance
    """
    log_level = log_level or settings.LOG_LEVEL
    log_file = log_file or str(settings.LOGS_DIR / "rag_brain.log")

    # Ensure logs directory exists
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler (simple format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler (detailed format with rotation)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Error file handler (only errors and above)
    error_file_handler = logging.handlers.RotatingFileHandler(
        str(settings.LOGS_DIR / "errors.log"),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_file_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name or "rag_brain")


# Initialize logging on import
logger = setup_logging()
