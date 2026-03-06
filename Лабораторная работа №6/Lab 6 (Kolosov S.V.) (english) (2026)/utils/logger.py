# Author: Kolosov S.V., IVT-3, 4th year
# Lab work №6, variant №1, 2026
# Centralized logging for the risk analysis application

import logging
from logging.handlers import RotatingFileHandler
import functools
import traceback
from typing import Callable, Any
import os

# Logger name for the entire application
LOGGER_NAME = "RiskAnalysisApp"
LOG_FILE_PATH = "logs/app.log"  # Place in the logs/ subdirectory

# Create directory if it does not exist
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

def setup_logger(
    name: str = LOGGER_NAME,
    log_file: str = LOG_FILE_PATH,
    level: int = logging.DEBUG
) -> logging.Logger:
    """
    Setup and return a logger with file and console handlers.
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate logs
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Create a global logger instance
logger = setup_logger()

def log_exception(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for automatically logging exceptions in functions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error in function '{func.__name__}': {e}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            raise
    return wrapper

def safe_call(
    func: Callable[..., Any],
    *args,
    default_return=None,
    **kwargs
):
    """
    Safely call a function and return a default value if an exception occurs.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Safe call of '{func.__name__}' failed: {e}")
        return default_return

if __name__ == "__main__":
    logger.debug("This is a DEBUG level message")
    logger.info("This is an INFO level message")
    logger.warning("This is a WARNING level message")
    logger.error("This is an ERROR level message")

    @log_exception
    def example_func_that_fails():
        return 1 / 0

    result = safe_call(example_func_that_fails, default_return="Execution error")
    print(result)

    print(f"Logger '{LOGGER_NAME}' is configured. Check the file {LOG_FILE_PATH}.")
