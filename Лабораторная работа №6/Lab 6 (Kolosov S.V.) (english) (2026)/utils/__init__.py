# Author: Kolosov S.V., IVT-3, 4th year
# Lab work №6, variant №1, 2026
# Initialization of the utils package

__all__ = ["logger", "file_io"]

# Import main components for easy access
try:
    from .logger import logger, setup_logger, log_exception, safe_call
    from .file_io import save_project, load_project, export_data_to_json, import_data_from_json
except ImportError as e:
    # If import fails, create stubs
    import logging
    logger = logging.getLogger("Utils")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.warning(f"Failed to import utils modules: {e}")

    def setup_logger(*args, **kwargs):
        return logging.getLogger("Utils")

    def log_exception(func):
        return func

    def safe_call(func, *args, default_return=None, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return default_return

    def save_project(*args, **kwargs):
        return False

    def load_project(*args, **kwargs):
        return None

    def export_data_to_json(*args, **kwargs):
        return False

    def import_data_from_json(*args, **kwargs):
        return None

__version__ = "1.0.0"
__author__ = "Kolosov S.V., IVT-3, 4th year"
