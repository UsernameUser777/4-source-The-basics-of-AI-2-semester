# utils/__init__.py
# Автор: Колосов С.В., ИВТ-3, 4 курс
# Лабораторная работа №6, вариант №1, 2026 г.
# Инициализация пакета utils

__all__ = ["logger", "file_io"]

# Импортируем основные компоненты для удобного доступа
try:
    from .logger import logger, setup_logger, log_exception, safe_call
    from .file_io import save_project, load_project, export_data_to_json, import_data_from_json
except ImportError as e:
    # Если импорт не удался, создаём заглушки
    import logging
    logger = logging.getLogger("Utils")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.warning(f"Не удалось импортировать модули utils: {e}")

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
__author__ = "Колосов С.В., ИВТ-3, 4 курс"
