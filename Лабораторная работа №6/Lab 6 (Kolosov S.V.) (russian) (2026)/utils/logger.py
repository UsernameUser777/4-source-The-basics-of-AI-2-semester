# utils/logger.py
# Автор: Колосов С.В., ИВТ-3, 4 курс
# Лабораторная работа №6, вариант №1, 2026 г.
# Централизованное логирование для приложения анализа рисков

import logging
from logging.handlers import RotatingFileHandler
import functools
import traceback
from typing import Callable, Any
import os

# Имя логгера для всего приложения
LOGGER_NAME = "RiskAnalysisApp"
LOG_FILE_PATH = "logs/app.log"  # Поместим в подкаталог logs/

# Создаём каталог, если не существует
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)


def setup_logger(
    name: str = LOGGER_NAME,
    log_file: str = LOG_FILE_PATH,
    level: int = logging.DEBUG
) -> logging.Logger:
    """
    Настройка и возврат логгера с обработчиками файла и консоли.
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Обработчик для файла с ротацией
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Избегаем дублирования логов
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# Создаём глобальный экземпляр логгера
logger = setup_logger()


def log_exception(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Декоратор для автоматического логирования исключений в функциях.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Ошибка в функции '{func.__name__}': {e}\n"
                f"Трассировка: {traceback.format_exc()}"
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
    Безопасный вызов функции. В случае ошибки — лог и возврат default_return.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Безопасный вызов '{func.__name__}' не удался: {e}")
        return default_return


if __name__ == "__main__":
    logger.debug("Это сообщение уровня DEBUG")
    logger.info("Это сообщение уровня INFO")
    logger.warning("Это сообщение уровня WARNING")
    logger.error("Это сообщение уровня ERROR")

    @log_exception
    def example_func_that_fails():
        return 1 / 0

    result = safe_call(example_func_that_fails, default_return="Ошибка выполнения")
    print(result)

    print(f"Логгер '{LOGGER_NAME}' настроен. Проверьте файл {LOG_FILE_PATH}.")
