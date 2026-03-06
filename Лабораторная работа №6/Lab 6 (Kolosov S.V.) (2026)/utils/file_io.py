# utils/file_io.py
# Автор: Колосов С.В., ИВТ-3, 4 курс
# Лабораторная работа №6, вариант №1, 2026 г.
# Функции для загрузки и сохранения проектов, импорта/экспорта данных

import json
import os
from typing import Dict, Any, Optional
import logging

# Импортируем логгер
try:
    from utils.logger import logger
except ImportError:
    logger = logging.getLogger("FileIO")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def save_project(data: Dict[str, Any], filepath: str) -> bool:
    """
    Сохраняет проект (данные приложения) в JSON-файл.
    """
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"Проект успешно сохранён: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения проекта в {filepath}: {e}")
        return False


def load_project(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Загружает проект из JSON-файла.
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Файл проекта не найден: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Проект успешно загружен из: {filepath}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON в {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Ошибка загрузки проекта из {filepath}: {e}")
        return None


def export_data_to_json(data: Dict[str, Any], filepath: str) -> bool:
    """
    Экспортирует произвольные данные в JSON-файл.
    """
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"Данные успешно экспортированы в: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Ошибка экспорта данных в {filepath}: {e}")
        return False


def import_data_from_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Импортирует данные из JSON-файла.
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Файл данных не найден: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Данные успешно импортированы из: {filepath}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON в {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Ошибка импорта данных из {filepath}: {e}")
        return None


if __name__ == "__main__":
    print("--- Тест file_io.py ---")

    test_data = {
        "options": [
            {"name": "Option A", "outcomes": [100, -50], "probabilities": [0.6, 0.4]},
            {"name": "Option B", "outcomes": [80, 20], "probabilities": [0.5, 0.5]}
        ],
        "history": ["Создан проект", "Добавлена опция A"],
        "settings": {"dark_mode": True, "save_history": True}
    }

    saved = save_project(test_data, "test_project.json")
    if saved:
        print("Проект сохранён в test_project.json")

    loaded_data = load_project("test_project.json")
    if loaded_data:
        print("Проект загружен:")
        print(json.dumps(loaded_data, indent=2, ensure_ascii=False))

    export_data_to_json({"result": 42}, "exported_result.json")
    print("Данные экспортированы в exported_result.json")

    imported_data = import_data_from_json("exported_result.json")
    if imported_data:
        print("Данные импортированы:")
        print(imported_data)

    print("Файл file_io.py работает корректно.")
