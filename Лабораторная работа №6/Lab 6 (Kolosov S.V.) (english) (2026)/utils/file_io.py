# Author: Kolosov S.V., IVT-3, 4th year
# Lab work №6, variant №1, 2026
# Functions for loading and saving projects, importing/exporting data

import json
import os
from typing import Dict, Any, Optional
import logging

# Import logger
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
    Saves the project (application data) to a JSON file.
    """
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Project successfully saved: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving project to {filepath}: {e}")
        return False

def load_project(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Loads a project from a JSON file.
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Project file not found: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Project successfully loaded from: {filepath}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading project from {filepath}: {e}")
        return None

def export_data_to_json(data: Dict[str, Any], filepath: str) -> bool:
    """
    Exports arbitrary data to a JSON file.
    """
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"Data successfully exported to: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error exporting data to {filepath}: {e}")
        return False

def import_data_from_json(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Imports data from a JSON file.
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Data file not found: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Data successfully imported from: {filepath}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error importing data from {filepath}: {e}")
        return None

if __name__ == "__main__":
    print("--- Testing file_io.py ---")

    test_data = {
        "options": [
            {"name": "Option A", "outcomes": [100, -50], "probabilities": [0.6, 0.4]},
            {"name": "Option B", "outcomes": [80, 20], "probabilities": [0.5, 0.5]}
        ],
        "history": ["Project created", "Option A added"],
        "settings": {"dark_mode": True, "save_history": True}
    }

    saved = save_project(test_data, "test_project.json")
    if saved:
        print("Project saved to test_project.json")

    loaded_data = load_project("test_project.json")
    if loaded_data:
        print("Project loaded:")
        print(json.dumps(loaded_data, indent=2, ensure_ascii=False))

    export_data_to_json({"result": 42}, "exported_result.json")
    print("Data exported to exported_result.json")

    imported_data = import_data_from_json("exported_result.json")
    if imported_data:
        print("Data imported:")
        print(imported_data)

    print("file_io.py is working correctly.")
