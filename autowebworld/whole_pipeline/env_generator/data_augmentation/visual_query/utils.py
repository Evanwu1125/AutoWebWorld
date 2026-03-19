"""
Utility functions for visual query generation.
"""

import json
from pathlib import Path
from typing import Dict, Any
import sys
import os

current_dir = os.path.dirname(__file__)
scripts_dir = os.path.join(current_dir, '..', 'scripts')
sys.path.insert(0, scripts_dir)

from convert_data_js import extract_js_arrays


def load_data(file_path: str) -> Dict[str, Any]:
    """
    Load data from file. Supports both .js and .json formats.

    Args:
        file_path: Path to the data file (.js or .json)

    Returns:
        Loaded data as dictionary
    """
    if file_path.endswith('.js'):
        with open(file_path, 'r', encoding='utf-8') as f:
            js_content = f.read()
        return extract_js_arrays(js_content)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def save_data(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

