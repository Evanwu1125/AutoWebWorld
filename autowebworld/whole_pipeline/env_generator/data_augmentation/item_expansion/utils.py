"""
Utility functions for Item Expansion module.
"""
import json
import re
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file and return as dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_mockdata(path: Union[str, Path]) -> Dict[str, Any]:
    return load_json(path)


def get_mockdata_schema(mockdata: Dict[str, Any]) -> Dict[str, List[str]]:
    schema = {}
    for key, value in mockdata.items():
        if isinstance(value, list) and len(value) > 0:
            # Get field names from first item
            first_item = value[0]
            if isinstance(first_item, dict):
                schema[key] = list(first_item.keys())
    return schema


def extract_field_from_selector(selector: str) -> Optional[str]:
    patterns = [
        # #filter-beds-3plus-checkbox -> beds
        r'#filter-([a-z]+)-[a-z0-9]+-checkbox',
        # #filter-direct-checkbox -> direct
        r'#filter-([a-z]+)-checkbox',
        # #filter-price-slider -> price
        r'#filter-([a-z]+)-slider',
        # #sort-option-price -> price
        r'#sort-option-([a-z]+)',
        # #sort-dropdown-{field} -> field
        r'#sort-dropdown-([a-z]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, selector, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    return None


def extract_condition_from_selector(selector: str) -> Optional[Dict[str, Any]]:
    match = re.search(r'#filter-([a-z]+)-(\d+)plus-checkbox', selector, re.IGNORECASE)
    if match:
        return {
            "field": match.group(1).lower(),
            "op": ">=",
            "value": int(match.group(2))
        }
    
    match = re.search(r'#filter-([a-z]+)-max(\d+)-checkbox', selector, re.IGNORECASE)
    if match:
        return {
            "field": match.group(1).lower(),
            "op": "<=",
            "value": int(match.group(2))
        }
    
    match = re.search(r'#filter-([a-z]+)-checkbox', selector, re.IGNORECASE)
    if match:
        field = match.group(1).lower()
        return {
            "field": field,
            "op": "==",
            "value": True,
            "is_boolean": True
        }
    
    return None


def normalize_entity_type(key: str) -> str:
    if key.endswith('_id'):
        return key[:-3]
    return key


def pluralize_entity_type(entity_type: str) -> str:
    if entity_type.endswith('s'):
        return entity_type + 'es'
    elif entity_type.endswith('y'):
        return entity_type[:-1] + 'ies'
    else:
        return entity_type + 's'


def is_static_data_loading(workspace_path: Union[str, Path]) -> bool:
    """
    Check if workspace's data.js uses static data loading.

    Static loading patterns:
    - ref([{...}, {...}]) with hardcoded data
    - state: () => ({ items: [{...}] }) with hardcoded data

    Dynamic loading patterns:
    - ref([]) empty array
    - initializeMockData() / generateMockData() functions
    - fetch() / axios API calls

    Args:
        workspace_path: Path to workspace directory

    Returns:
        True: Static loading (usable for grounding)
        False: Dynamic loading (not usable)
    """
    data_js_path = Path(workspace_path) / "vue_template/src/stores/data.js"

    if not data_js_path.exists():
        return False

    content = data_js_path.read_text(encoding='utf-8')

    # Dynamic loading patterns
    dynamic_patterns = [
        r'ref\(\[\]\)',                    # ref([]) empty array
        r'initializeMockData',             # initialization function
        r'generateMockData',               # generation function
        r'fetch\(',                        # fetch API
        r'axios\.',                        # axios calls
        r'\.get\([\'"].*api',              # API calls like .get('/api/...')
    ]

    for pattern in dynamic_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return False

    # Static loading: ref([{...}]) with actual data
    if re.search(r'ref\(\[\s*\{', content):
        return True

    # Or state: () => ({ items: [{...}] })
    if re.search(r'state.*\(\)\s*=>\s*\(\{.*\[\s*\{', content, re.DOTALL):
        return True

    return False


def randomize_datepicker_times(trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    datepicker_info = []
    for idx, action in enumerate(trajectory):
        if action.get("name") == "select" and action.get("params", {}).get("widget") == "date_picker":
            action_id = action.get("id", "").upper()
            is_start = "START" in action_id
            is_end = "END" in action_id

            selector_num = _extract_selector_number(action)

            datepicker_info.append({
                "idx": idx,
                "is_start": is_start,
                "is_end": is_end,
                "selector_num": selector_num
            })

    if not datepicker_info:
        return trajectory

    datepicker_info.sort(key=lambda x: (
        0 if x["is_start"] else (1 if x["is_end"] else 2),
        x["selector_num"] if x["selector_num"] is not None else 999
    ))

    prev_year, prev_month = 2020, 1

    for info in datepicker_info:
        idx = info["idx"]

        year = random.randint(prev_year, 2030)
        if year == prev_year:
            if prev_month < 12:
                month = random.randint(prev_month + 1, 12)
            else:
                year = prev_year + 1
                month = random.randint(1, 12)
        else:
            month = random.randint(1, 12)

        day = random.randint(1, 28)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)

        if "params" in trajectory[idx]:
            trajectory[idx]["params"]["year"] = year
            trajectory[idx]["params"]["month"] = month
            trajectory[idx]["params"]["day"] = day
            if "hour" in trajectory[idx]["params"]:
                trajectory[idx]["params"]["hour"] = hour
            if "minute" in trajectory[idx]["params"]:
                trajectory[idx]["params"]["minute"] = minute

        gui_procedure = trajectory[idx].get("gui_procedure", [])
        for step in gui_procedure:
            if step.get("op") == "click" and "selector" in step:
                selector = step["selector"]
                selector = re.sub(r'\.year-\d+', f'.year-{year}', selector)
                selector = re.sub(r'\.month-\d+', f'.month-{month}', selector)
                selector = re.sub(r'\.day-\d+', f'.day-{day}', selector)
                selector = re.sub(r'\.hour-\d+', f'.hour-{hour}', selector)
                selector = re.sub(r'\.minute-\d+', f'.minute-{minute}', selector)
                step["selector"] = selector

        prev_year, prev_month = year, month

    return trajectory


def _extract_selector_number(action: Dict[str, Any]) -> Optional[int]:
    gui_procedure = action.get("gui_procedure", [])
    for step in gui_procedure:
        if step.get("op") == "click" and "selector" in step:
            selector = step["selector"]
            match = re.search(r'#date-picker(\d+)', selector)
            if match:
                return int(match.group(1))
    return None
