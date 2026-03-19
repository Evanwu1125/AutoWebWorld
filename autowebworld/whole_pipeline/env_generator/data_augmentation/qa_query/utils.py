"""Utility functions for QA Query Generation"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_json(file_path: Path) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_item_by_id(items: List[Dict], item_id: str) -> Optional[Dict]:
    for item in items:
        if item.get('id') == item_id:
            return item
    return None


def build_item_to_collection_map(caption_data: Dict) -> Dict[str, str]:
    item_to_collection = {}
    for collection_name, items in caption_data.items():
        if isinstance(items, list):
            for item in items:
                if 'id' in item:
                    item_to_collection[item['id']] = collection_name
    return item_to_collection


def count_by_type(qa_pairs: List[Dict]) -> Dict[str, int]:
    counts = {}
    for qa in qa_pairs:
        qa_type = qa.get('type', 'unknown')
        counts[qa_type] = counts.get(qa_type, 0) + 1
    return counts


def calculate_cost(input_tokens: int, output_tokens: int, model: str, pricing: Dict) -> Dict[str, float]:
    if model not in pricing:
        return {
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0
        }
    
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output"]
    
    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6)
    }

