from typing import Dict, Any, List, Optional, Callable
from .filter_parser import FilterCondition


class ItemFilter:
    def __init__(self):
        self._operators: Dict[str, Callable[[Any, Any], bool]] = {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">": lambda a, b: self._safe_compare(a, b, lambda x, y: x > y),
            "<": lambda a, b: self._safe_compare(a, b, lambda x, y: x < y),
            ">=": lambda a, b: self._safe_compare(a, b, lambda x, y: x >= y),
            "<=": lambda a, b: self._safe_compare(a, b, lambda x, y: x <= y),
        }
    
    def _safe_compare(
        self, 
        a: Any, 
        b: Any, 
        op: Callable[[Any, Any], bool]
    ) -> bool:
        try:
            # Try numeric comparison
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return op(a, b)
            # Try string to number conversion
            if isinstance(a, str) and isinstance(b, (int, float)):
                return op(float(a), b)
            if isinstance(a, (int, float)) and isinstance(b, str):
                return op(a, float(b))
            # Default to False for incompatible types
            return False
        except (ValueError, TypeError):
            return False
    
    def filter_items(
        self,
        items: List[Dict[str, Any]],
        condition: Optional[FilterCondition] = None
    ) -> List[Dict[str, Any]]:
        if condition is None:
            return items
        
        return [item for item in items if self._matches(item, condition)]
    
    def _matches(self, item: Dict[str, Any], condition: FilterCondition) -> bool:
        field = condition.field
        
        # Get item value, trying different field name variations
        value = self._get_field_value(item, field)
        
        if value is None and not condition.is_boolean:
            # Field not found - consider as not matching for non-boolean
            return False
        
        if value is None and condition.is_boolean:
            # For boolean fields, treat missing as False
            value = False
        
        # Get the comparison operator
        op_func = self._operators.get(condition.operator)
        if not op_func:
            return False
        
        return op_func(value, condition.value)
    
    def _get_field_value(self, item: Dict[str, Any], field: str) -> Any:
        # Exact match
        if field in item:
            return item[field]
        
        # Case-insensitive match
        field_lower = field.lower()
        for key, value in item.items():
            if key.lower() == field_lower:
                return value
        
        # Try with common prefixes for boolean fields
        bool_prefixes = ['is_', 'has_', 'can_', 'should_']
        for prefix in bool_prefixes:
            candidate = prefix + field
            if candidate in item:
                return item[candidate]
            # Also try without underscore
            candidate_no_underscore = prefix.replace('_', '') + field
            for key in item:
                if key.lower() == candidate_no_underscore.lower():
                    return item[key]
        
        # Try camelCase version
        camel = field[0].lower() + field[1:].title().replace('_', '')
        if camel in item:
            return item[camel]
        
        return None
    
    def get_filtered_with_positions(
        self,
        items: List[Dict[str, Any]],
        condition: Optional[FilterCondition] = None
    ) -> List[tuple]:
        filtered = self.filter_items(items, condition)
        return [(item, idx + 1) for idx, item in enumerate(filtered)]

