"""
Position Calculator.

Calculates item positions based on trajectory type (sort, slider, filter).
"""
import re
from typing import Dict, Any, List, Optional, Tuple
from .trajectory_classifier import TrajectoryType
from .filter_parser import FilterCondition


class PositionCalculator:
    """
    Calculates item positions in list based on trajectory type.
    
    Position calculation varies by trajectory type:
        - CHECKBOX: Position in filtered list
        - SLIDER: Position after applying slider threshold
        - SORT: Position after sorting
        - SEARCH/SCROLL: Position not critical (uses item title)
    """
    
    def __init__(self):
        pass
    
    def calculate_position(
        self,
        item: Dict[str, Any],
        all_items: List[Dict[str, Any]],
        trajectory_type: TrajectoryType,
        filter_condition: Optional[FilterCondition] = None,
        sort_field: Optional[str] = None,
        sort_ascending: bool = True,
        id_field: str = "id"
    ) -> Dict[str, Any]:
        """
        Calculate item position based on trajectory type.
        
        Args:
            item: The target item
            all_items: All items in the list
            trajectory_type: Type of trajectory
            filter_condition: Filter condition (for CHECKBOX type)
            sort_field: Field to sort by (for SORT type)
            sort_ascending: Sort direction (for SORT type)
            id_field: Field name for item ID
            
        Returns:
            Dictionary with position info:
                - position: 1-indexed position in list
                - total: Total items in list
                - method: How position was calculated
                - slider_threshold: (for SLIDER) Threshold value used
        """
        item_id = item.get(id_field)
        
        if trajectory_type == TrajectoryType.CHECKBOX:
            return self._calculate_filtered_position(
                item, all_items, filter_condition, id_field
            )
        
        elif trajectory_type == TrajectoryType.SLIDER:
            return self._calculate_slider_position(
                item, all_items, filter_condition, id_field
            )
        
        elif trajectory_type == TrajectoryType.SORT:
            return self._calculate_sorted_position(
                item, all_items, sort_field, sort_ascending, id_field
            )
        
        else:
            # SEARCH, SCROLL, DIRECT - position in original list
            return self._calculate_original_position(item, all_items, id_field)
    
    def _calculate_filtered_position(
        self,
        item: Dict[str, Any],
        all_items: List[Dict[str, Any]],
        filter_condition: Optional[FilterCondition],
        id_field: str
    ) -> Dict[str, Any]:
        """Calculate position in filtered list."""
        from .item_filter import ItemFilter
        
        item_filter = ItemFilter()
        filtered_items = item_filter.filter_items(all_items, filter_condition)
        
        item_id = item.get(id_field)
        position = None
        for idx, filtered_item in enumerate(filtered_items):
            if filtered_item.get(id_field) == item_id:
                position = idx + 1
                break
        
        return {
            "position": position,
            "total": len(filtered_items),
            "method": "checkbox_filter",
            "filter_field": filter_condition.field if filter_condition else None
        }
    
    def _calculate_slider_position(
        self,
        item: Dict[str, Any],
        all_items: List[Dict[str, Any]],
        filter_condition: Optional[FilterCondition],
        id_field: str
    ) -> Dict[str, Any]:
        """
        Calculate position for slider filter.
        
        For slider, we dynamically set the threshold based on the item's value,
        then calculate the position in the filtered list.
        """
        if not filter_condition:
            return self._calculate_original_position(item, all_items, id_field)
        
        field = filter_condition.field
        item_value = item.get(field)
        
        if item_value is None:
            return self._calculate_original_position(item, all_items, id_field)
        
        # Calculate threshold: set slightly below item's value
        # This ensures the item is included in the filtered list
        if isinstance(item_value, (int, float)):
            # Set threshold to include this item and items with lower values
            threshold = item_value * 0.9 if item_value > 0 else item_value - 1
        else:
            threshold = item_value
        
        # Filter items that meet the threshold
        filtered_items = []
        for it in all_items:
            it_value = it.get(field)
            if it_value is not None and it_value >= threshold:
                filtered_items.append(it)
        
        # Sort by the field to get consistent ordering
        filtered_items.sort(key=lambda x: x.get(field, 0))
        
        item_id = item.get(id_field)
        position = None
        for idx, filtered_item in enumerate(filtered_items):
            if filtered_item.get(id_field) == item_id:
                position = idx + 1
                break
        
        return {
            "position": position,
            "total": len(filtered_items),
            "method": "slider_filter",
            "slider_field": field,
            "slider_threshold": threshold,
            "item_value": item_value
        }

    def _calculate_sorted_position(
        self,
        item: Dict[str, Any],
        all_items: List[Dict[str, Any]],
        sort_field: Optional[str],
        sort_ascending: bool,
        id_field: str
    ) -> Dict[str, Any]:
        """Calculate position after sorting."""
        if not sort_field:
            return self._calculate_original_position(item, all_items, id_field)

        # Sort items by the specified field
        sorted_items = sorted(
            all_items,
            key=lambda x: x.get(sort_field, 0) if x.get(sort_field) is not None else 0,
            reverse=not sort_ascending
        )

        item_id = item.get(id_field)
        position = None
        for idx, sorted_item in enumerate(sorted_items):
            if sorted_item.get(id_field) == item_id:
                position = idx + 1
                break

        return {
            "position": position,
            "total": len(sorted_items),
            "method": "sort",
            "sort_field": sort_field,
            "sort_ascending": sort_ascending
        }

    def _calculate_original_position(
        self,
        item: Dict[str, Any],
        all_items: List[Dict[str, Any]],
        id_field: str
    ) -> Dict[str, Any]:
        """Calculate position in original list order."""
        item_id = item.get(id_field)
        position = None
        for idx, it in enumerate(all_items):
            if it.get(id_field) == item_id:
                position = idx + 1
                break

        return {
            "position": position,
            "total": len(all_items),
            "method": "original"
        }

    def parse_sort_selector(self, selector: str) -> Tuple[Optional[str], bool]:

        match = re.search(r'#sort-option-(\w+)', selector, re.IGNORECASE)
        if match:
            field = match.group(1).lower()

            # Determine direction from field name
            ascending_keywords = ['oldest', 'cheapest', 'lowest', 'asc', 'az']
            descending_keywords = ['newest', 'expensive', 'highest', 'desc', 'za', 'price']

            is_ascending = True
            for kw in descending_keywords:
                if kw in field:
                    is_ascending = False
                    break

            # Clean field name (remove direction keywords)
            clean_field = field
            for kw in ascending_keywords + descending_keywords:
                clean_field = clean_field.replace(kw, '')
            clean_field = clean_field.strip('_-') or field

            return clean_field, is_ascending

        return None, True

