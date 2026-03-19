"""
Item Expander - Main Module.

Expands BFS trajectories to generate item-specific trajectories for all items.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

from .utils import load_json, save_json, get_mockdata_schema, randomize_datepicker_times
from .entity_detector import EntityDetector
from .failure_logger import FailureLogger


class ItemExpander:
    """
    Main class for expanding BFS trajectories.

    Takes a BFS trajectory with {ITEM_ANY} placeholder and generates
    N item-specific trajectories, one for each item in mockdata.
    """

    def __init__(
        self,
        mockdata: Dict[str, Any],
        model: str = "deepseek-v3.2-exp",
        failure_logger: FailureLogger = None
    ):
        """
        Initialize the ItemExpander.

        Args:
            mockdata: Dictionary containing entity lists
            model: LLM model to use
        """
        self.mockdata = mockdata
        self.schema = get_mockdata_schema(mockdata)
        self.failure_logger = failure_logger or FailureLogger()

        self.entity_detector = EntityDetector(mockdata, model=model)

    async def expand(self, trajectory: List[Dict[str, Any]], filename: str = None, has_item_any: bool = None) -> List[Dict[str, Any]]:
        # Check if trajectory contains {ITEM_ANY} placeholder
        if not self._contains_item_placeholder(trajectory):
            # No item placeholder, return original trajectory as-is
            return [{
                "mockdata_key": None,
                "item": None,
                "item_id": "original",
                "trajectory_type": "NO_ITEM",
                "filter_field": None,
                "trajectory": trajectory
            }]

        # Use LLM to detect trajectory_type, mockdata_key, filter_field
        detection = await self.entity_detector.detect(trajectory)

        trajectory_type = detection.get('trajectory_type')
        mockdata_key = detection.get('mockdata_key')
        filter_field = detection.get('filter_field')
        items = detection.get('items', [])

        if not items:
            if filename and has_item_any:
                llm_raw_response = detection.get('llm_raw_response', '')
                self.failure_logger.add_failure(
                    trajectory_file=filename,
                    has_item_any=True,
                    llm_detection={
                        'trajectory_type': trajectory_type,
                        'mockdata_key': mockdata_key,
                        'filter_field': filter_field
                    },
                    llm_raw_response=llm_raw_response,
                    failure_point='items_empty'
                )
            print(f"Warning: No items found for trajectory. Reason: {detection.get('reason')}")
            return []

        # Get ID field
        id_field = self.entity_detector.get_item_id_field(items)

        # Filter items based on trajectory type
        filtered_items = self._filter_items(items, trajectory_type, filter_field)

        if not filtered_items:
            if filename and has_item_any:
                llm_raw_response = detection.get('llm_raw_response', '')
                self.failure_logger.add_failure(
                    trajectory_file=filename,
                    has_item_any=True,
                    llm_detection={
                        'trajectory_type': trajectory_type,
                        'mockdata_key': mockdata_key,
                        'filter_field': filter_field
                    },
                    llm_raw_response=llm_raw_response,
                    failure_point='filter_empty'
                )
            print(f"Warning: No items passed filter for {trajectory_type} with field {filter_field}")
            return []

        # Generate expanded trajectories (one per item)
        expanded = []
        for item in filtered_items:
            item_id = item.get(id_field)

            # Calculate target_value (only for SLIDER type)
            target_value = None
            if trajectory_type == 'SLIDER' and filter_field:
                target_value = self._calculate_target_value(item, filter_field)

            # Extract sort_order (only for SORT type)
            sort_order = None
            if trajectory_type == 'SORT':
                sort_order = self._extract_sort_order(trajectory)

            # Calculate rank (all types)
            rank = self._calculate_rank(
                item=item,
                all_items=items,
                trajectory_type=trajectory_type,
                filter_field=filter_field,
                target_value=target_value,
                sort_order=sort_order
            )

            # Fill only {ITEM_ANY} placeholder, not input_text yet
            filled_trajectory = self._fill_item_placeholder(
                trajectory=trajectory,
                item=item,
                id_field=id_field
            )

            expanded_entry = {
                "mockdata_key": mockdata_key,
                "item": item,
                "item_id": item_id,
                "trajectory_type": trajectory_type,
                "filter_field": filter_field,
                "rank": rank,
                "target_value": target_value,
                "sort_order": sort_order,
                "trajectory": filled_trajectory
            }
            expanded.append(expanded_entry)

        return expanded

    def _calculate_target_value(
        self,
        item: Dict[str, Any],
        filter_field: str
    ) -> Optional[int]:
        """
        Calculate target_value for SLIDER type using round numbers.

        Args:
            item: Current item
            filter_field: Field to filter by

        Returns:
            target_value (int): Round number less than item's field value
        """
        item_value = item.get(filter_field)
        if item_value is None:
            return None

        # If item_value > 20000, use half value
        if item_value > 20000:
            return max(1, int(item_value / 2))

        # Round numbers list
        round_numbers = [1, 2, 3, 5, 8, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

        valid_values = [v for v in round_numbers if v < item_value]

        if valid_values:
            # Choose the maximum (keep filtering strict)
            return max(valid_values)
        else:
            # If no suitable round number, use half of item_value
            return max(1, int(item_value / 2))

    def _extract_sort_order(
        self,
        trajectory: List[Dict[str, Any]]
    ) -> str:
        """
        Extract sort_order from trajectory.

        Args:
            trajectory: List of actions

        Returns:
            sort_order: "inc" or "desc" (default: "desc")
        """
        # Look for SORT action in trajectory
        for action in trajectory:
            action_id = action.get('id', '')  # Fixed: use 'id' instead of 'action_id'

            if 'SORT' in action_id:
                # Extract selector from gui_procedure
                gui_procedure = action.get('gui_procedure', [])
                for step in gui_procedure:
                    selector = step.get('selector', '')
                    if selector and 'inc' in selector.lower():
                        return 'inc'
                # If found SORT action but no 'inc' in selector, return 'desc'
                return 'desc'

        # Default to desc
        return 'desc'

    def _calculate_rank(
        self,
        item: Dict[str, Any],
        all_items: List[Dict[str, Any]],
        trajectory_type: str,
        filter_field: Optional[str],
        target_value: Optional[int],
        sort_order: Optional[str]
    ) -> int:
        """
        Calculate rank for all trajectory types.

        Args:
            item: Current item
            all_items: All items in mockdata
            trajectory_type: CHECKBOX/SLIDER/SORT/SEARCH/SCROLL
            filter_field: Filter field name
            target_value: Slider threshold (SLIDER only)
            sort_order: Sort order (SORT only)

        Returns:
            rank (int): Position of current item (1-indexed)
        """
        if not item or not all_items:
            return 1

        item_id = item.get('id')

        if trajectory_type == 'CHECKBOX':
            # Filter items where filter_field=True (boolean) or contains filter_field string
            filtered = [it for it in all_items if it.get(filter_field) is True]
            if not filtered:
                # Fallback to string matching
                filtered = [it for it in all_items if self._contains_filter_value(it, filter_field)]

        elif trajectory_type == 'SLIDER':
            # Filter items where field_value >= target_value
            if target_value is None:
                filtered = all_items
            else:
                filtered = [it for it in all_items if it.get(filter_field, 0) >= target_value]

        elif trajectory_type == 'SORT':
            # Sort items by filter_field, then by id (lexicographic) for tie-breaking
            reverse = (sort_order == 'desc')
            filtered = sorted(
                all_items,
                key=lambda x: (x.get(filter_field, 0), x.get('id', '')),
                reverse=reverse
            )

        else:  # SEARCH, SCROLL, or others
            # Use original order
            filtered = all_items

        # Find current item's position in filtered list
        for i, it in enumerate(filtered):
            if it.get('id') == item_id:
                return i + 1  # 1-indexed

        # Fallback if not found
        return 1

    def _filter_items(
        self,
        items: List[Dict[str, Any]],
        trajectory_type: str,
        filter_field: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter items based on trajectory type.

        - CHECKBOX:
          * First try: items where filter_field=True (boolean field)
          * Fallback: items where filter_field string appears in any field value
        - SLIDER/SORT: remove items with duplicate filter_field values
        - SEARCH/SCROLL: all items
        """
        if trajectory_type == 'CHECKBOX' and filter_field:
            # Try boolean field first
            filtered = [item for item in items if item.get(filter_field) is True]

            if filtered:
                # Found boolean field matches
                print(f"[CHECKBOX] Filtered {len(filtered)}/{len(items)} items "
                      f"where {filter_field}=True (boolean field)")
                return filtered

            # Fallback: check if filter_field string appears in any field value
            filtered = []
            for item in items:
                if self._contains_filter_value(item, filter_field):
                    filtered.append(item)

            print(f"[CHECKBOX] Filtered {len(filtered)}/{len(items)} items "
                  f"where '{filter_field}' appears in any field (string match)")
            return filtered

        # For SORT and SLIDER, remove items with duplicate filter_field values
        if trajectory_type in ['SORT', 'SLIDER'] and filter_field:
            return self._remove_duplicate_field_items(items, filter_field, trajectory_type)

        # For other types, keep all items
        return items

    def _remove_duplicate_field_items(
        self,
        items: List[Dict[str, Any]],
        filter_field: str,
        trajectory_type: str
    ) -> List[Dict[str, Any]]:
        """
        Remove items that have duplicate values in filter_field.

        For SORT and SLIDER types, we need unique filter_field values
        to ensure correct rank calculation.

        Args:
            items: All items
            filter_field: Field name to check for duplicates
            trajectory_type: SORT or SLIDER

        Returns:
            List of items with unique filter_field values
        """
        # Count occurrences of each filter_field value
        field_values = []
        for item in items:
            value = item.get(filter_field)
            if value is not None:
                field_values.append(value)

        value_counts = Counter(field_values)

        # Find duplicate values
        duplicate_values = {v for v, count in value_counts.items() if count > 1}

        if duplicate_values:
            print(f"[{trajectory_type}] Found {len(duplicate_values)} duplicate values "
                  f"in field '{filter_field}': {duplicate_values}")

        # Filter out items with duplicate values
        filtered = []
        skipped_count = 0
        for item in items:
            value = item.get(filter_field)
            if value in duplicate_values:
                skipped_count += 1
                continue  # Skip this item
            filtered.append(item)

        if skipped_count > 0:
            print(f"[{trajectory_type}] Skipped {skipped_count}/{len(items)} items "
                  f"with duplicate '{filter_field}' values")
            print(f"[{trajectory_type}] Kept {len(filtered)}/{len(items)} items "
                  f"with unique '{filter_field}' values")

        return filtered

    def _contains_filter_value(self, item: Dict[str, Any], filter_value: str) -> bool:
        """
        Check if filter_value appears in any field of the item.

        Args:
            item: Item to check
            filter_value: String to search for (case-insensitive)

        Returns:
            True if filter_value appears in any field value
        """
        filter_lower = filter_value.lower()

        for field_name, field_value in item.items():
            # Skip non-string fields
            if isinstance(field_value, str):
                # Case-insensitive substring match
                if filter_lower in field_value.lower():
                    return True
            elif isinstance(field_value, bool):
                # Skip boolean fields (already handled above)
                continue
            elif isinstance(field_value, (int, float)):
                # Skip numeric fields
                continue
            elif isinstance(field_value, list):
                # Check list items
                for list_item in field_value:
                    if isinstance(list_item, str) and filter_lower in list_item.lower():
                        return True

        return False

    def _contains_item_placeholder(self, trajectory: List[Dict[str, Any]]) -> bool:
        """Check if trajectory contains {ITEM_ANY} placeholder."""
        trajectory_str = json.dumps(trajectory)
        return '{ITEM_ANY}' in trajectory_str

    def _get_item_name(self, item: Dict[str, Any]) -> str:
        """Get item name from the second key (after 'id') or fallback to common fields."""
        if not item:
            return ''

        # Get all keys in order (Python 3.7+ dicts maintain insertion order)
        keys = list(item.keys())

        # Try to use the second key (assuming first is 'id' or similar)
        if len(keys) >= 2:
            second_key = keys[1]
            second_value = item.get(second_key)

            # Only use if it's a non-empty string
            if isinstance(second_value, str) and second_value.strip():
                return second_value

        # Fallback to hardcoded list (extended with common fields)
        for field in ['name', 'title', 'type', 'operator', 'driver', 'destination',
                      'subject', 'description', 'title_id', 'airline', 'model']:
            if field in item and item[field]:
                return str(item[field])

        return ''

    def _has_item_id_param(self, params: dict) -> bool:
        if not params:
            return False
        return any(key.endswith('_id') for key in params.keys())

    def _fill_item_placeholder(
        self,
        trajectory: List[Dict[str, Any]],
        item: Dict[str, Any],
        id_field: str
    ) -> List[Dict[str, Any]]:
        """
        Fill {ITEM_ANY} placeholder and SEARCH type_text in trajectory.

        Note: Complex {input_text} are NOT filled here.
        They will be filled later by query_generator.
        """
        item_id = str(item.get(id_field, ''))
        item_name = self._get_item_name(item)

        # Process SEARCH operations: replace type_text with item name
        for action in trajectory:
            action_id = action.get('id', '')
            params = action.get('params', {})

            # If this is a SEARCH operation, replace type_text with item name
            if self._has_item_id_param(params) and 'SEARCH' in action_id:
                if item_name:
                    gui_procedure = action.get('gui_procedure', [])
                    for step in gui_procedure:
                        if step.get('op') == 'type_text':
                            step['text'] = item_name
                else:
                    # Log warning: item has no 'name' field
                    print(f"Warning: Item {item_id} has no 'name' field for SEARCH operation")

        # Convert to string for replacement
        trajectory_str = json.dumps(trajectory, ensure_ascii=False)

        # Replace {ITEM_ANY} with item id
        trajectory_str = trajectory_str.replace('{ITEM_ANY}', item_id)

        return json.loads(trajectory_str)

    async def expand_file(
        self,
        input_path: Path,
        output_dir: Path,
        trajectory_data: dict = None
    ) -> List[Path]:
        """
        Expand a trajectory file and save results.

        Args:
            input_path: Path to BFS trajectory JSON file
            output_dir: Directory to save expanded trajectories
            trajectory_data: Optional trajectory data for logging

        Returns:
            List of paths to generated files
        """
        trajectory = load_json(input_path)

        # Handle both list format and dict format
        if isinstance(trajectory, dict):
            trajectory = trajectory.get('actions', trajectory.get('trajectory', [trajectory]))

        has_item_any = '{ITEM_ANY}' in json.dumps(trajectory)
        expanded = await self.expand(trajectory, input_path.name, has_item_any)

        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []

        base_name = input_path.stem
        for idx, entry in enumerate(expanded):
            item_id = entry.get('item_id', idx)

            # Randomize date_picker times before saving
            entry['trajectory'] = randomize_datepicker_times(entry['trajectory'])

            output_path = output_dir / f"{base_name}_item_{item_id}.json"
            save_json(entry, output_path)
            generated_files.append(output_path)

        print(f"Generated {len(generated_files)} item-specific trajectories from {input_path.name}")
        return generated_files

    def get_entity_detector_stats(self) -> Dict[str, Any]:
        return self.entity_detector.get_usage_stats()
