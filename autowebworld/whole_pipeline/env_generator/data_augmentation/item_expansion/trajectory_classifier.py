from enum import Enum
from typing import Dict, Any, List, Optional


class TrajectoryType(Enum):
    """Types of trajectories based on item search/filter method."""
    CHECKBOX = "checkbox_filter"      # Boolean attribute filter
    SLIDER = "slider_filter"          # Threshold-based filter
    SORT = "sort"                     # Ordering by field
    SEARCH = "search"                 # Text search
    SCROLL = "scroll"                 # Scroll to find item
    # DIRECT = "direct"                 # Direct item selection (no filter/search)
    UNKNOWN = "unknown"


class TrajectoryClassifier:

    
    def __init__(self):
        pass
    
    def classify(self, trajectory: List[Dict[str, Any]]) -> TrajectoryType:

        for action in trajectory:
            action_type = self._classify_action(action)
            if action_type != TrajectoryType.UNKNOWN:
                return action_type
        
        return TrajectoryType.DIRECT
    
    def _classify_action(self, action: Dict[str, Any]) -> TrajectoryType:
        gui_procedure = action.get('gui_procedure', [])
        params = action.get('params', {})
        action_id = action.get('id', '')
        
        widget = params.get('widget', '')
        if widget == 'checkboxes':
            return TrajectoryType.CHECKBOX
        elif widget == 'sliders':
            return TrajectoryType.SLIDER
        elif widget == 'sort':
            return TrajectoryType.SORT
        
        if not gui_procedure:
            return TrajectoryType.UNKNOWN
        
        ops = [step.get('op', '') for step in gui_procedure]
        selectors = [step.get('selector', '') for step in gui_procedure]
        
        if len(ops) >= 2:
            if ops[0] == 'click' and ops[1] == 'drag':
                return TrajectoryType.SLIDER
            
            if ops[0] == 'click' and ops[1] == 'type_text':
                return TrajectoryType.SEARCH
            
        
        if len(ops) >= 1 and ops[0] == 'click':
            selector = selectors[0] if selectors else ''
            
            # Check selector for checkbox
            if 'checkbox' in selector.lower():
                return TrajectoryType.CHECKBOX
            
            # Check selector for sort
            if 'sort' in selector.lower():
                return TrajectoryType.SORT
        
        # Check for scroll operations
        if any('scroll' in op.lower() for op in ops):
            return TrajectoryType.SCROLL
        
        return TrajectoryType.UNKNOWN
    
    def get_filter_action(
        self, 
        trajectory: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:

        for action in trajectory:
            action_type = self._classify_action(action)
            if action_type in [
                TrajectoryType.CHECKBOX, 
                TrajectoryType.SLIDER, 
                TrajectoryType.SORT,
                TrajectoryType.SEARCH
            ]:
                return action
        return None
    
    def get_item_selection_action(
        self, 
        trajectory: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        for action in trajectory:
            params = action.get('params', {})
            for key, value in params.items():
                if isinstance(value, str) and '{ITEM_ANY}' in value:
                    return action
                if key.endswith('_id') and value == '{ITEM_ANY}':
                    return action
        return None

