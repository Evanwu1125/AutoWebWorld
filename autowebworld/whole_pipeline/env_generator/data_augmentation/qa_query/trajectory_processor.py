"""Trajectory processing utilities"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class TrajectoryProcessor:
    
    @staticmethod
    def extract_item_id_from_dirname(dirname: str) -> Optional[str]:
        """Extract item_id from directory name: macro_xxx_item_th_3 -> th_3"""
        if '_item_' in dirname:
            return dirname.split('_item_')[-1]
        return None
    
    @staticmethod
    def is_trajectory_success(traj_dir: Path) -> bool:
        """Check if trajectory succeeded by checking if result.json exists"""
        return (traj_dir / 'result.json').exists()
    
    @staticmethod
    def find_detail_action(actions: List[Dict]) -> Tuple[Optional[int], Optional[Dict]]:
        """Find DETAIL action in trajectory
        
        Strategy:
        1. Self-loop in DETAIL state (from == to and 'DETAIL' in state)
           - User is performing action within DETAIL page
           - Use image_before (already on DETAIL page)
        2. Transition TO DETAIL state (LIST → DETAIL)
           - User is navigating to DETAIL page
           - Use image_after (arrived at DETAIL page)
        """
        for i, action in enumerate(actions):
            from_state = action.get('from', '')
            to_state = action.get('to', '')
            
            # Priority 1: Self-loop in DETAIL
            if from_state == to_state and 'DETAIL' in from_state:
                return i, action
            
            # Priority 2: Transition TO DETAIL
            if 'DETAIL' in to_state:
                return i, action
        
        return None, None
    
    @staticmethod
    def truncate_trajectory(actions: List[Dict], detail_action_idx: int) -> List[Dict]:
        """Truncate trajectory up to and including the detail action"""
        return actions[:detail_action_idx + 1]
    
    @staticmethod
    def get_target_screenshot(detail_action: Dict, traj_dir: Path) -> Optional[Path]:
        """Get the screenshot path for the detail action
        
        Logic:
        - If self-loop (from == to): use image_before (already on DETAIL page)
        - If transition TO DETAIL: use image_after (arrived at DETAIL page)
        """
        steps = detail_action.get('steps', [])
        if not steps:
            return None
        
        first_step = steps[0]
        grounding = first_step.get('grounding', {})
        
        from_state = detail_action.get('from', '')
        to_state = detail_action.get('to', '')
        
        # Self-loop: already on DETAIL page, use image_before
        if from_state == to_state and 'DETAIL' in from_state:
            image_path = grounding.get('image_before')
        # Transition TO DETAIL: just arrived at DETAIL page, use image_after
        else:
            image_path = grounding.get('image_after')
        
        if image_path:
            return traj_dir / image_path
        
        return None

