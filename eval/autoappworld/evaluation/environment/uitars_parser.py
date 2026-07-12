"""
UI-TARS action parser for converting UI-TARS format actions to AutoAppWorld format.
"""

from __future__ import annotations

import re
import ast
from typing import Any, Dict, Optional, Tuple


def normalize_uitars_action(action_text: str) -> str:
    """
    Extract action from UI-TARS format.
    
    UI-TARS format:
    Thought: ...
    Action: click(point='<point>x y</point>')
    
    Args:
        action_text: Full response text from UI-TARS model
        
    Returns:
        Extracted action text
    """
    text = action_text.strip()
    
    # Extract from "Action:" line
    match = re.search(r"Action:\s*(.+?)(?:\n|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return text


def convert_point_to_coordinates(text: str) -> str:
    """
    Convert <point>x y</point> format to (x, y) format.
    
    Args:
        text: Text containing <point> tags
        
    Returns:
        Text with coordinates converted to (x, y) format
    """
    pattern = r"<point>(\d+)\s+(\d+)</point>"
    
    def replace_match(match):
        x, y = map(int, match.groups())
        return f"({x},{y})"
    
    return re.sub(pattern, replace_match, text)


def parse_uitars_action(action_text: str, screen_width: int = 1280, screen_height: int = 720) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Parse UI-TARS action format to AutoAppWorld format.

    UI-TARS uses function call format with <point> tags:
    - click(point='<point>x y</point>')
    - left_double(point='<point>x y</point>')
    - right_single(point='<point>x y</point>')
    - drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
    - hotkey(key='ctrl c')
    - type(content='xxx')
    - scroll(point='<point>x y</point>', direction='down or up or right or left')
    - wait()
    - finished(content='xxx')

    Args:
        action_text: Action text (e.g., "click(point='<point>100 200</point>')")
        screen_width: Screen width (not used, UI-TARS uses absolute coordinates)
        screen_height: Screen height (not used, UI-TARS uses absolute coordinates)
        
    Returns:
        Tuple of (action_kind, action_params, finish_message)
        - action_kind: "action", "wait", or "finish"
        - action_params: Dict with action details
        - finish_message: Optional finish message
    """
    text = action_text.strip()
    
    # Convert <point> tags to coordinates
    text = convert_point_to_coordinates(text)
    
    # Normalize parameter names
    text = text.replace("start_point=", "start_box=")
    text = text.replace("end_point=", "end_box=")
    text = text.replace("point=", "start_box=")
    
    # Parse the action using AST
    try:
        node = ast.parse(text, mode='eval')
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")
        
        call = node.body
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")
        
        # Get function name
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            raise ValueError("Invalid function name")
        
        # Get keyword arguments
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # Python 3.7 compatibility
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value
            
    except Exception as e:
        raise ValueError(f"Failed to parse UI-TARS action '{action_text}': {e}")
    
    # Parse coordinates from string format "(x,y)"
    def parse_coord(coord_str: str) -> Tuple[int, int]:
        """Parse coordinate string '(x,y)' to (x, y) tuple."""
        match = re.match(r'\((\d+),(\d+)\)', coord_str)
        if not match:
            raise ValueError(f"Invalid coordinate format: {coord_str}")
        return int(match.group(1)), int(match.group(2))
    
    # Map UI-TARS actions to AutoAppWorld format
    func_name_lower = func_name.lower()
    
    # click
    if func_name_lower == "click":
        coord_str = kwargs.get("start_box")
        if not coord_str:
            raise ValueError(f"click action requires start_box parameter")
        x, y = parse_coord(coord_str)
        return "action", {"action_type": "click", "x": x, "y": y}, None
    
    # left_double (double click)
    if func_name_lower == "left_double":
        coord_str = kwargs.get("start_box")
        if not coord_str:
            raise ValueError(f"left_double action requires start_box parameter")
        x, y = parse_coord(coord_str)
        # AutoAppWorld doesn't have double click, use two clicks
        return "action", {"action_type": "click", "x": x, "y": y}, None

    # right_single (right click)
    if func_name_lower == "right_single":
        coord_str = kwargs.get("start_box")
        if not coord_str:
            raise ValueError(f"right_single action requires start_box parameter")
        x, y = parse_coord(coord_str)
        # AutoAppWorld uses hover for right click context
        return "action", {"action_type": "hover", "x": x, "y": y}, None

    # drag
    if func_name_lower == "drag":
        start_coord_str = kwargs.get("start_box")
        end_coord_str = kwargs.get("end_box")
        if not start_coord_str or not end_coord_str:
            raise ValueError(f"drag action requires start_box and end_box parameters")
        x1, y1 = parse_coord(start_coord_str)
        x2, y2 = parse_coord(end_coord_str)
        return "action", {
            "action_type": "drag",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }, None

    # hotkey
    if func_name_lower == "hotkey":
        key_str = kwargs.get("key")
        if not key_str:
            raise ValueError(f"hotkey action requires key parameter")
        # Parse key combination (e.g., "ctrl c" -> ["Control", "C"])
        keys = key_str.split()
        # Normalize key names
        normalized_keys = []
        for key in keys:
            key_lower = key.lower()
            if key_lower == "ctrl":
                normalized_keys.append("Control")
            elif key_lower == "alt":
                normalized_keys.append("Alt")
            elif key_lower == "shift":
                normalized_keys.append("Shift")
            else:
                normalized_keys.append(key.upper())
        return "action", {"action_type": "hotkey", "keys": normalized_keys}, None

    # type
    if func_name_lower == "type":
        content = kwargs.get("content")
        if content is None:
            raise ValueError(f"type action requires content parameter")
        # AutoAppWorld uses "type" not "type_text"
        return "action", {"action_type": "type", "text": content}, None

    # scroll
    if func_name_lower == "scroll":
        direction = kwargs.get("direction")
        if not direction:
            raise ValueError(f"scroll action requires direction parameter")
        direction_lower = direction.lower()
        # Convert direction to scroll amount
        if direction_lower == "down":
            amount = -300
        elif direction_lower == "up":
            amount = 300
        elif direction_lower == "left":
            amount = 300  # Horizontal scroll
        elif direction_lower == "right":
            amount = -300  # Horizontal scroll
        else:
            amount = -300  # Default to down
        return "action", {"action_type": "scroll", "amount": amount}, None

    # wait
    if func_name_lower == "wait":
        return "wait", {}, None

    # finished
    if func_name_lower == "finished":
        content = kwargs.get("content", "")
        return "finish", {}, content

    raise ValueError(f"Unrecognized UI-TARS action: {func_name}")


