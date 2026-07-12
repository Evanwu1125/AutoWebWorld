"""
TongUI action parser for converting TongUI format actions to AutoAppWorld format.
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, Optional, Tuple


def normalize_tongui_action(action_text: str) -> str:
    """
    Extract action from TongUI format.
    
    TongUI format:
    <think>...</think>
    <action>{"action": "CLICK", "value": null, "position": [0.45, 0.32]}</action>
    
    Args:
        action_text: Full response text from TongUI model
        
    Returns:
        Extracted action JSON text
    """
    text = action_text.strip()
    
    # Extract from <action> tags
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return text


def parse_tongui_action(action_text: str, screen_width: int = 1280, screen_height: int = 720) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Parse TongUI action format to AutoAppWorld format.

    TongUI uses normalized coordinates (0-1 range) and JSON format:
    {"action": "ACTION_TYPE", "value": "element", "position": [x,y]}

    Supported TongUI actions:
    - CLICK: Click on an element, position [x,y] required
    - INPUT: Type a string, value and position [x,y] required
    - SELECT: Select a value, position [x,y] required
    - HOVER: Hover on an element, position [x,y] required
    - ANSWER: Answer the question, value required
    - ENTER: Enter operation
    - SCROLL: Scroll the screen, value is direction
    - SELECT_TEXT: Select text, position [[x1,y1], [x2,y2]] required
    - COPY: Copy text, value is the text

    Also supports multiple actions as JSON array.

    Args:
        action_text: Action JSON text
        screen_width: Screen width for coordinate conversion (default: 1280)
        screen_height: Screen height for coordinate conversion (default: 720)
        
    Returns:
        Tuple of (action_kind, action_params, finish_message)
        - action_kind: "action", "wait", or "finish"
        - action_params: Dict with action details
        - finish_message: Optional finish message
    """
    text = action_text.strip()

    # 🔥 Try to parse as JSON
    # First, try to handle Python-style format (None, True, False) -> JSON format (null, true, false)
    text_normalized = text.replace("None", "null").replace("True", "true").replace("False", "false")

    try:
        action_data = json.loads(text_normalized)
    except json.JSONDecodeError:
        # If JSON parsing fails, try to evaluate as Python literal (more permissive)
        try:
            import ast
            action_data = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            raise ValueError(f"Invalid TongUI action JSON: {action_text}")
    
    # Handle multiple actions (array)
    if isinstance(action_data, list):
        # For now, only execute the first action
        # TODO: Support multiple actions in sequence
        if len(action_data) == 0:
            raise ValueError("Empty action array")
        action_data = action_data[0]
    
    if not isinstance(action_data, dict):
        raise ValueError(f"Invalid TongUI action format: {action_text}")
    
    action_type = action_data.get("action", "").upper()
    value = action_data.get("value")
    position = action_data.get("position")
    
    # Convert normalized coordinates to absolute coordinates
    def denormalize_coord(norm_x: float, norm_y: float) -> Tuple[int, int]:
        """Convert normalized coordinates (0-1) to absolute pixel coordinates."""
        abs_x = int(norm_x * screen_width)
        abs_y = int(norm_y * screen_height)
        return abs_x, abs_y
    
    # CLICK
    if action_type == "CLICK":
        if position is None or len(position) != 2:
            raise ValueError(f"CLICK action requires position [x,y]: {action_text}")
        x, y = denormalize_coord(position[0], position[1])
        return "action", {"action_type": "click", "x": x, "y": y}, None
    
    # INPUT
    if action_type == "INPUT":
        if position is None or len(position) != 2:
            raise ValueError(f"INPUT action requires position [x,y]: {action_text}")
        if value is None:
            raise ValueError(f"INPUT action requires value: {action_text}")
        x, y = denormalize_coord(position[0], position[1])
        # First click to focus, then type
        # Return click action, the type action will be handled separately
        # For simplicity, we'll combine them into a single action with metadata
        return "action", {
            "action_type": "click_and_type",
            "x": x,
            "y": y,
            "text": value
        }, None
    
    # SELECT
    if action_type == "SELECT":
        if position is None or len(position) != 2:
            raise ValueError(f"SELECT action requires position [x,y]: {action_text}")
        x, y = denormalize_coord(position[0], position[1])
        return "action", {"action_type": "click", "x": x, "y": y}, None
    
    # HOVER
    if action_type == "HOVER":
        if position is None or len(position) != 2:
            raise ValueError(f"HOVER action requires position [x,y]: {action_text}")
        x, y = denormalize_coord(position[0], position[1])
        return "action", {"action_type": "hover", "x": x, "y": y}, None
    
    # ANSWER
    if action_type == "ANSWER":
        if value is None:
            raise ValueError(f"ANSWER action requires value: {action_text}")
        return "finish", {}, value
    
    # ENTER
    if action_type == "ENTER":
        return "action", {"action_type": "hotkey", "keys": ["Enter"]}, None
    
    # SCROLL
    if action_type == "SCROLL":
        if value is None:
            raise ValueError(f"SCROLL action requires value (direction): {action_text}")
        direction = value.lower()
        # Convert direction to scroll amount
        amount = -300 if direction == 'down' else 300 if direction == 'up' else 0
        return "action", {"action_type": "scroll", "amount": amount}, None

    # SELECT_TEXT
    if action_type == "SELECT_TEXT":
        if position is None or not isinstance(position, list) or len(position) != 2:
            raise ValueError(f"SELECT_TEXT action requires position [[x1,y1], [x2,y2]]: {action_text}")
        start_pos = position[0]
        end_pos = position[1]
        if len(start_pos) != 2 or len(end_pos) != 2:
            raise ValueError(f"SELECT_TEXT action requires position [[x1,y1], [x2,y2]]: {action_text}")
        x1, y1 = denormalize_coord(start_pos[0], start_pos[1])
        x2, y2 = denormalize_coord(end_pos[0], end_pos[1])
        return "action", {
            "action_type": "drag",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }, None

    # COPY
    if action_type == "COPY":
        # COPY action: copy text to clipboard
        # This is typically done via Ctrl+C after selecting text
        # For now, we'll just return a hotkey action
        return "action", {"action_type": "hotkey", "keys": ["Control", "C"]}, None

    raise ValueError(f"Unrecognized TongUI action type: {action_type}")

