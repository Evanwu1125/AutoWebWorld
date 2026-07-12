"""
UI-Venus action parser for converting UI-Venus format actions to AutoAppWorld format.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


def normalize_ui_venus_action(action_text: str) -> str:
    """
    Extract action from UI-Venus format.
    
    UI-Venus format:
    <think>...</think>
    <action>Click(box=(100, 200))</action>
    <conclusion>...</conclusion>
    
    Args:
        action_text: Full response text from UI-Venus model
        
    Returns:
        Extracted action text
    """
    text = action_text.strip()
    
    # Extract from <action> tags
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return text


def parse_ui_venus_action(action_text: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Parse UI-Venus action format to AutoAppWorld format.
    
    Supported UI-Venus actions:
    - Click(box=(x, y))
    - Drag(start=(x1, y1), end=(x2, y2))
    - Scroll(start=(x1, y1), end=(x2, y2), direction='down/up/right/left')
    - Scroll(direction='down/up/right/left')
    - Type(content='text')
    - Launch(app='app_name')
    - Wait()
    - Finished(content='result')
    - CallUser(content='message')
    - LongPress(box=(x, y))
    - PressBack()
    - PressHome()
    - PressEnter()
    - PressRecent()
    
    Returns:
        Tuple of (action_kind, action_params, finish_message)
        - action_kind: "action", "wait", or "finish"
        - action_params: Dict with action details
        - finish_message: Optional finish message
    """
    text = action_text.strip()
    lowered = text.lower()
    
    # Wait
    if re.match(r"^Wait\s*\(\s*\)", text, re.IGNORECASE):
        return "wait", {}, None
    
    # Finished
    finished_match = re.match(r"^Finished\s*\(\s*content\s*=\s*['\"](.*)['\"]\s*\)", text, re.IGNORECASE)
    if finished_match:
        return "finish", {}, finished_match.group(1).strip()
    
    # Click(box=(x, y))
    click_match = re.match(r"^Click\s*\(\s*box\s*=\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*\)", text, re.IGNORECASE)
    if click_match:
        return "action", {
            "action_type": "click",
            "x": int(float(click_match.group(1))),
            "y": int(float(click_match.group(2)))
        }, None
    
    # LongPress(box=(x, y))
    longpress_match = re.match(r"^LongPress\s*\(\s*box\s*=\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*\)", text, re.IGNORECASE)
    if longpress_match:
        return "action", {
            "action_type": "longpress",
            "x": int(float(longpress_match.group(1))),
            "y": int(float(longpress_match.group(2)))
        }, None
    
    # Drag(start=(x1, y1), end=(x2, y2))
    drag_match = re.match(
        r"^Drag\s*\(\s*start\s*=\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*,\s*end\s*=\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*\)",
        text,
        re.IGNORECASE
    )
    if drag_match:
        return "action", {
            "action_type": "drag",
            "x1": int(float(drag_match.group(1))),
            "y1": int(float(drag_match.group(2))),
            "x2": int(float(drag_match.group(3))),
            "y2": int(float(drag_match.group(4)))
        }, None
    
    # Scroll with coordinates and direction
    scroll_full_match = re.match(
        r"^Scroll\s*\(\s*start\s*=\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*,\s*end\s*=\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*,\s*direction\s*=\s*['\"](\w+)['\"]\s*\)",
        text,
        re.IGNORECASE
    )
    if scroll_full_match:
        # Convert to scroll amount based on direction
        direction = scroll_full_match.group(5).lower()
        y1 = int(float(scroll_full_match.group(2)))
        y2 = int(float(scroll_full_match.group(4)))
        amount = y2 - y1  # Positive = scroll down, Negative = scroll up
        if direction in ['up', 'down']:
            return "action", {"action_type": "scroll", "amount": amount}, None
        else:
            # For left/right, use horizontal scroll (not standard in current implementation)
            return "action", {"action_type": "scroll", "amount": amount}, None
    
    # Scroll(direction='down/up/right/left')
    scroll_dir_match = re.match(r"^Scroll\s*\(\s*direction\s*=\s*['\"](\w+)['\"]\s*\)", text, re.IGNORECASE)
    if scroll_dir_match:
        direction = scroll_dir_match.group(1).lower()
        # Default scroll amount
        amount = -300 if direction == 'down' else 300 if direction == 'up' else 0
        return "action", {"action_type": "scroll", "amount": amount}, None
    
    # Type(content='text')
    type_match = re.match(r"^Type\s*\(\s*content\s*=\s*['\"](.*)['\"]\s*\)", text, re.IGNORECASE | re.DOTALL)
    if type_match:
        return "action", {"action_type": "type", "text": type_match.group(1)}, None
    
    # PressBack(), PressHome(), PressEnter(), PressRecent()
    if re.match(r"^PressBack\s*\(\s*\)", text, re.IGNORECASE):
        return "action", {"action_type": "press_back"}, None
    if re.match(r"^PressHome\s*\(\s*\)", text, re.IGNORECASE):
        return "action", {"action_type": "press_home"}, None
    if re.match(r"^PressEnter\s*\(\s*\)", text, re.IGNORECASE):
        return "action", {"action_type": "press_enter"}, None
    if re.match(r"^PressRecent\s*\(\s*\)", text, re.IGNORECASE):
        return "action", {"action_type": "press_recent"}, None
    
    raise ValueError(f"Unrecognized UI-Venus action: {action_text}")

