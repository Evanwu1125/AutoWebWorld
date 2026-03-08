"""
OpenCUA action parser for converting PyAutoGUI code format to AutoAppWorld format.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


def normalize_opencua_action(action_text: str) -> str:
    """
    Extract PyAutoGUI code from OpenCUA format.
    
    OpenCUA format:
    Thought: ...
    Action: ...
    pyautogui.click(x=0.5, y=0.3)
    
    Args:
        action_text: Full response text from OpenCUA model
        
    Returns:
        Extracted PyAutoGUI code lines
    """
    text = action_text.strip()
    lines = text.split("\n")
    action_lines = []
    
    # First pass: lines that start with commands
    for raw in lines:
        line = raw.strip()
        if line.startswith("pyautogui.") or line.startswith("computer."):
            action_lines.append(line)
    
    # If we already have extracted lines, return them
    if action_lines:
        return "\n".join(action_lines)
    
    # Second pass: find commands anywhere within lines
    for raw in lines:
        line = raw.strip()
        if "pyautogui." in line:
            parts = line.split("pyautogui.")
            action_lines.append("pyautogui." + parts[1].strip())
        elif "computer." in line:
            parts = line.split("computer.")
            action_lines.append("computer." + parts[1].strip())
    
    return "\n".join(action_lines) if action_lines else text


def parse_opencua_action(
    action_text: str,
    screen_width: int,
    screen_height: int
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Parse OpenCUA PyAutoGUI code format to AutoAppWorld action format.

    OpenCUA-7B outputs ABSOLUTE pixel coordinates (not relative 0-1 coordinates).
    Example: pyautogui.click(x=407, y=94) means click at pixel (407, 94).

    Supported actions:
    - pyautogui.click(x=407, y=94) -> click
    - pyautogui.doubleClick(x=407, y=94) -> click (double click not supported, use single click)
    - pyautogui.rightClick(x=407, y=94) -> hover (right click not supported, use hover)
    - pyautogui.moveTo(x=407, y=94) -> hover
    - pyautogui.dragTo(x=407, y=94) -> click (drag needs start position, use click instead)
    - pyautogui.write(message='text') -> type
    - pyautogui.hotkey('ctrl', 'c') -> hotkey
    - pyautogui.press('enter') -> hotkey
    - pyautogui.scroll(amount) -> scroll
    - computer.triple_click(x=407, y=94) -> click (triple click not supported, use single click)
    - computer.terminate(status='success') -> finish

    Args:
        action_text: Normalized PyAutoGUI code
        screen_width: Screen width in pixels (not used, OpenCUA uses absolute coordinates)
        screen_height: Screen height in pixels (not used, OpenCUA uses absolute coordinates)

    Returns:
        Tuple of (action_kind, action_payload, finish_message)
        - action_kind: "action", "wait", or "finish"
        - action_payload: Dict with action details
        - finish_message: Optional finish message
    """
    text = action_text.strip()

    # Handle empty or invalid input
    if not text:
        return "wait", {}, None

    # Process each line (OpenCUA may output multiple actions)
    # For now, we'll take the first valid action
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # computer.terminate
        if line.startswith("computer.terminate"):
            status_match = re.search(r"status=['\"](\w+)['\"]", line)
            if status_match:
                status = status_match.group(1)
                return "finish", {}, status
            return "finish", {}, "success"

        # computer.triple_click -> click
        if line.startswith("computer.triple_click"):
            coord_match = re.search(r"x=([\d.]+),\s*y=([\d.]+)", line)
            if coord_match:
                x, y = map(float, coord_match.groups())
                # OpenCUA-7B uses absolute pixel coordinates, not relative (0-1)
                abs_x = int(x)
                abs_y = int(y)
                return "action", {"action_type": "click", "x": abs_x, "y": abs_y}, None

        # pyautogui actions
        if line.startswith("pyautogui."):
            # Extract coordinates (for click, moveTo, doubleClick, rightClick, dragTo)
            coord_match = re.search(r"x=([\d.]+),\s*y=([\d.]+)", line)
            if coord_match:
                x, y = map(float, coord_match.groups())
                # OpenCUA-7B uses absolute pixel coordinates, not relative (0-1)
                abs_x = int(x)
                abs_y = int(y)

                if "click" in line and "doubleClick" not in line and "rightClick" not in line:
                    return "action", {"action_type": "click", "x": abs_x, "y": abs_y}, None
                elif "moveTo" in line:
                    return "action", {"action_type": "hover", "x": abs_x, "y": abs_y}, None
                elif "doubleClick" in line:
                    # Double click not supported, use single click
                    return "action", {"action_type": "click", "x": abs_x, "y": abs_y}, None
                elif "rightClick" in line:
                    # Right click not supported, use hover
                    return "action", {"action_type": "hover", "x": abs_x, "y": abs_y}, None
                elif "dragTo" in line:
                    # Drag needs start position - for now, use click
                    # TODO: Track previous position for proper drag support
                    return "action", {"action_type": "click", "x": abs_x, "y": abs_y}, None
            
            # write(message='...')
            write_match = re.search(r"message=['\"](.+?)['\"]", line)
            if write_match:
                text_content = write_match.group(1)
                # AutoAppWorld uses "type" not "type_text"
                return "action", {"action_type": "type", "text": text_content}, None

            # write('...') positional
            write_positional = re.search(r"pyautogui\.write\((['\"])(.*?)\1\)", line)
            if write_positional:
                text_content = write_positional.group(2)
                # AutoAppWorld uses "type" not "type_text"
                return "action", {"action_type": "type", "text": text_content}, None

            # hotkey with keys=[...]
            keys_match = re.findall(r"keys=\[(.*?)\]", line)
            if keys_match:
                key_string = keys_match[0]
                key_list = re.findall(r"['\"]([^'\"]*)['\"]|(\w+)", key_string)
                keys = [m[0] or m[1] for m in key_list if m[0] or m[1]]
                normalized_keys = []
                for k in keys:
                    k = k.strip()
                    # Normalize cmd/command to Control
                    if k.lower() in ("cmd", "command", "ctrl"):
                        normalized_keys.append("Control")
                    elif k.lower() == "enter":
                        normalized_keys.append("Enter")
                    elif k.lower() == "shift":
                        normalized_keys.append("Shift")
                    elif k.lower() == "alt":
                        normalized_keys.append("Alt")
                    else:
                        normalized_keys.append(k)
                return "action", {"action_type": "hotkey", "keys": normalized_keys}, None

            # hotkey positional: pyautogui.hotkey('ctrl', 'v')
            if "hotkey(" in line and "keys=" not in line:
                inside = re.search(r"pyautogui\.hotkey\((.*)\)", line)
                if inside:
                    arg_str = inside.group(1)
                    parts = re.findall(r"['\"]([^'\"]+)['\"]", arg_str)
                    if parts:
                        normalized_keys = []
                        for p in parts:
                            p = p.strip()
                            if p.lower() in ("cmd", "command", "ctrl"):
                                normalized_keys.append("Control")
                            elif p.lower() == "enter":
                                normalized_keys.append("Enter")
                            elif p.lower() == "shift":
                                normalized_keys.append("Shift")
                            elif p.lower() == "alt":
                                normalized_keys.append("Alt")
                            else:
                                normalized_keys.append(p)
                        return "action", {"action_type": "hotkey", "keys": normalized_keys}, None

            # press positional: pyautogui.press('enter') or press(['ctrl','v'])
            if "press(" in line and "keys=" not in line:
                inside = re.search(r"pyautogui\.press\((.*)\)", line)
                if inside:
                    arg_str = inside.group(1).strip()
                    keys = []
                    if arg_str.startswith("["):
                        parts = re.findall(r"['\"]([^'\"]+)['\"]", arg_str)
                        keys = [p.strip() for p in parts]
                    else:
                        one = re.search(r"['\"]([^'\"]+)['\"]", arg_str)
                        if one:
                            keys = [one.group(1).strip()]
                    if keys:
                        normalized_keys = []
                        for k in keys:
                            if k.lower() in ("cmd", "command", "ctrl"):
                                normalized_keys.append("Control")
                            elif k.lower() == "enter":
                                normalized_keys.append("Enter")
                            elif k.lower() == "shift":
                                normalized_keys.append("Shift")
                            elif k.lower() == "alt":
                                normalized_keys.append("Alt")
                            else:
                                normalized_keys.append(k)
                        return "action", {"action_type": "hotkey", "keys": normalized_keys}, None

            # scroll
            scroll_match = re.search(r"pyautogui\.scroll\(([-\d]+)\)", line)
            if scroll_match:
                amount = int(scroll_match.group(1))
                # PyAutoGUI scroll: positive = up, negative = down
                # AutoAppWorld scroll: positive = down, negative = up
                # So we need to invert the sign
                return "action", {"action_type": "scroll", "amount": -amount}, None

    # If no valid action found, return wait
    return "wait", {}, None

