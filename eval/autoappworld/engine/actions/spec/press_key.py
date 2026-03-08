from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..base import Action
from ...core.types import ActionType


def parse_press_back(text: str) -> Optional[Action]:
    """
    Parse PressBack action from text.
    
    Supported formats:
    - PressBack()
    - press_back
    
    Args:
        text: Action text to parse
        
    Returns:
        Action object or None if parsing fails
    """
    text = text.strip()
    
    if re.match(r"PressBack\s*\(\s*\)", text, re.IGNORECASE) or re.match(r"press_back", text, re.IGNORECASE):
        return Action(action_type=ActionType.PRESS_BACK, params={})
    
    return None


def parse_press_home(text: str) -> Optional[Action]:
    """
    Parse PressHome action from text.
    
    Supported formats:
    - PressHome()
    - press_home
    
    Args:
        text: Action text to parse
        
    Returns:
        Action object or None if parsing fails
    """
    text = text.strip()
    
    if re.match(r"PressHome\s*\(\s*\)", text, re.IGNORECASE) or re.match(r"press_home", text, re.IGNORECASE):
        return Action(action_type=ActionType.PRESS_HOME, params={})
    
    return None


def parse_press_enter(text: str) -> Optional[Action]:
    """
    Parse PressEnter action from text.
    
    Supported formats:
    - PressEnter()
    - press_enter
    
    Args:
        text: Action text to parse
        
    Returns:
        Action object or None if parsing fails
    """
    text = text.strip()
    
    if re.match(r"PressEnter\s*\(\s*\)", text, re.IGNORECASE) or re.match(r"press_enter", text, re.IGNORECASE):
        return Action(action_type=ActionType.PRESS_ENTER, params={})
    
    return None


def parse_press_recent(text: str) -> Optional[Action]:
    """
    Parse PressRecent action from text.
    
    Supported formats:
    - PressRecent()
    - press_recent
    
    Args:
        text: Action text to parse
        
    Returns:
        Action object or None if parsing fails
    """
    text = text.strip()
    
    if re.match(r"PressRecent\s*\(\s*\)", text, re.IGNORECASE) or re.match(r"press_recent", text, re.IGNORECASE):
        return Action(action_type=ActionType.PRESS_RECENT, params={})
    
    return None

