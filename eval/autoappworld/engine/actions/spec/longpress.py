from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..base import Action
from ...core.types import ActionType


def parse_longpress(text: str) -> Optional[Action]:
    """
    Parse LongPress action from text.
    
    Supported formats:
    - LongPress(box=(x, y))
    - longpress (x, y)
    
    Args:
        text: Action text to parse
        
    Returns:
        Action object or None if parsing fails
    """
    text = text.strip()
    
    # Format 1: LongPress(box=(x, y))
    match = re.match(r"LongPress\s*\(\s*box\s*=\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*\)", text, re.IGNORECASE)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        return Action(action_type=ActionType.LONGPRESS, params={"x": x, "y": y})
    
    # Format 2: longpress (x, y)
    match = re.match(r"longpress\s+\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", text, re.IGNORECASE)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        return Action(action_type=ActionType.LONGPRESS, params={"x": x, "y": y})
    
    return None

