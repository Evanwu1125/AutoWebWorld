from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..base import Action, ActionSpec
from ...core.types import ActionType
from ._utils import get_action_type, get_hotkey


class HotkeySpec(ActionSpec):
    action_type: ActionType = ActionType.HOTKEY

    def parse(self, raw: Mapping[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Action:
        action_type = get_action_type(raw)
        if action_type != self.action_type:
            raise ValueError(f"Invalid action_type for HotkeySpec: {action_type}")
        keys = list(get_hotkey(raw))
        if not keys:
            raise ValueError("Hotkey keys cannot be empty")
        return Action(action_type=action_type, params={"keys": keys}, meta={"raw": dict(raw)})
