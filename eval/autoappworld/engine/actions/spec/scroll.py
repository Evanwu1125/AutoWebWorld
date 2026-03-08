from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..base import Action, ActionSpec
from ...core.types import ActionType
from ._utils import get_action_type, get_scroll


class ScrollSpec(ActionSpec):
    action_type: ActionType = ActionType.SCROLL

    def parse(self, raw: Mapping[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Action:
        action_type = get_action_type(raw)
        if action_type != self.action_type:
            raise ValueError(f"Invalid action_type for ScrollSpec: {action_type}")
        amount = get_scroll(raw)
        return Action(action_type=action_type, params={"amount": amount}, meta={"raw": dict(raw)})
