from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..base import Action, ActionSpec
from ...core.types import ActionType
from ._utils import get_action_type, get_drag_points


class DragSpec(ActionSpec):
    action_type: ActionType = ActionType.DRAG

    def parse(self, raw: Mapping[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Action:
        action_type = get_action_type(raw)
        if action_type != self.action_type:
            raise ValueError(f"Invalid action_type for DragSpec: {action_type}")
        x1, y1, x2, y2 = get_drag_points(raw)
        return Action(
            action_type=action_type,
            params={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            meta={"raw": dict(raw)},
        )
