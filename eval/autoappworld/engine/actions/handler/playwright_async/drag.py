from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from ...base import Action, ActionResult, AsyncActionHandler
from ....core.types import ActionType, BackendType
from ....utils._utils import require_ctx_value

if TYPE_CHECKING:
    from playwright.async_api import Page


class PlaywrightAsyncDragHandler(AsyncActionHandler):
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.DRAG

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        x1 = int(action.params["x1"])
        y1 = int(action.params["y1"])
        x2 = int(action.params["x2"])
        y2 = int(action.params["y2"])
        await page.mouse.move(x1, y1)
        await page.mouse.down()
        await page.mouse.move(x2, y2)
        await page.mouse.up()
        return ActionResult.success(
            action_type=action.action_type,
            backend=self.backend,
            data={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        )
