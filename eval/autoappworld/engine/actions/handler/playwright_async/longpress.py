from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from ...base import Action, ActionResult, AsyncActionHandler
from ....core.types import ActionType, BackendType
from ....utils._utils import require_ctx_value

if TYPE_CHECKING:
    from playwright.async_api import Page


class PlaywrightAsyncLongPressHandler(AsyncActionHandler):
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.LONGPRESS

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        x = int(action.params["x"])
        y = int(action.params["y"])
        
        # Simulate long press: mouse down, wait, mouse up
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.wait_for_timeout(1000)  # Hold for 1 second
        await page.mouse.up()
        
        return ActionResult.success(
            action_type=action.action_type,
            backend=self.backend,
            data={"x": x, "y": y},
        )

