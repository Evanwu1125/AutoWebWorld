from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from ...base import Action, ActionResult, AsyncActionHandler
from ....core.types import ActionType, BackendType
from ....utils._utils import require_ctx_value

if TYPE_CHECKING:
    from playwright.async_api import Page


class PlaywrightAsyncClickSelectorHandler(AsyncActionHandler):
    """Click handler based on selector."""
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.CLICK_SELECTOR

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        selector = str(action.params["selector"])
        timeout = action.params.get("timeout", 5000)

        try:
            element = page.locator(selector).first
            await element.wait_for(state="visible", timeout=timeout)
            bbox = await element.bounding_box()
            await element.click(timeout=timeout)

            return ActionResult.success(
                action_type=action.action_type,
                backend=self.backend,
                data={
                    "selector": selector,
                    "bbox": bbox,
                },
            )
        except Exception as e:
            return ActionResult.failure(
                action_type=action.action_type,
                backend=self.backend,
                error=str(e),
                data={"selector": selector},
            )
