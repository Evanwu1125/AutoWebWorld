from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from ...base import Action, ActionResult, AsyncActionHandler
from ....core.types import ActionType, BackendType
from ....utils._utils import require_ctx_value

if TYPE_CHECKING:
    from playwright.async_api import Page


class PlaywrightAsyncScrollSelectorHandler(AsyncActionHandler):
    """Scroll handler based on selector."""
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.SCROLL_SELECTOR

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        selector = str(action.params.get("selector", "html"))
        amount = int(action.params.get("amount", 100))
        timeout = action.params.get("timeout", 5000)

        try:
            element = page.locator(selector).first
            await element.wait_for(state="attached", timeout=timeout)
            bbox = await element.bounding_box()

            # Scroll element into view center, then scroll on top of it
            if bbox:
                center_x = bbox["x"] + bbox["width"] / 2
                center_y = bbox["y"] + bbox["height"] / 2
                await page.mouse.move(center_x, center_y)

            await page.mouse.wheel(0, amount)

            return ActionResult.success(
                action_type=action.action_type,
                backend=self.backend,
                data={
                    "selector": selector,
                    "amount": amount,
                    "bbox": bbox,
                },
            )
        except Exception as e:
            return ActionResult.failure(
                action_type=action.action_type,
                backend=self.backend,
                error=str(e),
                data={"selector": selector, "amount": amount},
            )
