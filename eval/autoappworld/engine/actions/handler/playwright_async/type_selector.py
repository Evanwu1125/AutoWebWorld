from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from ...base import Action, ActionResult, AsyncActionHandler
from ....core.types import ActionType, BackendType
from ....utils._utils import require_ctx_value

if TYPE_CHECKING:
    from playwright.async_api import Page


class PlaywrightAsyncTypeSelectorHandler(AsyncActionHandler):
    """Type handler based on selector."""
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.TYPE_SELECTOR

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        selector = str(action.params["selector"])
        text = str(action.params.get("text", ""))
        timeout = action.params.get("timeout", 5000)
        clear = action.params.get("clear", False)

        try:
            element = page.locator(selector).first
            await element.wait_for(state="visible", timeout=timeout)
            bbox = await element.bounding_box()

            if clear:
                await element.fill(text, timeout=timeout)
            else:
                await element.click(timeout=timeout)
                await page.keyboard.type(text)

            return ActionResult.success(
                action_type=action.action_type,
                backend=self.backend,
                data={
                    "selector": selector,
                    "text": text,
                    "bbox": bbox,
                },
            )
        except Exception as e:
            return ActionResult.failure(
                action_type=action.action_type,
                backend=self.backend,
                error=str(e),
                data={"selector": selector, "text": text},
            )
