from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from ...base import Action, ActionResult, AsyncActionHandler
from ....core.types import ActionType, BackendType
from ....utils._utils import require_ctx_value

if TYPE_CHECKING:
    from playwright.async_api import Page


class PlaywrightAsyncScrollHandler(AsyncActionHandler):
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.SCROLL

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        # amount = int(action.params.get("amount", 200))  # 🔥 default 200px

        amount = 400
        # 🔥 Use JavaScript to scroll by specified pixels
        # window.scrollBy(0, amount) - scroll down by amount pixels
        await page.evaluate(f"window.scrollBy(0, {amount})")

        return ActionResult.success(
            action_type=action.action_type,
            backend=self.backend,
            data={"amount": amount, "method": "javascript_scrollBy"},
        )

        # 🔥 Original logic (commented out): scroll using mouse wheel
        # delta_y = amount
        # mouse_info = await page.evaluate(
        #     """
        #     () => {
        #       if (!window.__actionMouse) {
        #         window.__actionMouse = {
        #           x: Math.floor(window.innerWidth / 2),
        #           y: Math.floor(window.innerHeight / 2),
        #         };
        #         window.addEventListener("mousemove", (e) => {
        #           window.__actionMouse = { x: e.clientX, y: e.clientY };
        #         }, { passive: true });
        #       }
        #       const el = document.elementFromPoint(window.__actionMouse.x, window.__actionMouse.y);
        #       return {
        #         x: window.__actionMouse.x,
        #         y: window.__actionMouse.y,
        #         element: el ? {
        #           tag: el.tagName,
        #           id: el.id || null,
        #           className: el.className || null,
        #           text: el.innerText || null,
        #           ariaLabel: el.getAttribute("aria-label"),
        #         } : null,
        #       };
        #     }
        #     """
        # )
        # await page.mouse.wheel(0, delta_y)
        # return ActionResult.success(
        #     action_type=action.action_type,
        #     backend=self.backend,
        #     data={"amount": amount, "delta_y": delta_y, "mouse": mouse_info},
        # )
