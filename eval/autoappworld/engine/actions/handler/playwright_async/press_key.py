from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from ...base import Action, ActionResult, AsyncActionHandler
from ....core.types import ActionType, BackendType
from ....utils._utils import require_ctx_value

if TYPE_CHECKING:
    from playwright.async_api import Page


class PlaywrightAsyncPressBackHandler(AsyncActionHandler):
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.PRESS_BACK

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        
        # Simulate browser back button (Alt+Left or Backspace)
        await page.keyboard.press("Alt+ArrowLeft")
        
        return ActionResult.success(
            action_type=action.action_type,
            backend=self.backend,
            data={},
        )


class PlaywrightAsyncPressHomeHandler(AsyncActionHandler):
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.PRESS_HOME

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        
        # Simulate Home key
        await page.keyboard.press("Home")
        
        return ActionResult.success(
            action_type=action.action_type,
            backend=self.backend,
            data={},
        )


class PlaywrightAsyncPressEnterHandler(AsyncActionHandler):
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.PRESS_ENTER

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        
        # Simulate Enter key
        await page.keyboard.press("Enter")
        
        return ActionResult.success(
            action_type=action.action_type,
            backend=self.backend,
            data={},
        )


class PlaywrightAsyncPressRecentHandler(AsyncActionHandler):
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.PRESS_RECENT

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        
        # Simulate Alt+Tab (switch to recent window/tab)
        # Note: This may not work in all browsers/contexts
        await page.keyboard.press("Alt+Tab")
        
        return ActionResult.success(
            action_type=action.action_type,
            backend=self.backend,
            data={},
        )

