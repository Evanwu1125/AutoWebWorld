from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from ...base import Action, ActionResult, AsyncActionHandler
from ....core.types import ActionType, BackendType
from ....utils._utils import normalize_playwright_keys, require_ctx_value

if TYPE_CHECKING:
    from playwright.async_api import Page


class PlaywrightAsyncHotkeyHandler(AsyncActionHandler):
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.HOTKEY

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        keys = action.params["keys"]
        chord = normalize_playwright_keys(keys)
        if not chord:
            raise ValueError("Empty hotkey chord")
        await page.keyboard.press(chord)
        return ActionResult.success(action_type=action.action_type, backend=self.backend, data={"keys": list(keys)})
