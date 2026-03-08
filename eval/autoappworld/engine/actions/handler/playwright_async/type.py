from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from ...base import Action, ActionResult, AsyncActionHandler
from ....core.types import ActionType, BackendType
from ....utils._utils import require_ctx_value

if TYPE_CHECKING:
    from playwright.async_api import Page


class PlaywrightAsyncTypeHandler(AsyncActionHandler):
    backend: BackendType = BackendType.PLAYWRIGHT
    action_type: ActionType = ActionType.TYPE

    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        page: Page = require_ctx_value(ctx, "page")
        text = str(action.params["text"])
        await page.keyboard.type(text)
        return ActionResult.success(
            action_type=action.action_type,
            backend=self.backend,
            data={"text": text},
        )
