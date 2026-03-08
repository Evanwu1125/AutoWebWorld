from __future__ import annotations

import asyncio
from typing import Any, Optional

from ..core.async_engine import AsyncActionEngine
from ..core.registry import register_all
from ..core.types import BackendType
from .base import BaseExecutor


class PlaywrightExecutor(BaseExecutor):
    backend: BackendType = BackendType.PLAYWRIGHT

    def __init__(self, engine: Optional[AsyncActionEngine] = None) -> None:
        super().__init__(engine=engine)
        register_all(self.engine)
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def init_env(
        self,
        url: Optional[str] = None,
        *,
        browser: str = "chromium",
        headless: bool = True,
        viewport: Optional[tuple[int, int]] = (1280, 720),
        **kwargs: Any,
    ) -> None:
        from playwright.async_api import async_playwright

        try:
            self._playwright = await async_playwright().start()
            browser_type = getattr(self._playwright, browser)
            self._browser = await browser_type.launch(headless=headless, **kwargs)
            context_kwargs: dict[str, Any] = {}
            if viewport is None:
                context_kwargs["no_viewport"] = True
            else:
                context_kwargs["viewport"] = {"width": viewport[0], "height": viewport[1]}
            self._context = await self._browser.new_context(**context_kwargs)
            self._page = await self._context.new_page()
            if url:
                await self._page.goto(url)
            self._ctx = {"page": self._page}
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        context = self._context
        browser = self._browser
        playwright = self._playwright
        self._context = None
        self._browser = None
        self._playwright = None

        await _safe_await(context.close() if context is not None else None, timeout=5)
        await _safe_await(browser.close() if browser is not None else None, timeout=5)
        await _safe_await(playwright.stop() if playwright is not None else None, timeout=5)



async def _safe_await(awaitable: Optional[Any], timeout: float) -> None:
    if awaitable is None:
        return
    try:
        await asyncio.wait_for(awaitable, timeout=timeout)
    except Exception:
        return
