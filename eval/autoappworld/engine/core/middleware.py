from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..actions.base import ActionResult


class AsyncActionMiddleware(ABC):
    @abstractmethod
    async def before_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def after_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
        result: ActionResult,
    ) -> ActionResult:
        raise NotImplementedError


class ActionPipeline:
    def __init__(self, middlewares: Optional[list[AsyncActionMiddleware]] = None) -> None:
        self._middlewares = middlewares or []

    def add(self, middleware: AsyncActionMiddleware) -> None:
        self._middlewares.append(middleware)

    async def before_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        for middleware in self._middlewares:
            ctx = await middleware.before_execute(raw, ctx)
        return ctx

    async def after_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
        result: ActionResult,
    ) -> ActionResult:
        for middleware in reversed(self._middlewares):
            result = await middleware.after_execute(raw, ctx, result)
        return result
