from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional

from ..actions.base import ActionResult
from ..core.async_engine import AsyncActionEngine
from ..core.types import BackendType


class BaseExecutor(ABC):
    """Base async executor that manages environment and runs actions."""

    backend: BackendType

    def __init__(self, engine: Optional[AsyncActionEngine] = None) -> None:
        self.engine = engine or AsyncActionEngine()
        self._ctx: Dict[str, Any] = {}

    @abstractmethod
    async def init_env(self, **kwargs: Any) -> None:
        """Initialize execution environment and populate default ctx."""
        raise NotImplementedError

    async def execute(
        self,
        raw: Mapping[str, Any],
        ctx: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        merged = dict(self._ctx)
        if ctx:
            merged.update(ctx)
        return await self.engine.execute(raw, backend=self.backend, ctx=merged)

    async def close(self) -> None:
        """Release resources for the executor."""
        return None
