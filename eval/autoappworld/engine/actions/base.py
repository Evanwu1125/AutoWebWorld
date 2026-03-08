from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, Field

from ..core.types import ActionType, BackendType


class Action(BaseModel):
    """Semantic action data produced by ActionSpec."""

    action_type: ActionType
    params: Dict[str, Any]
    meta: Dict[str, Any] = Field(default_factory=dict)


class ActionSpec(BaseModel, ABC):
    """Defines parsing and validation rules for a specific action type."""

    action_type: ActionType

    @abstractmethod
    def parse(self, raw: Mapping[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Action:
        """Parse raw input into a normalized Action instance."""
        raise NotImplementedError


class ActionResult(BaseModel):
    ok: bool
    action_type: ActionType
    backend: BackendType
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    exception: Optional[str] = None
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(
        cls,
        action_type: ActionType,
        backend: BackendType,
        data: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ActionResult":
        return cls(
            ok=True,
            action_type=action_type,
            backend=backend,
            data=data or {},
            artifacts=artifacts or {},
            meta=meta or {},
        )

    @classmethod
    def failure(
        cls,
        action_type: ActionType,
        backend: BackendType,
        error: str,
        exception: Optional[BaseException] = None,
        data: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ActionResult":
        exc_text = None
        if exception is not None:
            exc_text = f"{type(exception).__name__}: {exception}"
        return cls(
            ok=False,
            action_type=action_type,
            backend=backend,
            data=data or {},
            error=error,
            exception=exc_text,
            artifacts=artifacts or {},
            meta=meta or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        try:
            return self.model_dump()
        except AttributeError:
            return self.dict()


class ActionHandler(BaseModel, ABC):
    """Executes an action on a specific backend."""

    backend: BackendType
    action_type: ActionType

    @abstractmethod
    def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        """Execute the action and return a standardized result."""
        raise NotImplementedError


class AsyncActionHandler(BaseModel, ABC):
    """Executes an action on a specific backend asynchronously."""

    backend: BackendType
    action_type: ActionType

    @abstractmethod
    async def handle(self, action: Action, ctx: Optional[Dict[str, Any]] = None) -> ActionResult:
        """Execute the action and return a standardized result."""
        raise NotImplementedError
