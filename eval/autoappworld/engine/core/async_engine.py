from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from ..actions.base import Action, ActionResult, ActionSpec, AsyncActionHandler
from .middleware import ActionPipeline
from ..middleware.impl import (
    ActionIndexMiddleware,
    MousePositionMiddleware,
    ActionMetaMiddleware,
    CaptureMiddleware,
    DomCaptureMiddleware,
    AnnotateMiddleware,
)
from .types import ActionType, BackendType, parse_action_type, parse_backend


class AsyncActionEngine:
    """Async entry point: parse raw actions, route to handlers, and execute."""

    def __init__(self) -> None:
        self._specs: Dict[ActionType, ActionSpec] = {}
        self._handlers: Dict[Tuple[BackendType, ActionType], AsyncActionHandler] = {}
        self._pipeline = ActionPipeline(
            [
                ActionIndexMiddleware(),
                MousePositionMiddleware(),
                DomCaptureMiddleware(),
                AnnotateMiddleware(),
                CaptureMiddleware(),
                ActionMetaMiddleware(),
            ]
        )

    def register_spec(self, spec: ActionSpec) -> None:
        self._specs[spec.action_type] = spec

    def register_handler(self, handler: AsyncActionHandler) -> None:
        key = (handler.backend, handler.action_type)
        self._handlers[key] = handler

    def parse(self, raw: Mapping[str, Any], ctx: Optional[Dict[str, Any]] = None) -> Action:
        action_type = self._get_action_type(raw)
        spec = self._specs.get(action_type)
        if spec is None:
            raise ValueError(f"No ActionSpec registered for action_type '{action_type}'")
        return spec.parse(raw, ctx)

    async def execute(
        self,
        raw: Mapping[str, Any],
        backend: BackendType | str,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> ActionResult:
        backend_type = parse_backend(backend)
        try:
            action = self.parse(raw, ctx)
        except Exception as exc:
            action_type = self._safe_action_type(raw)
            return ActionResult.failure(
                action_type=action_type,
                backend=backend_type,
                error="Action parse failed",
                exception=exc,
            )

        handler = self._handlers.get((backend_type, action.action_type))
        if handler is None:
            return ActionResult.failure(
                action_type=action.action_type,
                backend=backend_type,
                error=(
                    f"No ActionHandler for backend '{backend_type.value}' "
                    f"and action_type '{action.action_type.value}'"
                ),
            )

        merged = dict(ctx or {})
        merged = await self._pipeline.before_execute(dict(raw), merged)
        try:
            result = await handler.handle(action, merged)
        except Exception as exc:
            result = ActionResult.failure(
                action_type=action.action_type,
                backend=backend_type,
                error="Action execution failed",
                exception=exc,
            )
        return await self._pipeline.after_execute(dict(raw), merged, result)

    @staticmethod
    def _get_action_type(raw: Mapping[str, Any]) -> ActionType:
        action_type = raw.get("action_type") or raw.get("type")
        if not action_type:
            raise ValueError("Missing action_type in raw action")
        return parse_action_type(action_type)

    @staticmethod
    def _safe_action_type(raw: Mapping[str, Any]) -> ActionType:
        action_type = raw.get("action_type") or raw.get("type") or ActionType.UNKNOWN
        try:
            return parse_action_type(action_type)
        except ValueError:
            return ActionType.UNKNOWN
