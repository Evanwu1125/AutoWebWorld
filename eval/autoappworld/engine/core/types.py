from __future__ import annotations

from enum import Enum
from typing import Any


class ActionType(str, Enum):
    UNKNOWN = "unknown"
    CLICK = "click"
    HOVER = "hover"
    TYPE = "type"
    HOTKEY = "hotkey"
    DRAG = "drag"
    SCROLL = "scroll"
    # Selector-based actions
    CLICK_SELECTOR = "click_selector"
    HOVER_SELECTOR = "hover_selector"
    TYPE_SELECTOR = "type_selector"
    SCROLL_SELECTOR = "scroll_selector"
    # UI-Venus specific actions
    LONGPRESS = "longpress"
    PRESS_BACK = "press_back"
    PRESS_HOME = "press_home"
    PRESS_ENTER = "press_enter"
    PRESS_RECENT = "press_recent"


class BackendType(str, Enum):
    PLAYWRIGHT = "playwright"
    PYAUTOGUI = "pyautogui"


def parse_action_type(value: Any) -> ActionType:
    if isinstance(value, ActionType):
        return value
    try:
        return ActionType(str(value))
    except Exception as exc:
        raise ValueError(f"Invalid action_type: {value!r}") from exc


def parse_backend(value: Any) -> BackendType:
    if isinstance(value, BackendType):
        return value
    try:
        return BackendType(str(value))
    except Exception as exc:
        raise ValueError(f"Invalid backend: {value!r}") from exc
