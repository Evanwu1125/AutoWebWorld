from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

from ...core.types import ActionType, parse_action_type


def get_action_type(raw: Mapping[str, Any]) -> ActionType:
    action_type = raw.get("action_type") or raw.get("type")
    if not action_type:
        raise ValueError("Missing action_type in raw action")
    return parse_action_type(action_type)


def _as_pair(value: Any) -> Optional[Tuple[int, int]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _as_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {field}: {value!r}")


def get_xy(raw: Mapping[str, Any]) -> Tuple[int, int]:
    pair = _as_pair(raw.get("position") or raw.get("pos") or raw.get("point"))
    if pair is not None:
        return pair
    if "x" in raw and "y" in raw:
        return _as_int(raw["x"], "x"), _as_int(raw["y"], "y")
    args = raw.get("args")
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return _as_int(args[0], "x"), _as_int(args[1], "y")
    raise ValueError("Missing x/y for action")


def get_drag_points(raw: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    from_pair = _as_pair(raw.get("from"))
    to_pair = _as_pair(raw.get("to"))
    if from_pair is not None and to_pair is not None:
        return from_pair[0], from_pair[1], to_pair[0], to_pair[1]
    if all(k in raw for k in ("x1", "y1", "x2", "y2")):
        return (
            _as_int(raw["x1"], "x1"),
            _as_int(raw["y1"], "y1"),
            _as_int(raw["x2"], "x2"),
            _as_int(raw["y2"], "y2"),
        )
    args = raw.get("args")
    if isinstance(args, (list, tuple)) and len(args) >= 4:
        return (
            _as_int(args[0], "x1"),
            _as_int(args[1], "y1"),
            _as_int(args[2], "x2"),
            _as_int(args[3], "y2"),
        )
    raise ValueError("Missing drag coordinates")


def get_text(raw: Mapping[str, Any]) -> str:
    for key in ("text", "content", "text_content", "value"):
        if key in raw:
            return str(raw[key])
    args = raw.get("args")
    if isinstance(args, (list, tuple)) and args:
        return str(args[0])
    raise ValueError("Missing text content")


def get_hotkey(raw: Mapping[str, Any]) -> Sequence[str]:
    keys = raw.get("keys")
    if isinstance(keys, (list, tuple)) and keys:
        return [str(k) for k in keys]
    hotkey = raw.get("hotkey")
    if isinstance(hotkey, str) and hotkey.strip():
        for sep in ("+", ",", " "):
            if sep in hotkey:
                return [k.strip() for k in hotkey.split(sep) if k.strip()]
        return [hotkey.strip()]
    args = raw.get("args")
    if isinstance(args, (list, tuple)) and args:
        return [str(k) for k in args]
    raise ValueError("Missing hotkey keys")


def get_scroll(raw: Mapping[str, Any]) -> int:
    for key in ("amount", "x", "delta", "scroll"):
        if key in raw:
            return _as_int(raw[key], key)
    args = raw.get("args")
    if isinstance(args, (list, tuple)) and args:
        return _as_int(args[0], "amount")
    raise ValueError("Missing scroll amount")
