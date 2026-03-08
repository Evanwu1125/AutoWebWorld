from __future__ import annotations

from typing import Any, Dict, Iterable, List


def require_ctx_value(ctx: Dict[str, Any] | None, key: str) -> Any:
    if ctx is None or key not in ctx:
        raise ValueError(f"Missing '{key}' in ctx")
    return ctx[key]




def normalize_pyautogui_keys(keys: Iterable[str]) -> List[str]:
    mapping = {
        "control": "ctrl",
        "ctrl": "ctrl",
        "delete": "delete",
        "backspace": "backspace",
        "enter": "enter",
        "tab": "tab",
        "shift": "shift",
        "alt": "alt",
        "meta": "win",
        "win": "win",
        "command": "command",
        "cmd": "command",
    }
    normalized = []
    for key in keys:
        key_str = str(key).strip()
        if not key_str:
            continue
        normalized.append(mapping.get(key_str.lower(), key_str.lower()))
    return normalized


def normalize_playwright_keys(keys: Iterable[str]) -> str:
    mapping = {
        "control": "Control",
        "ctrl": "Control",
        "delete": "Delete",
        "backspace": "Backspace",
        "enter": "Enter",
        "tab": "Tab",
        "shift": "Shift",
        "alt": "Alt",
        "meta": "Meta",
        "command": "Meta",
        "cmd": "Meta",
    }
    normalized = []
    for key in keys:
        key_str = str(key).strip()
        if not key_str:
            continue
        mapped = mapping.get(key_str.lower())
        normalized.append(mapped or key_str)
    return "+".join(normalized)
