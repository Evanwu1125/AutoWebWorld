from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageDraw

from ....core.types import ActionType

def build_basename(action_type: ActionType, action_index: Optional[int] = None) -> str:
    if action_index is not None:
        return f"{action_index}_{action_type.value}"
    return f"{action_type.value}_{int(time.time() * 1000)}"


def resolve_basename(
    action_type: ActionType,
    ctx: Optional[Dict[str, Any]],
) -> str:
    action_index = None
    name_format = None
    if ctx:
        action_index = ctx.get("action_index")
        name_format = ctx.get("artifact_name_format")

    if name_format:
        safe_index = "" if action_index is None else str(action_index)
        return str(name_format).format(index=safe_index, action=action_type.value)

    return build_basename(action_type, action_index)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _draw_marker(path: str, marker: Dict[str, int]) -> None:
    with Image.open(path) as image:
        draw = ImageDraw.Draw(image)
        if marker.get("type") == "point":
            x = int(marker["x"])
            y = int(marker["y"])
            radius = 6
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline="red",
                width=2,
            )
            draw.ellipse(
                (x - radius + 2, y - radius + 2, x + radius - 2, y + radius - 2),
                fill=(255, 0, 0, 64),
            )
        elif marker.get("type") == "rect":
            x1 = int(marker["x1"])
            y1 = int(marker["y1"])
            x2 = int(marker["x2"])
            y2 = int(marker["y2"])
            left = min(x1, x2)
            right = max(x1, x2)
            top = min(y1, y2)
            bottom = max(y1, y2)
            draw.rectangle((left, top, right, bottom), outline="red", width=2)
        image.save(path)


async def capture_before(
    page: Any,
    marker: Optional[Dict[str, int]],
    artifact_dir: str,
    basename: str,
) -> str:
    before_dir = str(Path(artifact_dir) / "before")
    ensure_dir(before_dir)
    before_path = str(Path(before_dir) / f"{basename}.png")
    await page.screenshot(path=before_path)
    if marker:
        _draw_marker(before_path, marker)
    return before_path


async def capture_after(page: Any, artifact_dir: str, basename: str) -> str:
    after_dir = str(Path(artifact_dir) / "after")
    ensure_dir(after_dir)
    after_path = str(Path(after_dir) / f"{basename}.png")
    await page.screenshot(path=after_path)
    return after_path
