from __future__ import annotations

from typing import Any, Dict, Optional

import os
import time
from PIL import Image, ImageDraw

from ..actions.base import ActionResult
from ..core.middleware import AsyncActionMiddleware
from ..actions.spec._utils import get_drag_points, get_xy
from ..core.types import ActionType, parse_action_type


class ActionIndexMiddleware(AsyncActionMiddleware):
    def __init__(
        self,
        *,
        name_format: str = "{index}_{action}",
    ) -> None:
        self._index = 0
        self._name_format = name_format

    async def before_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._index += 1
        ctx.setdefault("action_index", self._index)
        action_type = raw.get("action_type") or raw.get("type")
        if action_type is not None:
            ctx.setdefault("action_type", str(action_type))
        ctx.setdefault("artifact_name_format", self._name_format)
        return ctx

    async def after_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
        result: ActionResult,
    ) -> ActionResult:
        return result


class MousePositionMiddleware(AsyncActionMiddleware):
    async def before_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            action_type = parse_action_type(raw.get("action_type") or raw.get("type"))
        except Exception:
            return ctx

        if action_type in {ActionType.CLICK, ActionType.HOVER}:
            x, y = get_xy(raw)
            ctx["mouse_x"] = int(x)
            ctx["mouse_y"] = int(y)
        elif action_type is ActionType.DRAG:
            _, _, x2, y2 = get_drag_points(raw)
            ctx["mouse_x"] = int(x2)
            ctx["mouse_y"] = int(y2)
        elif action_type in {ActionType.SCROLL, ActionType.TYPE, ActionType.HOTKEY}:
            page = ctx.get("page")
            mouse_x = ctx.get("mouse_x")
            mouse_y = ctx.get("mouse_y")
            if page is not None and isinstance(mouse_x, int) and isinstance(mouse_y, int):
                await page.mouse.move(mouse_x, mouse_y)
        return ctx

    async def after_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
        result: ActionResult,
    ) -> ActionResult:
        return result


class ActionMetaMiddleware(AsyncActionMiddleware):
    async def before_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        return ctx

    async def after_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
        result: ActionResult,
    ) -> ActionResult:
        if "action_index" in ctx:
            result.meta.setdefault("action_index", ctx["action_index"])
        if "action_type" in ctx:
            result.meta.setdefault("action_type", ctx["action_type"])
        mouse_x = ctx.get("mouse_x")
        mouse_y = ctx.get("mouse_y")
        if isinstance(mouse_x, int) and isinstance(mouse_y, int):
            result.meta.setdefault("mouse", {"x": mouse_x, "y": mouse_y})
        return result


class CaptureMiddleware(AsyncActionMiddleware):
    async def before_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        # 🔥 Disable before/after directory creation
        # if not ctx.get("capture"):
        #     return ctx
        # page = ctx.get("page")
        # if page is None:
        #     return ctx
        # try:
        #     action_type = parse_action_type(raw.get("action_type") or raw.get("type"))
        # except Exception:
        #     return ctx

        # marker = _build_marker(action_type, raw)
        # if marker is None:
        #     return ctx

        # from ..actions.handler.playwright_async._artifacts import (
        #     capture_before,
        #     resolve_basename,
        # )

        # basename = resolve_basename(action_type, ctx)
        # ctx["__capture_basename"] = basename
        # ctx["__capture_marker"] = marker
        # ctx["__capture_before"] = await capture_before(
        #     page,
        #     marker,
        #     ctx.get("artifact_dir", "artifacts"),
        #     basename,
        # )
        return ctx

    async def after_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
        result: ActionResult,
    ) -> ActionResult:
        # 🔥 Disable before/after directory creation
        # if not ctx.get("capture"):
        #     return result
        # page = ctx.get("page")
        # basename = ctx.get("__capture_basename")
        # if page is None or basename is None:
        #     return result

        # from ..actions.handler.playwright_async._artifacts import capture_after

        # after_path = await capture_after(
        #     page,
        #     ctx.get("artifact_dir", "artifacts"),
        #     basename,
        # )
        # if ctx.get("__capture_before"):
        #     result.artifacts.setdefault("before", ctx["__capture_before"])
        # result.artifacts.setdefault("after", after_path)
        return result


class AnnotateMiddleware(AsyncActionMiddleware):
    async def before_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not ctx.get("annotate"):
            return ctx
        page = ctx.get("page")
        if page is None:
            return ctx
        try:
            action_type = parse_action_type(raw.get("action_type") or raw.get("type"))
        except Exception:
            return ctx

        base_dir = ctx.get("artifact_run_dir") or ctx.get("artifact_dir") or "artifacts"
        os.makedirs(base_dir, exist_ok=True)
        index = ctx.get("action_index", 0)
        fmt = ctx.get("artifact_name_format", "{index}_{action}")
        name = fmt.format(index=index, action=action_type.value)
        final_path = os.path.join(base_dir, f"{name}.png")
        temp_path = os.path.join(base_dir, f"_{name}_raw_{int(time.time() * 1000)}.png")

        await page.screenshot(path=temp_path, full_page=False)
        if action_type in {ActionType.CLICK, ActionType.HOVER, ActionType.DRAG}:
            _annotate_image(temp_path, final_path, action_type, raw)
        else:
            mouse_x = ctx.get("mouse_x")
            mouse_y = ctx.get("mouse_y")
            if isinstance(mouse_x, int) and isinstance(mouse_y, int):
                _annotate_point(temp_path, final_path, mouse_x, mouse_y)
            else:
                os.replace(temp_path, final_path)
        ctx["__annotate_path"] = final_path
        return ctx

    async def after_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
        result: ActionResult,
    ) -> ActionResult:
        annotate_path = ctx.get("__annotate_path")
        if annotate_path:
            result.artifacts.setdefault("annotated", annotate_path)
        return result


class DomCaptureMiddleware(AsyncActionMiddleware):
    async def before_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        page = ctx.get("page")
        if page is None:
            return ctx
        try:
            action_type = parse_action_type(raw.get("action_type") or raw.get("type"))
        except Exception:
            return ctx

        dom = await _capture_dom(page, action_type, raw)
        if dom is None:
            mouse_x = ctx.get("mouse_x")
            mouse_y = ctx.get("mouse_y")
            if isinstance(mouse_x, int) and isinstance(mouse_y, int):
                dom = await _capture_dom_point(page, mouse_x, mouse_y)
        if dom is not None:
            ctx["__dom"] = dom
        return ctx

    async def after_execute(
        self,
        raw: Dict[str, Any],
        ctx: Dict[str, Any],
        result: ActionResult,
    ) -> ActionResult:
        dom = ctx.get("__dom")
        if dom is not None:
            result.meta.setdefault("dom", dom)
        return result


def _build_marker(action_type: ActionType, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if action_type in {ActionType.CLICK, ActionType.HOVER}:
        x, y = get_xy(raw)
        return {"type": "point", "x": x, "y": y}
    if action_type is ActionType.DRAG:
        x1, y1, x2, y2 = get_drag_points(raw)
        return {"type": "rect", "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return None


def _annotate_image(
    src_path: str,
    dst_path: str,
    action_type: ActionType,
    raw: Dict[str, Any],
) -> None:
    with Image.open(src_path) as img:
        draw = ImageDraw.Draw(img)
        if action_type in {ActionType.CLICK, ActionType.HOVER}:
            x, y = get_xy(raw)
            r = 10
            draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=3)
            draw.line((x - r * 2, y, x + r * 2, y), fill="red", width=2)
            draw.line((x, y - r * 2, x, y + r * 2), fill="red", width=2)
        elif action_type is ActionType.DRAG:
            x1, y1, x2, y2 = get_drag_points(raw)
            draw.line((x1, y1, x2, y2), fill="red", width=3)
            r = 8
            draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline="red", width=3)
            draw.ellipse((x2 - r, y2 - r, x2 + r, y2 + r), outline="red", width=3)

        img.save(dst_path)

    if src_path != dst_path and os.path.exists(src_path):
        os.remove(src_path)


def _annotate_point(
    src_path: str,
    dst_path: str,
    x: int,
    y: int,
) -> None:
    with Image.open(src_path) as img:
        draw = ImageDraw.Draw(img)
        r = 10
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=3)
        draw.line((x - r * 2, y, x + r * 2, y), fill="red", width=2)
        draw.line((x, y - r * 2, x, y + r * 2), fill="red", width=2)
        img.save(dst_path)

    if src_path != dst_path and os.path.exists(src_path):
        os.remove(src_path)


async def _capture_dom_point(
    page: Any,
    x: int,
    y: int,
) -> Optional[Dict[str, Any]]:
    return await page.evaluate(
        """
        ({x, y}) => {
          const el = document.elementFromPoint(x, y);
          if (!el) return null;
          return {
            tag: el.tagName,
            id: el.id || null,
            className: el.className || null,
            text: el.innerText || null,
            ariaLabel: el.getAttribute("aria-label"),
          };
        }
        """,
        {"x": x, "y": y},
    )


async def _capture_dom(
    page: Any,
    action_type: ActionType,
    raw: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if action_type in {ActionType.CLICK, ActionType.HOVER}:
        x, y = get_xy(raw)
        return await page.evaluate(
            """
            ({x, y}) => {
              const el = document.elementFromPoint(x, y);
              if (!el) return null;
              return {
                tag: el.tagName,
                id: el.id || null,
                className: el.className || null,
                text: el.innerText || null,
                ariaLabel: el.getAttribute("aria-label"),
              };
            }
            """,
            {"x": x, "y": y},
        )
    if action_type is ActionType.DRAG:
        x1, y1, x2, y2 = get_drag_points(raw)
        start_element = await page.evaluate(
            """
            ({x, y}) => {
              const el = document.elementFromPoint(x, y);
              if (!el) return null;
              return {
                tag: el.tagName,
                id: el.id || null,
                className: el.className || null,
                text: el.innerText || null,
                ariaLabel: el.getAttribute("aria-label"),
              };
            }
            """,
            {"x": x1, "y": y1},
        )
        end_element = await page.evaluate(
            """
            ({x, y}) => {
              const el = document.elementFromPoint(x, y);
              if (!el) return null;
              return {
                tag: el.tagName,
                id: el.id || null,
                className: el.className || null,
                text: el.innerText || null,
                ariaLabel: el.getAttribute("aria-label"),
              };
            }
            """,
            {"x": x2, "y": y2},
        )
        return {
            "start_element": start_element,
            "end_element": end_element,
        }
    return None
