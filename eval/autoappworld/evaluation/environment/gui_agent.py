from __future__ import annotations

import asyncio
import os
import re
import json
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

from pydantic import Field

from autoappworld.engine import PlaywrightExecutor
from autoappworld.agent.gui_agent import (
    USER_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE_VENUS,
    USER_PROMPT_TEMPLATE_TONGUI,
    USER_PROMPT_TEMPLATE_UITARS,
    USER_PROMPT_TEMPLATE_OPENCUA
)

from ..core.base import BaseResult, Step, Trajectory
from ..core.environment import Environment, EnvironmentConfig
from ..core.runner import Runner
from .utils import extract_port_from_url, start_web_server, stop_process
from .ui_venus_parser import normalize_ui_venus_action, parse_ui_venus_action
from .tongui_parser import normalize_tongui_action, parse_tongui_action
from .uitars_parser import normalize_uitars_action, parse_uitars_action
from .opencua_parser import normalize_opencua_action, parse_opencua_action


class GuiAgentEnvironmentConfig(EnvironmentConfig):
    web_dir: Optional[str] = None
    max_steps: int = 30
    start_timeout: int = 60
    load_timeout_ms: int = 5000
    headless: bool = True
    viewport: Optional[tuple[int, int]] = Field(default_factory=lambda: (1280, 720))
    capture: bool = False
    annotate: bool = True
    artifact_dir: str = "artifacts"
    wait_seconds: float = 1.0
    artifact_name_format: str = "{index}_{action}"
    executable_path: Optional[str] = None  # Chrome/Chromium executable path


class GuiAgentEnvironment(Environment):
    def __init__(
        self,
        name: str,
        config: Optional[EnvironmentConfig | Dict[str, Any]] = None,
        *,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        gui_config = self._coerce_config(self.config)
        self.max_steps = max_steps if max_steps is not None else gui_config.max_steps

    @staticmethod
    def _coerce_config(config: EnvironmentConfig) -> GuiAgentEnvironmentConfig:
        if isinstance(config, GuiAgentEnvironmentConfig):
            return config
        return GuiAgentEnvironmentConfig.model_validate(config.to_dict())


class GuiAgentRunner(Runner):
    def __init__(self, env: GuiAgentEnvironment, runner_id: str = "gui_agent") -> None:
        self._env = env
        self.id = runner_id

    def run(self, sample: Any, ctx: Optional[Dict[str, Any]] = None) -> GuiAgentResult:
        return asyncio.run(self._run_async(sample, ctx or {}))

    async def run_async(
        self, sample: Any, ctx: Optional[Dict[str, Any]] = None
    ) -> GuiAgentResult:
        return await self._run_async(sample, ctx or {})

    # Main loop
    async def _run_async(self, sample: Any, ctx: Dict[str, Any]) -> GuiAgentResult:
        gui_config = self._env._coerce_config(self._env.config)
        web_dir = _ctx_value(ctx, "web_dir", gui_config.web_dir)
        url = _ctx_value(ctx, "url", None)
        server_process: Optional[subprocess.Popen[str]] = None

        if url is None:
            if not web_dir:
                raise ValueError("web_dir is required when url is not provided")
            server_process, url = start_web_server(
                web_dir=web_dir,
                timeout=_ctx_value(ctx, "start_timeout", gui_config.start_timeout),
                manager=_ctx_value(ctx, "manager", None),
                script=_ctx_value(ctx, "script", None),
            )

        agent = ctx.get("agent")
        if agent is None:
            raise ValueError("GuiAgent instance is required in ctx['agent']")

        task = _resolve_task(sample, ctx)
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        artifact_root = _ctx_value(ctx, "artifact_dir", gui_config.artifact_dir)
        artifact_run_dir = os.path.join(artifact_root, f"task_{run_stamp}")
        os.makedirs(artifact_run_dir, exist_ok=True)
        input_dir = os.path.join(artifact_run_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        annotation_dir = os.path.join(artifact_run_dir, "annotations")
        os.makedirs(annotation_dir, exist_ok=True)

        executor = PlaywrightExecutor()
        init_kwargs: Dict[str, Any] = {
            "url": url,
            "headless": _ctx_value(ctx, "headless", gui_config.headless),
            "viewport": _ctx_value(ctx, "viewport", gui_config.viewport),
        }
        executable_path = _ctx_value(ctx, "executable_path", gui_config.executable_path)
        if executable_path:
            init_kwargs["executable_path"] = executable_path
        await executor.init_env(**init_kwargs)
        page = executor._ctx.get("page")
        if page is None:
            raise RuntimeError("Playwright page not initialized")

        steps: list[Step] = []
        # 🎯 stop_reason is initialized to None and will be set to one of the following values during execution:
        # - "finish_action": Agent actively calls finish() to complete the task
        # - "max_steps": Maximum step limit reached
        # - "error": Exception occurred during execution (set in run_single_web_task.py)
        # Note: The try-except block here does not catch exceptions to set stop_reason="error"
        # Exceptions propagate up and are caught by the caller (run_single_web_task.py)
        stop_reason = None
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        finish_message: Optional[str] = None
        last_mouse: Optional[Tuple[int, int]] = None

        try:
            # await _wait_for_page_ready(
            #     page,
            #     timeout_ms=_ctx_value(ctx, "load_timeout_ms", gui_config.load_timeout_ms),
            # )
            max_steps = _ctx_value(ctx, "max_steps", self._env.max_steps)
            for step_index in range(1, max_steps + 1):
                # 🔥 Print step separator
                print(f"\n{'='*20} Step {step_index} {'='*20}\n")

                # 🔥 Wait for page to fully load and render
                # _wait_for_page_ready waits for DOM, resource loading, network idle, and checks page content
                await _wait_for_page_ready(
                    page,
                    timeout_ms=_ctx_value(ctx, "load_timeout_ms", gui_config.load_timeout_ms),
                )

                # 🔥 Extra wait time to ensure dynamic content is rendered (configurable via ctx["screenshot_delay"])
                # For complex SPA applications, a longer wait time may be needed
                screenshot_delay = _ctx_value(ctx, "screenshot_delay", 3.0)
                await asyncio.sleep(screenshot_delay)

                # 🔥 Save screenshot to input directory
                input_path = os.path.join(input_dir, f"{step_index:03d}.png")
                await page.screenshot(path=input_path, full_page=False)

                # 🔥 Select the prompt template based on agent_type
                agent_type = _ctx_value(ctx, "agent_type", "gui_agent")
                if agent_type == "ui_venus":
                    prompt_template = USER_PROMPT_TEMPLATE_VENUS
                elif agent_type in ["tongui", "showui"]:
                    prompt_template = USER_PROMPT_TEMPLATE_TONGUI
                elif agent_type == "ui_tars":
                    prompt_template = USER_PROMPT_TEMPLATE_UITARS
                elif agent_type == "open_cua":
                    prompt_template = USER_PROMPT_TEMPLATE_OPENCUA
                else:
                    prompt_template = USER_PROMPT_TEMPLATE

                # 🔥 Build user prompt (history is added only at the last step)
                # We first build the base user_prompt_text; history is added in agent.run_with_response
                user_prompt_text = prompt_template.format(task_prompt=task)

                user_input = {
                    "text": user_prompt_text,
                    "images": [input_path],
                }

                # 🔥 Before calling agent, check if history needs to be added
                # History is automatically appended to the last user message in agent._build_messages()
                action_text, response = await agent.run_with_response(user_input)
                raw_action_text = action_text

                # 🔥 Select the parser based on agent_type
                if agent_type == "ui_venus":
                    normalized_action_text = normalize_ui_venus_action(raw_action_text)
                    action_kind, action_payload, finish_message = parse_ui_venus_action(
                        normalized_action_text
                    )
                elif agent_type in ["tongui","showui"]:
                    normalized_action_text = normalize_tongui_action(raw_action_text)
                    # Get screen size from the screenshot
                    from PIL import Image
                    img = Image.open(input_path)
                    screen_width, screen_height = img.size
                    action_kind, action_payload, finish_message = parse_tongui_action(
                        normalized_action_text, screen_width, screen_height
                    )
                elif agent_type == "ui_tars":
                    normalized_action_text = normalize_uitars_action(raw_action_text)
                    # Get screen size from the screenshot (UI-TARS uses absolute coordinates)
                    from PIL import Image
                    img = Image.open(input_path)
                    screen_width, screen_height = img.size
                    action_kind, action_payload, finish_message = parse_uitars_action(
                        normalized_action_text, screen_width, screen_height
                    )
                elif agent_type == "open_cua":
                    normalized_action_text = normalize_opencua_action(raw_action_text)
                    # Get screen size from the screenshot (OpenCUA-7B uses absolute pixel coordinates)
                    from PIL import Image
                    img = Image.open(input_path)
                    screen_width, screen_height = img.size
                    action_kind, action_payload, finish_message = parse_opencua_action(
                        normalized_action_text, screen_width, screen_height
                    )
                else:
                    normalized_action_text = _normalize_action_text(raw_action_text)
                    action_kind, action_payload, finish_message = _parse_action(
                        normalized_action_text
                    )
                thinking_text = _extract_thinking_text(response)
                step_cost = _extract_cost(response)
                if step_cost is not None:
                    total_cost += step_cost
                input_tokens, output_tokens = _extract_tokens(response)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                step_action = _build_step_action(
                    action_kind,
                    raw_action_text,
                    action_payload,
                    finish_message,
                    response,
                    thinking_text,
                    step_cost,
                    input_tokens,
                    output_tokens,
                )
                if action_kind == "action":
                    last_mouse = _update_last_mouse_from_action(action_payload, last_mouse)

                if action_kind == "finish":
                    # ✅ Agent actively completed the task, setting stop_reason = "finish_action"
                    stop_reason = "finish_action"
                    finish_artifact = await _save_action_screenshot(
                        page=page,
                        enabled=_ctx_value(ctx, "annotate", gui_config.annotate),
                        annotation_dir=annotation_dir,
                        name_format=gui_config.artifact_name_format,
                        index=step_index,
                        action_name="finish",
                    )
                    steps.append(
                        Step(
                            action=step_action,
                            dom=None,
                            url=page.url,
                            artifact=finish_artifact,
                            ts=time.time(),
                        )
                    )
                    await _write_trajectory_output_async(
                        trajectory=Trajectory(steps=steps, stop_reason=stop_reason),
                        artifact_run_dir=artifact_run_dir,
                        meta=_build_meta(
                            runner_id=self.id,
                            web_dir=web_dir,
                            url=url,
                            port_value=extract_port_from_url(url) if url else None,
                            stop_reason=stop_reason,
                            finish_message=finish_message,
                            artifact_run_dir=artifact_run_dir,
                            task=task,
                            model_name=_ctx_value(ctx, "model_name", None),
                            total_cost=total_cost,
                            total_input_tokens=total_input_tokens,
                            total_output_tokens=total_output_tokens,
                        ),
                    )
                    break

                if action_kind == "wait":
                    await asyncio.sleep(_ctx_value(ctx, "wait_seconds", gui_config.wait_seconds))
                    wait_artifact = await _save_action_screenshot(
                        page=page,
                        enabled=_ctx_value(ctx, "annotate", gui_config.annotate),
                        annotation_dir=annotation_dir,
                        name_format=gui_config.artifact_name_format,
                        index=step_index,
                        action_name="wait",
                    )
                    steps.append(
                        Step(
                            action=step_action,
                            dom=None,
                            url=page.url,
                            artifact=wait_artifact,
                            ts=time.time(),
                        )
                    )
                    await _write_trajectory_output_async(
                        trajectory=Trajectory(steps=steps, stop_reason=stop_reason),
                        artifact_run_dir=artifact_run_dir,
                        meta=_build_meta(
                            runner_id=self.id,
                            web_dir=web_dir,
                            url=url,
                            port_value=extract_port_from_url(url) if url else None,
                            stop_reason=stop_reason,
                            finish_message=None,
                            artifact_run_dir=artifact_run_dir,
                            task=task,
                            model_name=_ctx_value(ctx, "model_name", None),
                            total_cost=total_cost,
                            total_input_tokens=total_input_tokens,
                            total_output_tokens=total_output_tokens,
                        ),
                    )
                    continue

                exec_ctx = dict(ctx)
                exec_ctx.setdefault("capture", gui_config.capture)
                exec_ctx.setdefault("annotate", gui_config.annotate)
                exec_ctx.setdefault("artifact_dir", artifact_root)
                exec_ctx.setdefault("artifact_run_dir", annotation_dir)
                exec_ctx.setdefault("artifact_name_format", gui_config.artifact_name_format)
                if last_mouse is not None:
                    exec_ctx["mouse_x"], exec_ctx["mouse_y"] = last_mouse
                await _wait_for_page_ready(
                    page,
                    timeout_ms=_ctx_value(ctx, "load_timeout_ms", gui_config.load_timeout_ms),
                )

                # 🔥 Handle TongUI's click_and_type action
                if action_payload.get("action_type") == "click_and_type":
                    # Execute click first
                    click_payload = {
                        "action_type": "click",
                        "x": action_payload["x"],
                        "y": action_payload["y"]
                    }
                    result = await executor.execute(click_payload, ctx=exec_ctx)
                    # Wait briefly for the input field to gain focus
                    await asyncio.sleep(0.3)
                    # Then execute type
                    type_payload = {
                        "action_type": "type",
                        "text": action_payload["text"]
                    }
                    result = await executor.execute(type_payload, ctx=exec_ctx)
                else:
                    result = await executor.execute(action_payload, ctx=exec_ctx)

                dom = None
                if isinstance(result.meta, dict):
                    dom_info = result.meta.get("dom")
                    if isinstance(dom_info, dict):
                        dom = dom_info
                    mouse_info = result.meta.get("mouse")
                    if isinstance(mouse_info, dict):
                        step_action["mouse"] = mouse_info
                        mx = mouse_info.get("x")
                        my = mouse_info.get("y")
                        if isinstance(mx, (int, float)) and isinstance(my, (int, float)):
                            last_mouse = (int(mx), int(my))

                url_value = None
                if "page" in executor._ctx:
                    url_value = executor._ctx["page"].url

                steps.append(
                    Step(
                        action=step_action,
                        dom=dom,
                        url=url_value,
                        artifact=dict(result.artifacts),
                        ts=time.time(),
                    )
                )
                await _write_trajectory_output_async(
                    trajectory=Trajectory(steps=steps, stop_reason=stop_reason),
                    artifact_run_dir=artifact_run_dir,
                    meta=_build_meta(
                        runner_id=self.id,
                        web_dir=web_dir,
                        url=url,
                        port_value=extract_port_from_url(url) if url else None,
                        stop_reason=stop_reason,
                        finish_message=None,
                        artifact_run_dir=artifact_run_dir,
                        task=task,
                        model_name=_ctx_value(ctx, "model_name", None),
                        total_cost=total_cost,
                        total_input_tokens=total_input_tokens,
                        total_output_tokens=total_output_tokens,
                    ),
                )
            else:
                # ⏱️ Maximum step limit reached, setting stop_reason = "max_steps"
                stop_reason = "max_steps"
        finally:
            try:
                await asyncio.shield(executor.close())
            except Exception:
                pass
            if server_process is not None:
                stop_process(server_process)

        result_id = _resolve_result_id(sample, task)
        trajectory = Trajectory(steps=steps, stop_reason=stop_reason)
        port_value = extract_port_from_url(url) if url else None

        # 🔥 Build GuiAgentResult object for this task execution
        # This is the complete result for a single task, containing three core parts:
        # 1. id: Unique task identifier (e.g. "ArXiv--0")
        # 2. data: Core data containing the full execution trajectory (Trajectory)
        #    - trajectory.steps: Action, URL, DOM, screenshot path for each step
        #    - trajectory.stop_reason: Stop reason ("finish_action" / "max_steps" / "error")
        # 3. meta: Metadata containing execution environment info and statistics
        #    - runner_id: Runner ID
        #    - url: Target website URL
        #    - artifact_run_dir: Output directory path (e.g. "task_20260118_123456/")
        #    - task: Task description
        #    - model_name: LLM model name used
        #    - total_cost: Total API call cost
        #    - total_input_tokens: Total input token count
        #    - total_output_tokens: Total output token count
        # 4. timestamp: Result generation timestamp
        #
        # Return value usage:
        # - Received by run_single_web_task.py and saved as result.json
        # - Contains complete task execution info for subsequent analysis and evaluation
        result = GuiAgentResult(
            id=result_id,
            data={"trajectory": trajectory},
            meta=_build_meta(
                runner_id=self.id,
                web_dir=web_dir,
                url=url,
                port_value=port_value,
                stop_reason=stop_reason,
                finish_message=finish_message if stop_reason == "finish_action" else None,
                artifact_run_dir=artifact_run_dir,
                task=task,
                model_name=_ctx_value(ctx, "model_name", None),
                total_cost=total_cost,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
            ),
            timestamp=time.time(),
        )
        await _write_trajectory_output_async(trajectory, artifact_run_dir, result.meta)
        return result


class GuiAgentResult(BaseResult):
    """
    GUI Agent task execution result class.

    Inherits from BaseResult and contains the following fields:
    - id (str): Unique task identifier, e.g. "ArXiv--0"
    - data (Dict): Core data containing trajectory (Trajectory object)
    - meta (Dict): Metadata containing execution environment info and statistics
    - timestamp (float): Result generation timestamp

    Convenience properties:
    - trajectory(): Get the execution trajectory object
    - total_cost: Total API call cost
    - total_input_tokens: Total input token count
    - total_output_tokens: Total output token count

    Example usage:
        result = GuiAgentResult(
            id="ArXiv--0",
            data={"trajectory": trajectory},
            meta={"total_cost": 0.015, ...},
            timestamp=time.time()
        )
        print(result.total_cost)  # 0.015
        print(len(result.trajectory().steps))  # Number of steps
    """

    def trajectory(self) -> Trajectory:
        """Get the task execution trajectory object."""
        raw = self.data.get("trajectory")
        if raw is None:
            return Trajectory()
        if isinstance(raw, Trajectory):
            return raw
        return Trajectory.model_validate(raw)

    @property
    def total_cost(self) -> float:
        """Total API call cost (USD)."""
        value = self.meta.get("total_cost")
        return float(value) if isinstance(value, (int, float)) else 0.0

    @property
    def total_input_tokens(self) -> int:
        """Total input token count."""
        value = self.meta.get("total_input_tokens")
        return int(value) if isinstance(value, (int, float)) else 0

    @property
    def total_output_tokens(self) -> int:
        """Total output token count."""
        value = self.meta.get("total_output_tokens")
        return int(value) if isinstance(value, (int, float)) else 0


def _resolve_task(sample: Any, ctx: Dict[str, Any]) -> str:
    if "task" in ctx and ctx["task"]:
        return str(ctx["task"])
    if isinstance(sample, dict) and sample.get("task"):
        return str(sample["task"])
    if isinstance(sample, str) and sample:
        return sample
    return "complete the task"


def _resolve_result_id(sample: Any, task: str) -> str:
    if isinstance(sample, dict):
        if "id" in sample:
            return str(sample["id"])
    return task[:50].replace("\n", " ").strip() or "run"


def _normalize_action_text(action_text: str) -> str:
    text = action_text.strip()
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _parse_action(action_text: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Parse action from LLM response.

    Supports two formats:
    1. New JSON format: {'action': 'click', 'coordinate': [x, y]}
    2. Old text format: click (x, y)
    """
    text = action_text.strip()
    lowered = text.lower()

    # ============================================================================
    # NEW FORMAT: Flexible keyword-based parsing
    # ============================================================================
    # Use keyword detection and regex extraction for more flexible parsing

    # Check for 'wait' keyword (must check before 'click' to avoid false positives)
    if 'wait' in lowered and 'click' not in lowered:
        # wait action - just return wait, ignore any time parameter
        return "wait", {}, None

    # Check for 'click' keyword - extract coordinates from [x, y]
    if 'click' in lowered:
        coord_match = re.search(r'\[(\d+),\s*(\d+)\]', text)
        if coord_match:
            return "action", {"action_type": "click", "x": int(coord_match.group(1)), "y": int(coord_match.group(2))}, None

    # Check for 'hover' keyword - extract coordinates from [x, y]
    if 'hover' in lowered:
        coord_match = re.search(r'\[(\d+),\s*(\d+)\]', text)
        if coord_match:
            return "action", {"action_type": "hover", "x": int(coord_match.group(1)), "y": int(coord_match.group(2))}, None

    # Check for 'drag' keyword - extract two coordinate pairs [x1, y1] and [x2, y2]
    if 'drag' in lowered:
        coord_matches = re.findall(r'\[(\d+),\s*(\d+)\]', text)
        if len(coord_matches) >= 2:
            return "action", {
                "action_type": "drag",
                "x1": int(coord_matches[0][0]),
                "y1": int(coord_matches[0][1]),
                "x2": int(coord_matches[1][0]),
                "y2": int(coord_matches[1][1]),
            }, None

    # Check for 'type_text' keyword - extract text after 'text'
    if 'type_text' in lowered or 'type text' in lowered:
        # Try to extract from JSON-like format: 'text': 'content' or "text": "content"
        text_match = re.search(r"['\"]text['\"]:\s*['\"](.+?)['\"]", text, re.DOTALL)
        if text_match:
            return "action", {"action_type": "type", "text": text_match.group(1)}, None
        # Fallback: extract everything after 'text'
        text_match = re.search(r"text['\"]?\s*:\s*['\"]?(.+)", text, re.IGNORECASE)
        if text_match:
            content = text_match.group(1).strip().strip("'\"")
            return "action", {"action_type": "type", "text": content}, None

    # Check for 'press_enter' keyword
    if 'press_enter' in lowered:
        return "action", {"action_type": "hotkey", "keys": ["Enter"]}, None

    # Check for 'scroll' keyword - extract direction (down/up)
    if 'scroll' in lowered:
        # Try to find 'down' or 'up' in the text
        if 'down' in lowered:
            return "action", {"action_type": "scroll", "amount": -500}, None
        elif 'up' in lowered:
            return "action", {"action_type": "scroll", "amount": 500}, None
        else:
            # Default to scroll down
            return "action", {"action_type": "scroll", "amount": -500}, None

    # Check for 'answer' keyword - extract text after 'text'
    if 'answer' in lowered:
        # Try to extract from JSON-like format: 'text': 'content' or "text": "content"
        text_match = re.search(r"['\"]text['\"]:\s*['\"](.+?)['\"](?:\s*\})?$", text, re.DOTALL)
        if text_match:
            return "finish", {}, text_match.group(1)
        # Fallback: extract everything after 'text'
        text_match = re.search(r"text['\"]?\s*:\s*['\"]?(.+)", text, re.IGNORECASE)
        if text_match:
            content = text_match.group(1).strip().strip("'\"}")
            return "finish", {}, content

    # ============================================================================
    # OLD FORMAT: Text-based actions (kept for backward compatibility)
    # ============================================================================
    if lowered == "wait":
        return "wait", {}, None

    if lowered.startswith("finish"):
        finish_match = re.match(r"^finish\s*:\s*(.*)$", text, re.IGNORECASE)
        return "finish", {}, (finish_match.group(1).strip() if finish_match else "")

    click = re.match(r"^click\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$", text, re.IGNORECASE)
    if click:
        return "action", {"action_type": "click", "x": int(click.group(1)), "y": int(click.group(2))}, None

    hover = re.match(r"^hover\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$", text, re.IGNORECASE)
    if hover:
        return "action", {"action_type": "hover", "x": int(hover.group(1)), "y": int(hover.group(2))}, None

    drag = re.match(
        r"^drag from\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*to\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$",
        text,
        re.IGNORECASE,
    )
    if drag:
        return (
            "action",
            {
                "action_type": "drag",
                "x1": int(drag.group(1)),
                "y1": int(drag.group(2)),
                "x2": int(drag.group(3)),
                "y2": int(drag.group(4)),
            },
            None,
        )

    scroll = re.match(r"^scroll\s*\(\s*(-?\d+)\s*\)\s*$", text, re.IGNORECASE)
    if scroll:
        return "action", {"action_type": "scroll", "amount": int(scroll.group(1))}, None

    type_text = re.match(r"^type text\s*:\s*(.*)$", text, re.IGNORECASE)
    if type_text:
        return "action", {"action_type": "type", "text": type_text.group(1)}, None

    press_key = re.match(r"^press key\s*:\s*(.*)$", text, re.IGNORECASE)
    if press_key:
        key = press_key.group(1).strip()
        return "action", {"action_type": "hotkey", "keys": [key]}, None

    hotkey = re.match(r"^hotkey\s*\((.*)\)\s*$", text, re.IGNORECASE)
    if hotkey:
        keys = [k.strip() for k in hotkey.group(1).split(",") if k.strip()]
        if not keys:
            raise ValueError(f"Invalid hotkey keys in action: {text}")
        return "action", {"action_type": "hotkey", "keys": keys}, None

    raise ValueError(f"Unrecognized action: {action_text}")


def _build_step_action(
    action_kind: str,
    action_text: str,
    action_payload: Dict[str, Any],
    finish_message: Optional[str],
    response: Any,
    thinking_text: str,
    step_cost: Optional[float],
    input_tokens: int,
    output_tokens: int,
) -> Dict[str, Any]:
    action: Dict[str, Any] = {
        "action_kind": action_kind,
        "raw_action": action_text,
        "parsed_action": dict(action_payload),
        "model_response": getattr(response, "content", ""),
        "thinking": thinking_text,
        "cost": step_cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if action_kind == "finish":
        action["action_type"] = "finish"
        action["finish_message"] = finish_message or ""
    elif action_kind == "wait":
        action["action_type"] = "wait"
    else:
        action.setdefault("action_type", action_payload.get("action_type", "action"))
    return action


def _ctx_value(ctx: Dict[str, Any], key: str, default: Any) -> Any:
    value = ctx.get(key)
    if value is None:
        return default
    return value


def _update_last_mouse_from_action(
    action_payload: Dict[str, Any],
    last_mouse: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    action_type = str(action_payload.get("action_type", "")).lower()
    if action_type in {"click", "hover"}:
        x = action_payload.get("x")
        y = action_payload.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return (int(x), int(y))
    if action_type == "drag":
        x2 = action_payload.get("x2")
        y2 = action_payload.get("y2")
        if isinstance(x2, (int, float)) and isinstance(y2, (int, float)):
            return (int(x2), int(y2))
    return last_mouse


def _build_meta(
    *,
    runner_id: str,
    web_dir: Optional[str],
    url: Optional[str],
    port_value: Optional[int],
    stop_reason: Optional[str],
    finish_message: Optional[str],
    artifact_run_dir: str,
    task: str,
    model_name: Optional[str],
    total_cost: float,
    total_input_tokens: int,
    total_output_tokens: int,
) -> Dict[str, Any]:
    return {
        "runner_id": runner_id,
        "web_dir": web_dir,
        "url": url,
        "port": port_value,
        "stop_reason": stop_reason,
        "finish_message": finish_message,
        "artifact_run_dir": artifact_run_dir,
        "task": task,
        "model": model_name,
        "total_cost": total_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


async def _write_trajectory_output_async(
    trajectory: Trajectory,
    artifact_run_dir: str,
    meta: Dict[str, Any],
) -> None:
    await asyncio.to_thread(_write_trajectory_output, trajectory, artifact_run_dir, meta)


def _write_trajectory_output(
    trajectory: Trajectory,
    artifact_run_dir: str,
    meta: Dict[str, Any],
) -> None:
    output = {
        "meta": meta,
        "stop_reason": trajectory.stop_reason,
        "steps": [
            {
                "index": idx + 1,  # 🔥 Add index field, starting from 1
                "action": step.action,
                "url": step.url,
                # "dom": step.dom,  # 🔥 DOM info commented out
                "artifact": step.artifact,
                "ts": step.ts,
            }
            for idx, step in enumerate(trajectory.steps)  # 🔥 Use enumerate to get index
        ],
    }
    os.makedirs(artifact_run_dir, exist_ok=True)
    path = os.path.join(artifact_run_dir, "trajectory_output.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=True, indent=2)


def _extract_thinking_text(response: Any) -> str:
    """
    Extract thinking/reasoning text from LLM response.

    Supports both formats:
    - New format: <think>...</think>
    - Old format: <reason>...</reason>
    """
    content = getattr(response, "content", "") or ""

    # Try new format first (<think>...</think>)
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback to old format (<reason>...</reason>)
    match = re.search(r"<reason>(.*?)</reason>", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return ""


def _extract_cost(response: Any) -> Optional[float]:
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            return float(cost)
    return None


def _extract_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            return input_tokens, output_tokens
        if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
            return int(input_tokens), int(output_tokens)
    return 0, 0


async def _wait_for_page_ready(page: Any, timeout_ms: int) -> None:
    """
    Wait for the page to fully load and render.

    Enhanced version:
    1. Wait for DOM to load
    2. Wait for all resources to load
    3. Wait for network idle
    4. Extra wait to ensure rendering is complete
    """
    try:
        # Wait for DOM content to load
        await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)

        # Wait for all resources (images, stylesheets, etc.) to load
        await page.wait_for_load_state("load", timeout=timeout_ms)

        # Wait for network idle (no network requests for at least 500ms)
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception as e:
            # networkidle may time out, but this does not affect subsequent steps
            print(f"⚠️ networkidle timeout: {e}")

        # 🔥 Extra brief wait to ensure JavaScript rendering is complete
        # Frameworks like React/Vue may need additional time to render after DOM loads
        await asyncio.sleep(0.5)

        # 🔥 Wait until the page has at least some visible content (avoid blank screen)
        try:
            await page.wait_for_function(
                """
                () => {
                    // Check whether body has content
                    const body = document.body;
                    if (!body) return false;

                    // Check whether there are visible elements
                    const hasVisibleContent = body.offsetHeight > 0 && body.offsetWidth > 0;

                    // Check whether there is text content or images
                    const hasText = body.innerText && body.innerText.trim().length > 0;
                    const hasImages = document.querySelectorAll('img').length > 0;

                    return hasVisibleContent && (hasText || hasImages);
                }
                """,
                timeout=5000  # Wait at most 5 seconds
            )
        except Exception as e:
            # If the check fails, continue anyway (may be a special page)
            print(f"⚠️ Page content check failed: {e}")

    except Exception as e:
        # Log other exceptions for easier debugging
        print(f"⚠️ _wait_for_page_ready error: {e}")
        return


async def _save_action_screenshot(
    *,
    page: Any,
    enabled: bool,
    annotation_dir: str,
    name_format: str,
    index: int,
    action_name: str,
) -> Dict[str, str]:
    if not enabled:
        return {}
    os.makedirs(annotation_dir, exist_ok=True)
    name = name_format.format(index=index, action=action_name)
    path = os.path.join(annotation_dir, f"{name}.png")
    await page.screenshot(path=path, full_page=False)
    return {"annotated": path}
