from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Any, Dict, Optional

from pydantic import Field

from autoappworld.engine import PlaywrightExecutor

from ..core.base import BaseResult, Step, Trajectory
from ..core.environment import EnvironmentConfig
from ..core.environment import Environment
from ..core.runner import Runner
from .utils import (
    extract_port_from_url,
    get_action_type,
    next_action,
    resolve_action_source,
    sample_id,
    start_web_server,
    stop_process,
)

class WebEnvironmentConfig(EnvironmentConfig):
    web_dir: Optional[str] = None
    max_actions: int = 50
    start_timeout: int = 60
    headless: bool = True
    viewport: Optional[tuple[int, int]] = Field(default_factory=lambda: (1280, 720))
    capture: bool = False
    artifact_dir: str = "artifacts"
    executable_path: Optional[str] = None  # Browser executable path
    action_delay: float = 0.1  # Wait time after each action (seconds)


class WebEnvironment(Environment):
    def __init__(
        self,
        name: str,
        web_dir: Optional[str] = None,
        config: Optional[EnvironmentConfig | Dict[str, Any]] = None,
        *,
        max_actions: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, config=config)
        web_config = self._coerce_web_config(self.config)
        self.web_dir = web_dir if web_dir is not None else web_config.web_dir
        self.max_actions = (
            max_actions if max_actions is not None else web_config.max_actions
        )

    def run(
        self,
        sample: Any,
        runner_id: str,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> BaseResult:
        runner = self.get_runner(runner_id)
        return asyncio.run(self._run_async(sample, runner, ctx or {}))

    async def _run_async(
        self,
        sample: Any,
        runner: Runner,
        ctx: Dict[str, Any],
    ) -> BaseResult:
        web_config = self._coerce_web_config(self.config)
        web_dir = _ctx_value(ctx, "web_dir", self.web_dir or web_config.web_dir)
        url = _ctx_value(ctx, "url", None)
        server_process: Optional[subprocess.Popen[str]] = None

        if url is None:
            if not web_dir:
                raise ValueError("web_dir is required when url is not provided")
            server_process, url = start_web_server(
                web_dir=web_dir,
                timeout=_ctx_value(ctx, "start_timeout", web_config.start_timeout),
                manager=None,
                script=None,
            )

        executor = PlaywrightExecutor()
        executable_path = _ctx_value(ctx, "executable_path", web_config.executable_path)
        init_kwargs = {}
        if executable_path:
            init_kwargs["executable_path"] = executable_path
        await executor.init_env(
            url=url,
            headless=_ctx_value(ctx, "headless", web_config.headless),
            viewport=_ctx_value(ctx, "viewport", web_config.viewport),
            **init_kwargs,
        )

        steps: list[Step] = []
        stop_reason = None
        action_source = resolve_action_source(runner, sample, ctx)
        last_result = None

        try:
            max_actions = _ctx_value(ctx, "max_actions", self.max_actions)
            for action_index in range(1, max_actions + 1):
                action = next_action(action_source, sample, ctx, last_result)
                if action is None:
                    stop_reason = "no_action"
                    break

                action_type = get_action_type(action)
                if action_type in {"finish", "done", "stop"}:
                    stop_reason = "finish_action"
                    break

                exec_ctx = dict(ctx)
                exec_ctx.setdefault("capture", web_config.capture)
                exec_ctx.setdefault("artifact_dir", web_config.artifact_dir)
                result = await executor.execute(action, ctx=exec_ctx)
                last_result = result
                action_delay = _ctx_value(ctx, "action_delay", web_config.action_delay)
                time.sleep(action_delay)
                dom = None
                if isinstance(result.meta, dict):
                    dom_info = result.meta.get("dom")
                    if isinstance(dom_info, dict):
                        dom = dom_info

                url_value = None
                if "page" in executor._ctx:
                    url_value = executor._ctx["page"].url

                steps.append(
                    Step(
                        action=dict(action),
                        dom=dom,
                        url=url_value,
                        artifact=dict(result.artifacts),
                        ts=time.time(),
                    )
                )
            else:
                stop_reason = "max_actions"
        finally:
            try:
                await asyncio.shield(executor.close())
            except Exception:
                pass
            if server_process is not None:
                stop_process(server_process)

        result_id = sample_id(sample)
        trajectory = Trajectory(steps=steps, stop_reason=stop_reason)
        port_value = extract_port_from_url(url) if url else None
        return BaseResult(
            id=result_id,
            data={"trajectory": trajectory},
            meta={
                "runner_id": runner.id,
                "web_dir": web_dir,
                "url": url,
                "port": port_value,
                "stop_reason": stop_reason,
            },
            timestamp=time.time(),
        )

    @staticmethod
    def _coerce_web_config(config: EnvironmentConfig) -> WebEnvironmentConfig:
        if isinstance(config, WebEnvironmentConfig):
            return config
        return WebEnvironmentConfig.model_validate(config.to_dict())


def _ctx_value(ctx: Dict[str, Any], key: str, default: Any) -> Any:
    value = ctx.get(key)
    if value is None:
        return default
    return value
