from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

from .base import BaseResult
from .runner import Runner


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EnvironmentConfig":
        return cls.model_validate(data or {})

    @classmethod
    def from_yaml(cls, path: str) -> "EnvironmentConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("EnvironmentConfig YAML must be a mapping")
        return cls.model_validate(raw)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        extra = getattr(self, "__pydantic_extra__", None)
        if extra and key in extra:
            return extra[key]
        return default

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class Environment:
    def __init__(
        self,
        name: str,
        config: Optional[EnvironmentConfig | Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.config = self._normalize_config(config)
        self._runners: Dict[str, Runner] = {}

    @staticmethod
    def _normalize_config(
        config: Optional[EnvironmentConfig | Dict[str, Any]],
    ) -> EnvironmentConfig:
        if config is None:
            return EnvironmentConfig()
        if isinstance(config, EnvironmentConfig):
            return config
        return EnvironmentConfig.model_validate(config)

    def register_runner(self, runner: Runner) -> None:
        self._runners[runner.id] = runner

    def get_runner(self, runner_id: str) -> Runner:
        runner = self._runners.get(runner_id)
        if runner is None:
            raise ValueError(f"Unknown runner_id: {runner_id}")
        return runner

    def run(
        self,
        sample: Any,
        runner_id: str,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> BaseResult:
        runner = self.get_runner(runner_id)
        return runner.run(sample, ctx=ctx)
