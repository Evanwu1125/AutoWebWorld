from __future__ import annotations

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class Step(BaseModel):
    action: Dict[str, Any]
    dom: Optional[Dict[str, Any]] = None
    url: Optional[str] = None
    artifact: Dict[str, Any] = Field(default_factory=dict)
    ts: Optional[Union[int, float, str]] = None


class Trajectory(BaseModel):
    steps: list[Step] = Field(default_factory=list)
    stop_reason: Optional[str] = None


class BaseResult(BaseModel):
    id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Union[int, float, str]

    def trajectory(self) -> Optional[Trajectory]:
        raw = self.data.get("trajectory")
        if raw is None:
            return None
        if isinstance(raw, Trajectory):
            return raw
        return Trajectory.model_validate(raw)

