from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .base import BaseResult


class Runner(ABC):
    id: str

    @abstractmethod
    def run(self, sample: Any, ctx: Optional[Dict[str, Any]] = None) -> BaseResult:
        raise NotImplementedError
