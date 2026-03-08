from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

from .base import BaseResult


class Metric(ABC):
    name: str

    @abstractmethod
    def compute(
        self,
        sample: Any,
        result: BaseResult,
        reference: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError
