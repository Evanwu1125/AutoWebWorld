from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set

from ..core.base import BaseResult
from ..core.metric import Metric


def _extract_selectors(result: BaseResult) -> Set[str]:
    trajectory = result.data.get("trajectory", {})
    steps = trajectory.get("steps", [])
    selectors: Set[str] = set()
    for step in steps:
        dom = step.get("dom") or {}
        selector = dom.get("selector")
        if selector:
            selectors.add(str(selector))
    return selectors


class SelectorCoverageMetric(Metric):
    name = "selector_coverage"

    def compute(
        self,
        sample: Any,
        result: BaseResult,
        reference: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reference = reference or {}
        target_selectors = {str(s) for s in reference.get("selectors", [])}
        observed_selectors = _extract_selectors(result)
        matched = target_selectors & observed_selectors
        missing = sorted(target_selectors - observed_selectors)
        extra = sorted(observed_selectors - target_selectors)
        total = len(target_selectors)
        coverage = len(matched) / total if total else 1.0
        return {
            "coverage": coverage,
            "matched": len(matched),
            "total": total,
            "missing": missing,
            "extra": extra,
        }

    def aggregate(self, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        records_list = list(records)
        if not records_list:
            return {"coverage_mean": 0.0, "samples": 0}
        total_coverage = sum(record.get("coverage", 0.0) for record in records_list)
        return {
            "coverage_mean": total_coverage / len(records_list),
            "samples": len(records_list),
        }
