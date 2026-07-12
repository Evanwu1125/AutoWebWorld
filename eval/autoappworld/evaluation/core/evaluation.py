from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .base import BaseResult
from .environment import Environment
from .metric import Metric


def _get_sample_id(sample: Any, fallback: str) -> str:
    if isinstance(sample, dict) and "id" in sample:
        return str(sample["id"])
    if hasattr(sample, "id"):
        return str(getattr(sample, "id"))
    return fallback


class Evaluation:
    def __init__(self, dataset: Iterable[Any], metrics: List[Metric]) -> None:
        self.dataset = list(dataset)
        self.metrics = metrics

    def run(
        self,
        env: Environment,
        runner_id: str,
        references: Optional[Dict[str, Dict[str, Any]]] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        references = references or {}
        records_by_metric: Dict[str, List[Dict[str, Any]]] = {
            metric.name: [] for metric in self.metrics
        }
        samples_output: List[Dict[str, Any]] = []
        for idx, sample in enumerate(self.dataset):
            sample_id = _get_sample_id(sample, str(idx))
            result = env.run(sample, runner_id=runner_id, ctx=ctx)
            reference = references.get(sample_id)
            metric_outputs: Dict[str, Dict[str, Any]] = {}
            for metric in self.metrics:
                record = metric.compute(sample, result, reference)
                metric_outputs[metric.name] = record
                records_by_metric[metric.name].append(record)
            samples_output.append(
                {
                    "id": sample_id,
                    "result": result,
                    "metrics": metric_outputs,
                }
            )
        summary = {
            metric.name: metric.aggregate(records_by_metric[metric.name])
            for metric in self.metrics
        }
        return {
            "runner_id": runner_id,
            "summary": summary,
            "samples": samples_output,
        }
