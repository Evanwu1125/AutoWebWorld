import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class FailureLogger:
    def __init__(self, error_log_path: Optional[Path] = None):
        self.failures: List[Dict[str, Any]] = []
        self.total_trajectories = 0
        self.successful = 0
        self.error_log_path = error_log_path

    def add_failure(
        self,
        trajectory_file: str,
        has_item_any: bool,
        llm_detection: Dict[str, Any],
        llm_raw_response: str,
        failure_point: str
    ):
        self.failures.append({
            "trajectory_file": trajectory_file,
            "has_item_any": has_item_any,
            "llm_detection": llm_detection,
            "llm_raw_response": llm_raw_response,
            "failure_point": failure_point
        })

    def increment_total(self):
        self.total_trajectories += 1

    def increment_successful(self):
        self.successful += 1

    def save(self, output_path: Path):
        data = {
            "summary": {
                "total_trajectories": self.total_trajectories,
                "successful": self.successful,
                "failed": len(self.failures)
            },
            "failures": self.failures
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

