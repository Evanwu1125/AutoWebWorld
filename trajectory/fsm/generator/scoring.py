from __future__ import annotations

from typing import Any, Dict, List


UNREACHABLE_TERMINAL_WEIGHT = 100
ORPHAN_PAGE_WEIGHT = 50


def _to_int_score(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def apply_reachability_score(
    base_report: Dict[str, Any],
    local_reachability_issues: List[Dict[str, Any]],
    reachability: Dict[str, Any],
) -> Dict[str, Any]:
    report = dict(base_report) if isinstance(base_report, dict) else {}

    base_issues = report.get("issues", [])
    if not isinstance(base_issues, list):
        base_issues = []

    bfs_issues = local_reachability_issues if isinstance(local_reachability_issues, list) else []
    for idx, issue in enumerate(bfs_issues, start=1):
        if isinstance(issue, dict) and not issue.get("id"):
            issue["id"] = f"BFS{idx:03d}"

    unreachable_count = sum(
        1 for issue in bfs_issues if isinstance(issue, dict) and issue.get("type") == "unreachable_terminal"
    )
    orphan_count = sum(
        1 for issue in bfs_issues if isinstance(issue, dict) and issue.get("type") == "orphan_page"
    )
    penalty = (
        unreachable_count * UNREACHABLE_TERMINAL_WEIGHT
        + orphan_count * ORPHAN_PAGE_WEIGHT
    )

    llm_score = _to_int_score(report.get("score", 0), default=0)
    final_score = max(0, min(100, llm_score - penalty))

    report["issues"] = base_issues + bfs_issues
    report["reachability"] = reachability if isinstance(reachability, dict) else {}
    report["score"] = final_score
    report["verdict"] = "pass" if final_score == 100 else "fail"
    return report
