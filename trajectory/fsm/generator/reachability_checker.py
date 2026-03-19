from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Set, Tuple

from trajectory.bfs.bfs_action import (
    default_signature_for_page,
    has_pending_form_actions,
    has_unhandled_interceptors,
    is_captcha_action,
    is_interceptor_action,
    preconditions_ok,
    state_key,
    transition,
)
from trajectory.bfs.normalize import normalize_fsm


def _extract_page_ids(fsm_data: Dict[str, Any]) -> List[str]:
    page_ids: List[str] = []
    for page in fsm_data.get("pages", []):
        if isinstance(page, dict):
            page_id = page.get("id")
            if isinstance(page_id, str):
                page_ids.append(page_id)
    return page_ids


def _extract_terminal_ids(norm_meta: Dict[str, Any]) -> List[str]:
    terminals: List[str] = []
    for terminal_id in norm_meta.get("terminal_pages", []):
        if isinstance(terminal_id, str):
            terminals.append(terminal_id)
    return terminals


def _run_semantic_bfs(
    fsm_data: Dict[str, Any],
    schema_lookup: Dict[str, Dict[str, str]],
    initial_page_id: str,
    terminal_ids: Set[str],
    max_depth: int,
) -> Tuple[Set[str], Dict[str, int]]:
    page_index = {
        page.get("id"): page
        for page in fsm_data.get("pages", [])
        if isinstance(page, dict) and isinstance(page.get("id"), str)
    }
    if initial_page_id not in page_index:
        return set(), {}

    start_sig = default_signature_for_page(initial_page_id, schema_lookup)
    start_state = (initial_page_id, start_sig, 0)

    reachable_page_ids: Set[str] = {initial_page_id}
    terminal_shortest_hops: Dict[str, int] = {}
    if initial_page_id in terminal_ids:
        terminal_shortest_hops[initial_page_id] = 0

    queue = deque([start_state])
    visited_state_keys = {state_key(initial_page_id, start_sig)}

    while queue:
        page_id, signature, depth = queue.popleft()
        if depth >= max_depth:
            continue

        actions = page_index.get(page_id, {}).get("actions", [])
        for action in actions:
            if not isinstance(action, dict):
                continue
            if not preconditions_ok(signature, action.get("preconditions")):
                continue

            page_schema = schema_lookup.get(page_id, {})
            if has_unhandled_interceptors(signature, page_schema) and not is_interceptor_action(action):
                continue
            if is_captcha_action(action) and has_pending_form_actions(signature, actions):
                continue

            params = action.get("parameters", {})
            if not isinstance(params, dict):
                params = {}

            try:
                next_page_id, next_signature = transition(
                    page_id,
                    signature,
                    action,
                    params,
                    schema_lookup,
                )
            except Exception:
                continue

            if next_page_id not in page_index:
                continue

            next_depth = depth + 1
            next_state_key = state_key(next_page_id, next_signature)
            if next_state_key in visited_state_keys:
                continue

            visited_state_keys.add(next_state_key)
            reachable_page_ids.add(next_page_id)
            if next_page_id in terminal_ids and next_page_id not in terminal_shortest_hops:
                terminal_shortest_hops[next_page_id] = next_depth
            queue.append((next_page_id, next_signature, next_depth))

    return reachable_page_ids, terminal_shortest_hops


def run_bfs_summary(fsm_data: Dict[str, Any], max_depth: int = 40) -> Dict[str, Any]:
    max_depth = max(1, int(max_depth))

    norm = normalize_fsm(fsm_data)
    initial_page_id = norm.get("meta", {}).get("initial_page_id")
    if not isinstance(initial_page_id, str):
        initial_page_id = "HOME"

    page_ids = _extract_page_ids(fsm_data)
    terminal_ids = _extract_terminal_ids(norm.get("meta", {}))
    terminal_set = set(terminal_ids)

    reachable_page_ids, terminal_shortest_hops = _run_semantic_bfs(
        fsm_data=fsm_data,
        schema_lookup=norm.get("schema_lookup", {}),
        initial_page_id=initial_page_id,
        terminal_ids=terminal_set,
        max_depth=max_depth,
    )

    reachability_terminal_pages: List[Dict[str, Any]] = []
    for terminal_id in terminal_ids:
        shortest_hops = terminal_shortest_hops.get(terminal_id)
        reachability_terminal_pages.append(
            {
                "id": terminal_id,
                "reachable": shortest_hops is not None,
                "shortest_hops": shortest_hops if shortest_hops is not None else None,
            }
        )

    orphan_page_ids = sorted(
        page_id
        for page_id in page_ids
        if page_id not in terminal_set and page_id not in reachable_page_ids
    )
    unreachable_terminal_ids = sorted(
        terminal_id
        for terminal_id in terminal_ids
        if terminal_id not in terminal_shortest_hops
    )

    return {
        "reachability": {
            "initial_page_id": initial_page_id,
            "terminal_pages": reachability_terminal_pages,
        },
        "reachable_page_ids": sorted(reachable_page_ids),
        "orphan_page_ids": orphan_page_ids,
        "unreachable_terminal_ids": unreachable_terminal_ids,
        "max_depth": max_depth,
    }


def build_reachability_issues(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    reachability = summary.get("reachability", {})
    initial_page_id = reachability.get("initial_page_id")
    if not isinstance(initial_page_id, str):
        initial_page_id = "HOME"

    issues: List[Dict[str, Any]] = []

    for terminal_id in summary.get("unreachable_terminal_ids", []):
        if not isinstance(terminal_id, str):
            continue
        issues.append(
            {
                "where": {
                    "page": terminal_id,
                    "action": None,
                    "path": "$.meta.terminal_pages",
                },
                "type": "unreachable_terminal",
                "evidence": (
                    f"Terminal {terminal_id} is unreachable from {initial_page_id} "
                    "under local semantic BFS."
                ),
                "fix_suggestion": (
                    "Add or repair a reachable navigation path to this terminal, "
                    "or remove it from meta.terminal_pages."
                ),
            }
        )

    for page_id in summary.get("orphan_page_ids", []):
        if not isinstance(page_id, str):
            continue
        issues.append(
            {
                "where": {
                    "page": page_id,
                    "action": None,
                    "path": "$.pages",
                },
                "type": "orphan_page",
                "evidence": (
                    f"Page {page_id} is unreachable from {initial_page_id} "
                    "under local semantic BFS."
                ),
                "fix_suggestion": (
                    "Add or repair at least one reachable incoming navigation path "
                    "from an already reachable page."
                ),
            }
        )

    return issues
