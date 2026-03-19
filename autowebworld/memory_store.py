from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... truncated ..."


class MemoryStore:
    def __init__(self, path: Path):
        self.path = path.resolve()

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict[str, Any]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Memory file must be a JSON object: {self.path}")
        return data

    def save(self, memory: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")

    def initialize(
        self, fsm_data: dict[str, Any], fsm_source: str, project_root: str
    ) -> dict[str, Any]:
        pages = fsm_data.get("pages") or []
        meta = fsm_data.get("meta") or {}
        app_name = str(meta.get("app") or "unknown_app")

        pending_tasks: list[dict[str, Any]] = [
            {
                "id": "setup-runtime",
                "title": "Set up app shell, router, and FSM runtime core",
                "detail": "Create React app shell and core state/effects runtime to support FSM execution.",
                "attempts": 0,
                "created_at": _now_iso(),
            }
        ]
        for page in pages:
            page_id = str(page.get("id") or "").strip()
            if not page_id:
                continue
            action_count = len(page.get("actions") or [])
            pending_tasks.append(
                {
                    "id": f"page:{page_id}",
                    "title": f"Implement page {page_id}",
                    "detail": (
                        f"Create page route/component for `{page_id}` and wire its {action_count} "
                        "actions with parameters, preconditions, effects, and navigation."
                    ),
                    "attempts": 0,
                    "created_at": _now_iso(),
                }
            )
        pending_tasks.append(
            {
                "id": "final-polish",
                "title": "Final polish and build stabilization",
                "detail": "Improve consistency, tighten UX details, and ensure production build passes.",
                "attempts": 0,
                "created_at": _now_iso(),
            }
        )

        return {
            "version": 1,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "project_root": project_root,
            "fsm_source": fsm_source,
            "app_name": app_name,
            "pending_tasks": pending_tasks,
            "done_tasks": [],
            "known_issues": [],
            "turns": [],
        }

    def next_pending_task(self, memory: dict[str, Any]) -> dict[str, Any] | None:
        pending = memory.get("pending_tasks") or []
        return pending[0] if pending else None

    def increment_attempt(self, memory: dict[str, Any], task_id: str) -> None:
        for task in memory.get("pending_tasks") or []:
            if task.get("id") == task_id:
                task["attempts"] = int(task.get("attempts") or 0) + 1
                task["last_attempt_at"] = _now_iso()
                memory["updated_at"] = _now_iso()
                return

    def mark_task_done(self, memory: dict[str, Any], task_id: str, note: str = "") -> bool:
        pending = memory.get("pending_tasks") or []
        for idx, task in enumerate(pending):
            if task.get("id") != task_id:
                continue
            done_task = dict(task)
            done_task["completed_at"] = _now_iso()
            if note:
                done_task["note"] = _truncate(note, 1200)
            (memory.get("done_tasks") or []).append(done_task)
            pending.pop(idx)
            memory["updated_at"] = _now_iso()
            return True
        return False

    def add_issue(self, memory: dict[str, Any], issue_type: str, detail: str) -> None:
        issues = memory.setdefault("known_issues", [])
        issues.append({"time": _now_iso(), "type": issue_type, "detail": _truncate(detail, 2000)})
        if len(issues) > 200:
            del issues[:-200]
        memory["updated_at"] = _now_iso()

    def ensure_build_fix_task(self, memory: dict[str, Any], build_log: str) -> None:
        pending = memory.setdefault("pending_tasks", [])
        for task in pending:
            if task.get("id") == "build-fix":
                return
        pending.insert(
            0,
            {
                "id": "build-fix",
                "title": "Fix current build errors",
                "detail": (
                    "Resolve `npm run build` failures first.\n\nLatest build log excerpt:\n"
                    f"{_truncate(build_log, 1200)}"
                ),
                "attempts": 0,
                "created_at": _now_iso(),
            },
        )
        memory["updated_at"] = _now_iso()

    def record_turn(
        self,
        memory: dict[str, Any],
        turn_index: int,
        task_id: str,
        task_title: str,
        agent_output: str,
        build_ok: bool | None,
        build_log: str,
    ) -> None:
        turns = memory.setdefault("turns", [])
        turns.append(
            {
                "turn": turn_index,
                "time": _now_iso(),
                "task_id": task_id,
                "task_title": task_title,
                "build_ok": build_ok,
                "agent_output": _truncate(agent_output, 4000),
                "build_log_tail": _truncate(build_log, 2500),
            }
        )
        if len(turns) > 120:
            del turns[:-120]
        memory["updated_at"] = _now_iso()

    def render_brief(
        self, memory: dict[str, Any], max_pending: int = 8, max_done: int = 6, max_issues: int = 4
    ) -> str:
        pending = memory.get("pending_tasks") or []
        done = memory.get("done_tasks") or []
        issues = memory.get("known_issues") or []

        lines = [
            f"app_name: {memory.get('app_name', 'unknown')}",
            f"pending_count: {len(pending)}",
            f"done_count: {len(done)}",
            "",
            "pending_preview:",
        ]
        for task in pending[:max_pending]:
            lines.append(f"- {task.get('id')}: {task.get('title')}")
        if len(pending) > max_pending:
            lines.append(f"- ... ({len(pending) - max_pending} more)")

        lines.append("")
        lines.append("done_preview:")
        for task in done[-max_done:]:
            lines.append(f"- {task.get('id')}: {task.get('title')}")
        if not done:
            lines.append("- (none)")

        lines.append("")
        lines.append("issues_preview:")
        for issue in issues[-max_issues:]:
            lines.append(f"- {issue.get('type')}: {str(issue.get('detail', ''))[:180]}")
        if not issues:
            lines.append("- (none)")

        return "\n".join(lines)

    def write_todo_md(self, todo_path: Path, memory: dict[str, Any]) -> None:
        pending = memory.get("pending_tasks") or []
        done = memory.get("done_tasks") or []
        lines: list[str] = [
            "# AutoWebWorld Todo",
            "",
            f"- App: `{memory.get('app_name', 'unknown')}`",
            f"- Updated: `{memory.get('updated_at', _now_iso())}`",
            "",
            "## Pending",
        ]
        if pending:
            for task in pending:
                attempts = int(task.get("attempts") or 0)
                lines.append(f"- [ ] `{task.get('id')}` {task.get('title')} (attempts={attempts})")
        else:
            lines.append("- [x] No pending tasks")

        lines.extend(["", "## Done"])
        if done:
            for task in done:
                lines.append(f"- [x] `{task.get('id')}` {task.get('title')}")
        else:
            lines.append("- [ ] No completed tasks yet")

        todo_path.parent.mkdir(parents=True, exist_ok=True)
        todo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_style_txt_if_missing(self, style_path: Path, fsm_data: dict[str, Any]) -> None:
        if style_path.exists():
            return
        app = str((fsm_data.get("meta") or {}).get("app") or "Productivity App")
        page_count = len(fsm_data.get("pages") or [])
        lines = [
            f"App: {app}",
            "",
            "Color direction:",
            "- Base: soft neutral background with strong text contrast",
            "- Primary: blue/teal action accents for high clarity",
            "- Feedback: clear success/warning/error states with consistent badges",
            "",
            "Layout direction:",
            "- Desktop: left navigation + top utility bar + content canvas",
            "- Mobile: compact header + stacked sections + sticky primary actions",
            f"- Route count target: {page_count} pages mapped from FSM ids",
            "",
            "Component direction:",
            "- Action cards with explicit labels and parameter inputs",
            "- Consistent button hierarchy (primary/secondary/ghost)",
            "- Table/list components with stable spacing and hover states",
            "",
            "Interaction direction:",
            "- Fast transitions, no flashy motion; prioritize task throughput",
            "- Validation near inputs, with concise inline error text",
            "- Keep state visibility explicit (selected item, active filters, drafts)",
        ]
        style_path.parent.mkdir(parents=True, exist_ok=True)
        style_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
