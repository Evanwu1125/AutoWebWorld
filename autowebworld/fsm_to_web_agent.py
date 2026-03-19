#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


DEFAULT_FSM_PATH = Path(
    "trajectory/fsm/generator/fsm_perfect_outputs/team_chat-slack/perfect_fsm_team_chat-slack_100.json"
)
DEFAULT_PROJECT_DIR = Path("autowebworld/web_outputs/team_chat-slack-web")
DEFAULT_TEMPLATE_DIR = Path("autowebworld/template/react_template")


def _import_agents_sdk(agents_src: str | None):
    candidates: list[Path] = []
    if agents_src:
        candidates.append(Path(agents_src))
    env_src = os.getenv("OPENAI_AGENTS_SRC")
    if env_src:
        candidates.append(Path(env_src))
    candidates.append(Path.cwd().parent / "openai-agents-python" / "src")
    candidates.append(Path("/Users/evanwu/Desktop/autoguiworld/openai-agents-python/src"))

    tried: list[str] = []
    for candidate in candidates:
        if candidate.exists():
            tried.append(str(candidate))
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        try:
            from agents import Agent, Runner, function_tool  # type: ignore

            return Agent, Runner, function_tool
        except Exception:
            continue

    raise RuntimeError(
        "Cannot import openai-agents SDK. "
        "Install with `pip install openai-agents` or set --agents-src/OPENAI_AGENTS_SRC "
        f"to repo src path. Tried: {tried}"
    )


class ProjectTools:
    def __init__(self, project_root: Path, fsm_path: Path):
        self.project_root = project_root.resolve()
        self.fsm_path = fsm_path.resolve()

    def _resolve(self, rel_path: str) -> Path:
        target = (self.project_root / rel_path).resolve()
        if target != self.project_root and self.project_root not in target.parents:
            raise ValueError(f"Path escapes project root: {rel_path}")
        return target

    def list_files(self, rel_dir: str = ".") -> str:
        root = self._resolve(rel_dir)
        if not root.exists():
            return f"Path not found: {rel_dir}"
        if root.is_file():
            return str(root.relative_to(self.project_root))

        lines: list[str] = []
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(self.project_root)
            depth = len(rel.parts)
            if depth > 6:
                continue
            suffix = "/" if p.is_dir() else ""
            lines.append(f"{rel}{suffix}")
            if len(lines) >= 500:
                lines.append("... truncated ...")
                break
        return "\n".join(lines) if lines else "(empty)"

    def read_file(self, rel_path: str) -> str:
        p = self._resolve(rel_path)
        if not p.exists() or not p.is_file():
            return f"File not found: {rel_path}"
        data = p.read_text(encoding="utf-8")
        if len(data) > 200_000:
            return data[:200_000] + "\n... truncated ..."
        return data

    def write_file(self, rel_path: str, content: str) -> str:
        p = self._resolve(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"ok: wrote {rel_path} ({len(content)} chars)"

    def run_command(self, cmd: str, timeout_sec: int = 240) -> str:
        blocked = ["git reset --hard", "git checkout --", "rm -rf /", "sudo "]
        if any(token in cmd for token in blocked):
            return f"blocked command: {cmd}"

        completed = subprocess.run(
            cmd,
            shell=True,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out = (
            f"$ {cmd}\n"
            f"[exit_code] {completed.returncode}\n"
            f"[stdout]\n{completed.stdout}\n"
            f"[stderr]\n{completed.stderr}"
        )
        if len(out) > 20_000:
            out = out[:20_000] + "\n... truncated ..."
        return out

    def read_fsm_json(self) -> str:
        data = self.fsm_path.read_text(encoding="utf-8")
        if len(data) > 350_000:
            return data[:350_000] + "\n... truncated ..."
        return data


def ensure_project_from_template_or_vite(
    project_root: Path, npm_client: str, template_dir: Path | None
) -> None:
    package_json = project_root / "package.json"
    if package_json.exists():
        return

    project_root.parent.mkdir(parents=True, exist_ok=True)
    if project_root.exists() and any(project_root.iterdir()):
        raise RuntimeError(
            f"Project dir {project_root} exists and is not empty, but no package.json found."
        )

    if template_dir is not None and template_dir.exists():
        shutil.copytree(template_dir, project_root, dirs_exist_ok=False)
    else:
        subprocess.run(
            [npm_client, "create", "vite@latest", str(project_root), "--", "--template", "react-ts"],
            check=True,
        )
        subprocess.run([npm_client, "install"], cwd=project_root, check=True)


async def run_agent(
    fsm_path: Path,
    project_root: Path,
    model: str,
    max_turns: int,
    agents_src: str | None,
) -> str:
    Agent, Runner, function_tool = _import_agents_sdk(agents_src)
    tools = ProjectTools(project_root=project_root, fsm_path=fsm_path)

    @function_tool
    def read_fsm_json() -> str:
        """Read full FSM JSON spec used as the source of truth."""
        return tools.read_fsm_json()

    @function_tool
    def list_files(rel_dir: str = ".") -> str:
        """List files under project root (depth-limited)."""
        return tools.list_files(rel_dir)

    @function_tool
    def read_file(rel_path: str) -> str:
        """Read file text from project root."""
        return tools.read_file(rel_path)

    @function_tool
    def write_file(rel_path: str, content: str) -> str:
        """Write file text to project root (creates parent directories)."""
        return tools.write_file(rel_path, content)

    @function_tool
    def run_command(cmd: str, timeout_sec: int = 240) -> str:
        """Run shell command in project root, e.g. `npm run build`."""
        return tools.run_command(cmd, timeout_sec=timeout_sec)

    instructions = (
        "You are a senior frontend coding agent. Build a runnable React+TypeScript Vite web app "
        "from FSM spec. Use tool calls aggressively. Always start by calling read_fsm_json(). "
        "Then implement app code under project root. The project root may already be initialized "
        "from a React template; preserve and evolve that structure instead of rewriting from scratch. "
        "Requirements: "
        "1) one route/screen per FSM page id, "
        "2) state machine runtime that supports preconditions, effects(set/inc/clear/append_unique), and navigation, "
        "3) actions rendered as clickable controls, "
        "4) parameter inputs rendered when action has parameters, "
        "5) run `npm run build` and fix errors until build passes. "
        "Keep output production-oriented and minimal."
    )

    agent = Agent(
        name="FSM Frontend Builder",
        model=model,
        instructions=instructions,
        tools=[read_fsm_json, list_files, read_file, write_file, run_command],
    )

    user_task = (
        f"Build the website in this project root from FSM file: {fsm_path}. "
        "After coding, ensure `npm run build` succeeds, then summarize key files changed and how to run."
    )
    result = await Runner.run(agent, input=user_task, max_turns=max_turns)
    return str(result.final_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frontend web app from perfect FSM via openai-agents.")
    parser.add_argument("--fsm", default=str(DEFAULT_FSM_PATH), help="Path to perfect FSM JSON")
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR), help="Frontend output project directory")
    parser.add_argument(
        "--template-dir",
        default=str(DEFAULT_TEMPLATE_DIR),
        help="Template project dir to copy before agent coding (if project is not initialized)",
    )
    parser.add_argument("--model", default="gpt-5", help="Model name for openai-agents")
    parser.add_argument("--max-turns", type=int, default=60, help="Max agent turns")
    parser.add_argument(
        "--agents-src",
        default=None,
        help="Optional local openai-agents-python src dir (e.g. /path/openai-agents-python/src)",
    )
    parser.add_argument("--npm-client", default="npm", help="npm client binary name")
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip Vite project initialization (assume project already exists)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fsm_path = Path(args.fsm).resolve()
    project_root = Path(args.project_dir).resolve()
    script_dir = Path(__file__).resolve().parent
    template_dir = ((script_dir.parent / args.template_dir).resolve() if args.template_dir and not Path(args.template_dir).is_absolute()
                    else Path(args.template_dir).resolve()) if args.template_dir else None

    if not fsm_path.exists():
        raise FileNotFoundError(f"FSM not found: {fsm_path}")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if not args.skip_init:
        ensure_project_from_template_or_vite(
            project_root=project_root,
            npm_client=args.npm_client,
            template_dir=template_dir,
        )

    final_output = asyncio.run(
        run_agent(
            fsm_path=fsm_path,
            project_root=project_root,
            model=args.model,
            max_turns=args.max_turns,
            agents_src=args.agents_src,
        )
    )
    print("\n===== AGENT RESULT =====\n")
    print(final_output)


if __name__ == "__main__":
    main()
