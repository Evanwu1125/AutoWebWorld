#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_tools import ProjectTools, ensure_project_from_template_or_vite


DEFAULT_FSM_PATH = Path(
    "trajectory/fsm/generator/fsm_perfect_outputs/team_chat-slack/perfect_fsm_team_chat-slack_100.json"
)
DEFAULT_PROJECT_DIR = Path("autowebworld/web_outputs")
DEFAULT_TEMPLATE_DIR = Path("autowebworld/template/react_template")
DEFAULT_CONTEXT_ROOT = Path("autowebworld/react_context")
DEFAULT_SYSTEM_PROMPT_PATH = SCRIPT_DIR / "prompts" / "react_system_prompt.txt"
DEFAULT_INSTRUCTION_PROMPT_PATH = SCRIPT_DIR / "prompts" / "instruction_prompt.txt"
COMPLETION_TOKEN = "ALL_TASKS_DONE"


def _import_mini_swe_agent(mini_src: str | None):
    candidates: list[Path] = []
    if mini_src:
        candidates.append(Path(mini_src))
    env_src = os.getenv("MINI_SWE_AGENT_SRC")
    if env_src:
        candidates.append(Path(env_src))
    candidates.append(Path.cwd() / "mini-swe-agent" / "src")
    candidates.append(Path("/Users/evanwu/Desktop/autoguiworld/mini-swe-agent/src"))

    tried: list[str] = []
    for candidate in candidates:
        if candidate.exists():
            tried.append(str(candidate))
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        try:
            import yaml
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.config import builtin_config_dir
            from minisweagent.environments.local import LocalEnvironment
            from minisweagent.models.litellm_model import LitellmModel
            from minisweagent.models.utils.content_string import get_content_string

            return DefaultAgent, LocalEnvironment, LitellmModel, builtin_config_dir, yaml, get_content_string
        except Exception:
            continue

    raise RuntimeError(
        "Cannot import mini-swe-agent. "
        "Install with `pip install mini-swe-agent` or set --mini-src/MINI_SWE_AGENT_SRC "
        f"to the repo src path. Tried: {tried}"
    )


def _new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def _new_timestamp_subdir(theme_root: Path) -> Path:
    theme_root = theme_root.resolve()
    theme_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = theme_root / stamp
    index = 1
    while candidate.exists():
        candidate = theme_root / f"{stamp}_{index:02d}"
        index += 1
    return candidate


def _normalize_token(raw: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw.strip()).strip("-_").lower()
    return token


def _extract_dash_suffix(raw: str) -> str:
    token = _normalize_token(raw)
    if "-" not in token:
        return ""
    suffix = token.rsplit("-", 1)[-1].strip("-_")
    return re.sub(r"[^a-z0-9]+", "", suffix)


def _extract_fsm_theme(fsm_path: Path) -> tuple[str, str]:
    # Primary source: theme folder/file naming convention like `team_chat-slack`.
    parent_name = fsm_path.parent.name
    parent_theme = _extract_dash_suffix(parent_name)
    if parent_theme:
        return parent_theme, f"path_parent:{parent_name}"

    stem = fsm_path.stem
    stem_match = re.match(r"perfect_fsm_(.+?)_\d+$", stem)
    stem_raw = stem_match.group(1) if stem_match else stem
    stem_theme = _extract_dash_suffix(stem_raw)
    if stem_theme:
        return stem_theme, f"file_stem:{stem_raw}"

    try:
        meta_app = str((json.loads(fsm_path.read_text(encoding="utf-8")) or {}).get("meta", {}).get("app", ""))
    except Exception:
        meta_app = ""
    meta_theme = _extract_dash_suffix(meta_app.replace("_app", ""))
    if meta_theme:
        return meta_theme, f"meta.app:{meta_app}"

    return "web", "fallback:web"


def _print_block(title: str, content: str) -> None:
    print(f"\n===== {title} =====")
    print(content)
    print(f"===== END {title} =====\n")


def run_agent_turn(
    *,
    project_root: Path,
    model: str,
    max_turns: int,
    mini_src: str | None,
    system_prompt: str,
    instruction_prompt_template: str,
    preloaded_skill_text: str,
    fsm_input: str,
    real_web_theme: str,
    trajectory_output_path: Path,
    model_api_base: str,
    model_provider: str,
    cmd_timeout_sec: int,
    cost_limit: float,
    stream_agent_output: bool,
    stream_command_output: bool,
) -> tuple[str, str, str]:
    (
        DefaultAgent,
        LocalEnvironment,
        LitellmModel,
        builtin_config_dir,
        yaml,
        get_content_string,
    ) = _import_mini_swe_agent(mini_src)

    mini_config = yaml.safe_load((builtin_config_dir / "mini.yaml").read_text(encoding="utf-8"))
    agent_cfg = dict(mini_config.get("agent") or {})
    model_cfg = dict(mini_config.get("model") or {})
    env_cfg = dict(mini_config.get("environment") or {})
    model_kwargs = dict(model_cfg.get("model_kwargs") or {})
    if model_api_base:
        model_kwargs["api_base"] = model_api_base
    if model_provider:
        model_kwargs["custom_llm_provider"] = model_provider

    base_system_template = str(agent_cfg.get("system_template") or "").strip()
    extra_system = system_prompt.strip()
    if extra_system:
        agent_cfg["system_template"] = (
            f"{base_system_template}\n\nAdditional project policy:\n{extra_system}"
            if base_system_template
            else extra_system
        )
    agent_cfg["step_limit"] = max_turns
    agent_cfg["cost_limit"] = cost_limit
    agent_cfg["output_path"] = trajectory_output_path

    llm = LitellmModel(
        model_name=model,
        model_kwargs=model_kwargs,
        format_error_template=str(model_cfg.get("format_error_template") or "{{ error }}"),
        observation_template=str(model_cfg.get("observation_template") or ""),
        multimodal_regex=str(model_cfg.get("multimodal_regex") or ""),
        cost_tracking="ignore_errors",
    )
    env = LocalEnvironment(
        cwd=str(project_root),
        env=dict(env_cfg.get("env") or {}),
        timeout=cmd_timeout_sec,
        stream_output=stream_command_output,
    )

    class StreamingAgent(DefaultAgent):
        def add_messages(self, *messages: dict) -> list[dict]:
            for msg in messages:
                role = msg.get("role") or msg.get("type", "unknown")
                if role == "assistant":
                    content = get_content_string(msg).strip()
                    if content:
                        print("\n[assistant]")
                        print(content)
                    actions = msg.get("extra", {}).get("actions", [])
                    if actions:
                        print("[assistant actions]")
                        for action in actions:
                            cmd = str(action.get("command") or "").strip()
                            if cmd:
                                print(f"$ {cmd}")
                elif role == "exit":
                    content = get_content_string(msg).strip()
                    if content:
                        print("\n[assistant exit]")
                        print(content)
            return super().add_messages(*messages)

    agent_cls = StreamingAgent if stream_agent_output else DefaultAgent
    agent = agent_cls(llm, env, **agent_cfg)

    user_task_template = instruction_prompt_template.replace("{{REAL_WEB_THEME}}", real_web_theme)
    if preloaded_skill_text.strip():
        user_task_template = (
            "Preloaded design skill guidance (already loaded; do not check environment availability):\n"
            f"{preloaded_skill_text.strip()}\n\n"
            f"{user_task_template}"
        )
    if "{{FSM_JSON}}" in user_task_template:
        user_task = user_task_template.replace("{{FSM_JSON}}", fsm_input)
    else:
        user_task = (
            f"{user_task_template.rstrip()}\n\n"
            f"REAL_WEB_THEME: {real_web_theme}\n\n"
            f"FSM JSON:\n{fsm_input}"
        )

    if not stream_agent_output:
        _print_block("MODEL INPUT", user_task)
    else:
        print("\n===== MODEL INPUT =====")
        print("(stream mode enabled; model input prompt is omitted from terminal output)")
        print("===== END MODEL INPUT =====\n")
    result = agent.run(user_task)
    exit_status = str(result.get("exit_status") or "unknown")
    submission = str(result.get("submission") or "").strip()
    final_output = (
        f"exit_status: {exit_status}\n"
        f"submission:\n{submission or '(no submission text returned by agent)'}\n\n"
        f"trajectory_file: {trajectory_output_path}"
    )
    if not stream_agent_output:
        _print_block("MODEL OUTPUT", final_output)
    else:
        print("\n===== MODEL OUTPUT =====")
        print(final_output)
        print("===== END MODEL OUTPUT =====\n")
    return user_task, final_output, submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn-based React coding agent for FSM-driven web builds (prompt-managed tasks)."
    )
    parser.add_argument("--fsm", default=str(DEFAULT_FSM_PATH), help="Input FSM JSON path")
    parser.add_argument(
        "--project-dir",
        default=str(DEFAULT_PROJECT_DIR),
        help="Base output dir; final theme dir is auto-derived as <base>/<theme>-web/<timestamp>",
    )
    parser.add_argument(
        "--template-dir",
        default=str(DEFAULT_TEMPLATE_DIR),
        help="Template project dir copied on init if package.json is missing",
    )
    parser.add_argument(
        "--system-prompt",
        default=str(DEFAULT_SYSTEM_PROMPT_PATH),
        help="Path to system prompt text file",
    )
    parser.add_argument(
        "--instruction-prompt",
        default=str(DEFAULT_INSTRUCTION_PROMPT_PATH),
        help="Path to user instruction prompt template file",
    )
    parser.add_argument(
        "--preloaded-skill-file",
        default=None,
        help="Optional local skill/reference markdown file to inline into the user task prompt",
    )
    parser.add_argument(
        "--context-root",
        default=str(DEFAULT_CONTEXT_ROOT),
        help="Root directory for per-run React context folders",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id for context folder name; defaults to timestamp",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name for mini-swe-agent")
    parser.add_argument(
        "--mini-src",
        default=None,
        help="Optional mini-swe-agent src path (e.g. /path/mini-swe-agent/src)",
    )
    parser.add_argument(
        "--model-api-base",
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="Optional OpenAI-compatible API base URL passed to litellm model_kwargs.api_base",
    )
    parser.add_argument(
        "--model-provider",
        default="openai",
        help="litellm custom provider name, e.g. openai/azure/groq",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable name that stores API key (copied to OPENAI_API_KEY if needed)",
    )
    parser.add_argument("--max-turns", type=int, default=50, help="Max model steps per turn loop")
    parser.add_argument("--cost-limit", type=float, default=0.0, help="Per-turn model cost limit. 0 disables.")
    parser.add_argument("--cmd-timeout-sec", type=int, default=300, help="Command timeout used inside mini environment.")
    parser.add_argument(
        "--stream-agent-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream assistant/tool actions to terminal during each turn",
    )
    parser.add_argument(
        "--stream-command-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream shell command stdout/stderr from local environment to terminal",
    )
    parser.add_argument("--build-timeout-sec", type=int, default=480, help="Build timeout seconds")
    parser.add_argument("--npm-client", default="npm", help="npm client name")
    parser.add_argument("--fsm-copy-name", default="fsm.json", help="FSM copy filename in project")
    parser.add_argument("--style-file", default="style.txt", help="Style guidance filename in project")
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip template/Vite initialization and assume project already exists",
    )
    parser.add_argument(
        "--reference-image",
        default=None,
        help="Path to a reference homepage screenshot; copied into project and noted in the agent prompt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fsm_path = Path(args.fsm).resolve()
    project_base_root = Path(args.project_dir).resolve()
    fsm_theme, theme_source = _extract_fsm_theme(fsm_path)
    project_theme_root = (project_base_root / f"{fsm_theme}-web").resolve()
    project_root = _new_timestamp_subdir(project_theme_root)
    template_dir = ((SCRIPT_DIR.parent / args.template_dir).resolve() if args.template_dir and not Path(args.template_dir).is_absolute()
                    else Path(args.template_dir).resolve()) if args.template_dir else None
    system_prompt_path = Path(args.system_prompt).resolve()
    instruction_prompt_path = Path(args.instruction_prompt).resolve()
    preloaded_skill_path = (
        ((SCRIPT_DIR.parent / args.preloaded_skill_file).resolve() if args.preloaded_skill_file and not Path(args.preloaded_skill_file).is_absolute()
         else Path(args.preloaded_skill_file).resolve())
        if args.preloaded_skill_file
        else None
    )
    context_root = Path(args.context_root).resolve()

    if not fsm_path.exists():
        raise FileNotFoundError(f"FSM file not found: {fsm_path}")
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {system_prompt_path}")
    if not instruction_prompt_path.exists():
        raise FileNotFoundError(f"Instruction prompt file not found: {instruction_prompt_path}")
    if preloaded_skill_path is not None and not preloaded_skill_path.exists():
        raise FileNotFoundError(f"Preloaded skill file not found: {preloaded_skill_path}")
    if args.api_key_env and os.getenv(args.api_key_env):
        os.environ.setdefault("OPENAI_API_KEY", str(os.getenv(args.api_key_env)))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set (checked --api-key-env={args.api_key_env}).")

    if not args.skip_init:
        ensure_project_from_template_or_vite(
            project_root=project_root,
            npm_client=args.npm_client,
            template_dir=template_dir,
        )

    tools = ProjectTools(project_root=project_root, fsm_path=fsm_path, npm_client=args.npm_client)
    copied_fsm_path = tools.copy_fsm_into_project(args.fsm_copy_name)

    # Copy reference screenshot into the project so the coding agent can read it as a file
    reference_image_project_path: Path | None = None
    if args.reference_image:
        src = Path(args.reference_image).resolve()
        if src.exists():
            dest = project_root / "reference_homepage.png"
            shutil.copy2(src, dest)
            reference_image_project_path = dest
            print(f"[reference] screenshot copied to {dest}")
        else:
            print(f"[warn] --reference-image path not found, skipping: {src}")
    run_id = args.run_id or _new_run_id(prefix="react")
    run_context_dir = (context_root / run_id).resolve()
    run_context_dir.mkdir(parents=True, exist_ok=False)
    turn_logs_dir = (run_context_dir / "turns").resolve()
    turn_logs_dir.mkdir(parents=True, exist_ok=True)

    context_run_meta_path = (run_context_dir / "run_meta.json").resolve()
    project_style_path = (project_root / args.style_file).resolve()

    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    instruction_prompt_template = instruction_prompt_path.read_text(encoding="utf-8")
    preloaded_skill_text = (
        preloaded_skill_path.read_text(encoding="utf-8")
        if preloaded_skill_path is not None
        else ""
    )
    fsm_input = copied_fsm_path.read_text(encoding="utf-8")

    # Prepend reference screenshot note to the instruction template if available
    if reference_image_project_path is not None:
        reference_note = (
            "A reference screenshot of the real website's homepage has been saved to "
            f"`reference_homepage.png` in your project directory. "
            "Use it to guide the visual design, layout, color scheme, and page structure "
            "of the web application you are building.\n\n"
        )
        instruction_prompt_template = reference_note + instruction_prompt_template

    context_run_meta = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "cli_args": sys.argv[1:],
        "backend": "mini-swe-agent",
        "project_theme_root": str(project_theme_root),
        "project_base_root": str(project_base_root),
        "project_dir": str(project_root),
        "fsm_theme": fsm_theme,
        "fsm_theme_source": theme_source,
        "loop_mode": "single_agent_run_then_build_once",
        "max_turns": args.max_turns,
        "model": args.model,
        "model_api_base": args.model_api_base,
        "model_provider": args.model_provider,
        "stream_agent_output": args.stream_agent_output,
        "stream_command_output": args.stream_command_output,
        "instruction_prompt": str(instruction_prompt_path),
        "completion_token_hint": COMPLETION_TOKEN,
    }
    context_run_meta_path.write_text(
        json.dumps(context_run_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    final_build_ok: bool | None = None
    turn = 1
    print(f"[turn {turn}] running agent")
    try:
        trajectory_output_path = (turn_logs_dir / f"turn_{turn:03d}.traj.json").resolve()
        model_input, agent_output, submission = run_agent_turn(
            project_root=project_root,
            model=args.model,
            max_turns=args.max_turns,
            mini_src=args.mini_src,
            system_prompt=system_prompt,
            instruction_prompt_template=instruction_prompt_template,
            preloaded_skill_text=preloaded_skill_text,
            fsm_input=fsm_input,
            real_web_theme=fsm_theme,
            trajectory_output_path=trajectory_output_path,
            model_api_base=args.model_api_base,
            model_provider=args.model_provider,
            cmd_timeout_sec=args.cmd_timeout_sec,
            cost_limit=args.cost_limit,
            stream_agent_output=args.stream_agent_output,
            stream_command_output=args.stream_command_output,
        )
    except Exception as exc:
        err = f"Agent execution failed: {exc}"
        _print_block(f"MODEL ERROR (turn {turn})", err)
        (turn_logs_dir / f"turn_{turn:03d}_agent_error.txt").write_text(
            err + "\n", encoding="utf-8"
        )
        raise

    (turn_logs_dir / f"turn_{turn:03d}_model_input.txt").write_text(
        model_input + "\n", encoding="utf-8"
    )
    (turn_logs_dir / f"turn_{turn:03d}_agent_output.txt").write_text(
        agent_output + "\n", encoding="utf-8"
    )

    if COMPLETION_TOKEN not in submission:
        print(
            f"[warn] completion token '{COMPLETION_TOKEN}' not found in submission; "
            "continuing to host build once without reopening a new turn."
        )

    build_ok, build_log = tools.run_build(timeout_sec=args.build_timeout_sec)
    final_build_ok = build_ok
    build_log_path = (turn_logs_dir / f"turn_{turn:03d}_final_build.log").resolve()
    build_log_path.write_text(build_log + "\n", encoding="utf-8")

    if not build_ok:
        (turn_logs_dir / f"turn_{turn:03d}_build_failed_note.txt").write_text(
            "Host build failed in single-run mode. Fix issues and rerun the script.\n"
            f"See: {build_log_path.name}\n",
            encoding="utf-8",
        )

    print("\n===== TURN SUMMARY =====")
    print(f"project_theme_root: {project_theme_root}")
    print(f"fsm_theme: {fsm_theme}")
    print(f"project_dir: {project_root}")
    print(f"run_id: {run_id}")
    print(f"context_dir: {run_context_dir}")
    print(f"fsm_copy: {copied_fsm_path}")
    print(f"instruction_prompt: {instruction_prompt_path}")
    print(f"style: {project_style_path}")
    print(f"final_build_ok: {final_build_ok if final_build_ok is not None else 'not reached'}")


if __name__ == "__main__":
    main()
