"""
frontend_codegen.py — Use AgentLoop to generate a React web app from FSM JSON + reference screenshot.

Reads the FSM output and reference image, copies the React template,
then runs an agent loop that writes frontend code into the project.

Usage:
    python -m src.agent.frontend_codegen --theme "team_chat-slack"
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from .config import load_model_config
from .browser import load_image_as_base64
from .loop import AgentLoop, AgentResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent          # src/agent/
_PROJECT_ROOT = _THIS_DIR.parent.parent               # whole_pipeline/
_FSM_OUTPUTS_DIR = _PROJECT_ROOT / "fsm_outputs"
_TEMPLATE_DIR = _PROJECT_ROOT.parent / "template" / "react_template"
_PROMPTS_DIR = _THIS_DIR / "prompts"
_FRONTEND_CODEGEN_PROMPT = _PROMPTS_DIR / "frontend_codegen_prompt.txt"


def _slugify(theme: str) -> str:
    return re.sub(r"[^\w\-]", "_", theme.lower())


# ---------------------------------------------------------------------------
# Core: generate_frontend
# ---------------------------------------------------------------------------

def generate_frontend(
    theme: str,
    *,
    model: str | None = None,
    image_model: str = "gemini-3.1-flash-image-preview",
    fsm_path: Path | str | None = None,
    reference_image_path: Path | str | None = None,
    template_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    max_tokens: int | None = None,
    max_turns: int = 60,
) -> AgentResult:
    """Generate a React web app from FSM JSON + reference screenshot.

    Args:
        theme: Application theme, e.g. "team_chat-slack".
        model: Model name to use (looks up in config). If None, uses default_model.
        image_model: Model for image generation (default: dall-e-3).
        fsm_path: Path to FSM JSON (default: fsm_outputs/{theme}/fsm.json).
        reference_image_path: Path to reference screenshot (default: fsm_outputs/{theme}/reference.png).
        template_dir: Path to React template (default: autowebworld/template/react_template).
        config_path: Path to yaml config file.
        max_tokens: Max response tokens per turn.
        max_turns: Max agent loop turns.

    Returns:
        AgentResult with the final text and message history.
    """
    slug = _slugify(theme)
    theme_dir = _FSM_OUTPUTS_DIR / slug

    # --- Resolve paths ---
    fsm_file = Path(fsm_path) if fsm_path else theme_dir / "fsm.json"
    ref_image = Path(reference_image_path) if reference_image_path else theme_dir / "reference.png"
    tmpl_dir = Path(template_dir) if template_dir else _TEMPLATE_DIR

    if not fsm_file.exists():
        raise FileNotFoundError(f"FSM JSON not found: {fsm_file}")
    if not tmpl_dir.exists():
        raise FileNotFoundError(f"React template not found: {tmpl_dir}")

    # --- Load config ---
    cfg = load_model_config(config_path, model_name=model)
    resolved_model = cfg.get("model", model or "gpt-5.4")
    base_url = cfg.get("api_base")
    api_key = cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
    max_tokens = max_tokens or cfg.get("max_output_tokens") or 16000

    if not api_key:
        raise RuntimeError("API key not found in config or OPENAI_API_KEY env var")

    # --- Copy template to web/ ---
    web_dir = theme_dir / "web"
    if web_dir.exists():
        shutil.rmtree(web_dir)
    shutil.copytree(tmpl_dir, web_dir, ignore=shutil.ignore_patterns("node_modules", ".git", "pnpm-lock.yaml"))
    print(f"[frontend] Template copied to {web_dir}")

    # --- Load FSM JSON ---
    fsm_json = fsm_file.read_text(encoding="utf-8")
    fsm_data = json.loads(fsm_json)
    page_count = len(fsm_data.get("pages", []))
    print(f"[frontend] FSM loaded: {page_count} pages")

    # --- Load system prompt ---
    if not _FRONTEND_CODEGEN_PROMPT.exists():
        raise FileNotFoundError(f"System prompt not found: {_FRONTEND_CODEGEN_PROMPT}")
    system_prompt = _FRONTEND_CODEGEN_PROMPT.read_text(encoding="utf-8").strip()

    # --- Build user message ---
    user_parts = [
        f"Generate a React web application for the theme: **{theme}**\n\n",
        f"## FSM JSON\n\n```json\n{fsm_json}\n```\n\n",
        "Build this as a self-contained frontend. Use realistic mock data and React state for interactions.\n\n",
    ]

    if ref_image.exists():
        image_b64 = load_image_as_base64(ref_image)
        user_parts.append(
            "[Reference Screenshot] The image above shows the real website's homepage. "
            "Use it as visual design reference for colors, layout, and styling.\n"
        )
        print(f"[frontend] Reference image loaded: {ref_image}")
    else:
        print(f"[frontend] No reference image found at {ref_image}, proceeding without")

    user_message = "".join(user_parts)

    # --- Create and run AgentLoop ---
    print(f"[frontend] Starting agent loop (model={resolved_model}, max_turns={max_turns}) ...")

    loop = AgentLoop(
        api_key=api_key,
        model=resolved_model,
        system_prompt=system_prompt,
        cwd=web_dir,
        base_url=base_url,
        max_tokens=max_tokens,
        max_turns=max_turns,
        log_path=theme_dir / "frontend_codegen.log",
        cmd_timeout=300,
    )

    result = loop.run(user_message)

    print(f"\n[frontend] Agent finished. exit_status={result.exit_status}, turns={result.turn_count}")

    print(f"[frontend] Output directory: {web_dir}")

    return result


# Keep backward compatibility
generate_web = generate_frontend


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate React web app from FSM JSON")
    parser.add_argument("--theme", required=True, help="App theme, e.g. 'team_chat-slack'")
    parser.add_argument("--model", default=None, help="Model name from config (default: default_model)")
    parser.add_argument("--image-model", default="dall-e-3", help="Model for image generation (default: dall-e-3)")
    parser.add_argument("--fsm", default=None, help="Path to FSM JSON file")
    parser.add_argument("--reference-image", default=None, help="Path to reference screenshot")
    parser.add_argument("--template", default=None, help="Path to React template directory")
    parser.add_argument("--config", default=None, help="Path to yaml config file")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=50)
    args = parser.parse_args()

    result = generate_frontend(
        theme=args.theme,
        model=args.model,
        image_model=args.image_model,
        fsm_path=args.fsm,
        reference_image_path=args.reference_image,
        template_dir=args.template,
        config_path=args.config,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
    )

    print(f"\n[frontend] Final response:\n{result.final_text[:500]}")
