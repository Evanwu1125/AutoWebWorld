"""
fsm.py — Single-call FSM generator via OpenAI-compatible API.

Reads the system/instruction prompts from src/agent/prompts/,
automatically discovers the reference website URL for the given theme,
screenshots it, and sends one API call with the screenshot to generate FSM JSON.

Usage:
    python -m src.agent.fsm --theme "team_chat-slack"
    python -m src.agent.fsm --theme "e_commerce-amazon" --reference-image ./ref.png
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import load_model_config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent       # src/agent/
_PROJECT_ROOT = _THIS_DIR.parent.parent            # whole_pipeline/
_FSM_OUTPUTS_DIR = _PROJECT_ROOT / "fsm_outputs"
_PROMPTS_DIR = _THIS_DIR / "prompts"
_DEFAULT_SYSTEM_PROMPT = _PROMPTS_DIR / "fsm_system_prompt.txt"
_DEFAULT_INSTRUCTION_PROMPT = _PROMPTS_DIR / "fsm_instruction_prompt.txt"

# Inline fallback for complexity profile (matches utils.DEFAULT_COMPLEXITY_PROFILE)
_DEFAULT_COMPLEXITY_PROFILE: dict[str, Any] = {
    "interceptors": {"cookie": False, "permissions": []},
    "pages": {"min": 20, "max": 30},
    "terminals": {"count": 8},
    "path_length": {"len": [4, 8], "count_interceptors": False},
    "home_nav_variants": ["direct"],
    "list_filters": {
        "enabled": False, "two_step_parity": True,
        "allowed_shapes": ["text_inputs", "checkboxes", "sliders", "sort"],
    },
    "policy": {
        "home_policy": "navigation_first",
        "back_policy": "recommended",
        "interceptors": {"cookie_placement": "home", "permission_placement": "first_relevant_non_home"},
    },
}

def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _slugify(theme: str) -> str:
    return re.sub(r"[^\w\-]", "_", theme.lower())


def _parse_theme(theme: str) -> tuple[str, str]:
    """Parse theme string into (category, brand). E.g. 'team_chat-slack' -> ('team_chat', 'slack')."""
    if "-" in theme:
        category, brand = theme.split("-", 1)
        return category, brand
    return theme, theme


# ---------------------------------------------------------------------------
# URL discovery
# ---------------------------------------------------------------------------

def discover_url(brand: str, client: OpenAI, model: str) -> str:
    """Ask LLM for the official homepage URL of the given brand/product.

    Args:
        brand: Product name, e.g. "slack", "amazon", "gmail".
        client: An initialized OpenAI client.
        model: Model to use.

    Returns:
        The discovered URL string (always starts with https://).
    """
    prompt = (
        f"What is the official homepage URL of the web product or service named '{brand}'?\n"
        "Reply with ONLY the URL, nothing else. Example: https://www.example.com"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
    )
    url = (response.choices[0].message.content or "").strip().strip('"').strip("'")
    if not url.startswith("http"):
        url = "https://" + url
    return url


# ---------------------------------------------------------------------------
# Core: generate_fsm
# ---------------------------------------------------------------------------

def generate_fsm(
    theme: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    system_prompt_path: Path | str | None = None,
    instruction_prompt_path: Path | str | None = None,
    complexity_profile: dict[str, Any] | None = None,
    reference_image_path: str | None = None,
    config_path: Path | str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Generate an FSM JSON for the given theme in one API call.

    Model, base_url, and api_key are loaded from fsm.config.yaml by default.
    Explicit arguments override config values.

    Args:
        theme: Application theme, e.g. "team_chat-slack".
        api_key: Anthropic API key (overrides config / env var).
        model: Claude model (overrides config).
        base_url: API base URL (overrides config).
        system_prompt_path: Override for the system prompt file.
        instruction_prompt_path: Override for the instruction prompt file.
        complexity_profile: Complexity profile dict (uses default if None).
        reference_image_path: Local image to use directly (skips auto-discovery).
        config_path: Path to yaml config file (default: config/fsm.config.yaml).
        max_tokens: Max response tokens (overrides config).

    Returns:
        Parsed FSM dict.
    """
    # --- Config ---
    cfg = load_model_config(config_path)
    model = model or cfg.get("model", "claude-sonnet-4-20250514")
    base_url = base_url or cfg.get("api_base")
    max_tokens = max_tokens or cfg.get("max_output_tokens") or 16000

    # --- Client ---
    key = api_key or cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("API key not found in config, arguments, or OPENAI_API_KEY env var")
    client_kw: dict[str, Any] = {"api_key": key}
    if base_url:
        # OpenAI SDK expects base_url to end with /v1
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        client_kw["base_url"] = base_url
    client = OpenAI(**client_kw)

    # --- Prompts ---
    sys_path = Path(system_prompt_path) if system_prompt_path else _DEFAULT_SYSTEM_PROMPT
    inst_path = Path(instruction_prompt_path) if instruction_prompt_path else _DEFAULT_INSTRUCTION_PROMPT
    if not sys_path.exists():
        raise FileNotFoundError(f"System prompt not found: {sys_path}")
    if not inst_path.exists():
        raise FileNotFoundError(f"Instruction prompt not found: {inst_path}")

    system_prompt = _load_text(sys_path)
    instruction_template = _load_text(inst_path)

    profile = complexity_profile or _DEFAULT_COMPLEXITY_PROFILE
    profile_json = json.dumps(profile, ensure_ascii=False, indent=2)

    instruction = (
        instruction_template
        .replace("{theme}", theme)
        .replace("{COMPLEXITY_PROFILE_JSON}", profile_json)
    )

    # --- Reference image ---
    image_b64: str | None = None
    category, brand = _parse_theme(theme)

    if reference_image_path and Path(reference_image_path).exists():
        from .browser import load_image_as_base64
        image_b64 = load_image_as_base64(reference_image_path)
        print(f"[fsm] Reference image loaded: {reference_image_path}")
    else:
        from .browser import screenshot_url, load_image_as_base64
        discovered_url = discover_url(brand, client, model)
        print(f"[fsm] Auto-discovered URL for '{brand}': {discovered_url}")
        theme_dir = _FSM_OUTPUTS_DIR / _slugify(theme)
        theme_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = str(theme_dir / "reference.png")
        print(f"[fsm] Screenshotting {discovered_url} via Playwright + stealth ...")
        try:
            screenshot_url(discovered_url, screenshot_path)
            image_b64 = load_image_as_base64(screenshot_path)
            print(f"[fsm] Screenshot saved: {screenshot_path}")
        except Exception as e:
            print(f"[fsm] ⚠️ Screenshot failed: {e}")
            print(f"[fsm] Continuing without reference image...")
            image_b64 = None

    if image_b64:
        instruction += (
            "\n\n[Reference Screenshot] The image attached shows the real website's homepage. "
            "Use it to understand the navigation structure, feature set, page hierarchy, "
            "and key UI patterns when designing the FSM. Mirror the real product's scope and flows."
        )

    # --- Build message content ---
    user_content: list[dict[str, Any]] | str
    if image_b64:
        user_content = [
            {"type": "text", "text": instruction},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                },
            },
        ]
    else:
        user_content = instruction

    # --- API call (streaming) ---
    print(f"[fsm] Calling {model} for theme '{theme}' ...")
    raw_text = ""
    input_tokens = 0
    output_tokens = 0

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            raw_text += delta.content
        if chunk.usage:
            input_tokens = chunk.usage.prompt_tokens
            output_tokens = chunk.usage.completion_tokens
    print(f"[fsm] Done. input_tokens={input_tokens}, output_tokens={output_tokens}")

    # --- Parse ---
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
        raw_text = re.sub(r"\n?```\s*$", "", raw_text)
    fsm_data = json.loads(raw_text)
    meta = fsm_data.setdefault("meta", {})
    meta.setdefault("app", f"{theme}_app")

    return fsm_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate FSM JSON via Claude (single call)")
    parser.add_argument("--theme", required=True, help="App theme, e.g. 'team_chat-slack'")
    parser.add_argument("--model", default=None, help="Claude model (overrides config)")
    parser.add_argument("--base-url", default=None, help="API base URL (overrides config)")
    parser.add_argument("--config", default=None, help="Path to yaml config file")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path (default: fsm_<theme>.json)")
    parser.add_argument("--reference-image", default=None, help="Local reference screenshot (skips auto-discovery)")
    parser.add_argument("--profile", default=None, help="Complexity profile JSON file path")
    parser.add_argument("--pages-min", type=int, default=None, help="Min pages (overrides profile)")
    parser.add_argument("--pages-max", type=int, default=None, help="Max pages (overrides profile)")
    parser.add_argument("--terminals", type=int, default=None, help="Terminal page count (overrides profile)")
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    profile = None
    if args.profile:
        profile = json.loads(Path(args.profile).read_text())

    if args.pages_min or args.pages_max or args.terminals:
        if profile is None:
            profile = dict(_DEFAULT_COMPLEXITY_PROFILE)
        if args.pages_min is not None:
            profile.setdefault("pages", {})["min"] = args.pages_min
        if args.pages_max is not None:
            profile.setdefault("pages", {})["max"] = args.pages_max
        if args.terminals is not None:
            profile.setdefault("terminals", {})["count"] = args.terminals

    fsm = generate_fsm(
        theme=args.theme,
        model=args.model,
        base_url=args.base_url,
        config_path=args.config,
        complexity_profile=profile,
        reference_image_path=args.reference_image,
        max_tokens=args.max_tokens,
    )

    out_path = args.output or str(_FSM_OUTPUTS_DIR / _slugify(args.theme) / "fsm.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(fsm, ensure_ascii=False, indent=2) + "\n")

    pages = len(fsm.get("pages", []))
    terminals = fsm.get("meta", {}).get("terminal_pages", [])
    print(f"\n[fsm] Saved: {out_path}")
    print(f"      Pages: {pages}")
    print(f"      Terminals: {terminals}")
