from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable

from .browser import create_image_content_block, screenshot_url
from .skill import SkillRegistry

MAX_OUTPUT_CHARS = 50000

DANGEROUS_COMMANDS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]


def safe_path(cwd: Path, p: str) -> Path:
    """Resolve *p* relative to *cwd* and reject escapes."""
    path = (cwd / p).resolve()
    if not path.is_relative_to(cwd.resolve()):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def run_bash(command: str, *, cwd: Path, timeout: int = 120) -> str:
    if any(d in command for d in DANGEROUS_COMMANDS):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:MAX_OUTPUT_CHARS] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Timeout ({timeout}s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, *, cwd: Path, limit: int | None = None) -> str:
    try:
        lines = safe_path(cwd, path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str, *, cwd: Path) -> str:
    try:
        fp = safe_path(cwd, path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str, *, cwd: Path) -> str:
    try:
        fp = safe_path(cwd, path)
        file_content = fp.read_text(encoding="utf-8")
        if old_text not in file_content:
            return f"Error: Text not found in {path}"
        fp.write_text(file_content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool JSON schemas (sent to the Anthropic API)
# ---------------------------------------------------------------------------

BUILTIN_TOOLS: list[dict] = [
    {
        "name": "bash",
        "description": "Run a shell command in the project workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "description": "Max lines to read"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (creates parent dirs if needed).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace the first occurrence of exact text in a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "screenshot_url",
        "description": (
            "Take a screenshot of a web page. The screenshot image will be "
            "returned so you can see it. Use this to inspect reference websites."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to screenshot"},
                "filename": {
                    "type": "string",
                    "description": "Optional filename (saved under .screenshots/). Defaults to screenshot.png",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "view_image",
        "description": (
            "View a local image file. The image will be returned so you can "
            "see it. Use this to inspect reference screenshots or design assets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the image file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "load_skill",
        "description": "Load the full body of a named skill into the current context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the skill to load"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "todo",
        "description": "Rewrite the current session plan for multi-step work. Keep exactly one step in_progress.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Optional present-continuous label.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["items"],
        },
    },
]

# Tools whose results include an image content block alongside the text.
VISION_TOOLS: set[str] = {"screenshot_url", "view_image"}


def create_builtin_handlers(
    cwd: Path,
    cmd_timeout: int = 120,
    skill_registry: SkillRegistry | None = None,
) -> dict[str, Callable[..., Any]]:
    """Return a handler dict bound to *cwd*. Keys match BUILTIN_TOOLS names.

    Vision tool handlers (screenshot_url, view_image) return a **dict** with
    keys ``text`` and ``image_path`` so the agent loop can build a rich
    tool_result that includes both text and an image content block.
    """
    screenshots_dir = cwd / ".screenshots"

    def _handle_screenshot_url(**kw) -> dict:
        url = kw["url"]
        fname = kw.get("filename") or "screenshot.png"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        out_path = screenshots_dir / fname
        try:
            screenshot_url(url, str(out_path))
            return {
                "text": f"Screenshot saved to .screenshots/{fname}",
                "image_path": str(out_path),
            }
        except Exception as e:
            return {"text": f"Error taking screenshot: {e}", "image_path": None}

    def _handle_view_image(**kw) -> dict:
        raw_path = kw["path"]
        try:
            resolved = safe_path(cwd, raw_path)
        except ValueError as e:
            return {"text": f"Error: {e}", "image_path": None}
        if not resolved.exists():
            return {"text": f"Error: File not found: {raw_path}", "image_path": None}
        return {
            "text": f"Viewing image: {raw_path}",
            "image_path": str(resolved),
        }

    return {
        "bash": lambda **kw: run_bash(kw["command"], cwd=cwd, timeout=cmd_timeout),
        "read_file": lambda **kw: run_read(kw["path"], cwd=cwd, limit=kw.get("limit")),
        "write_file": lambda **kw: run_write(kw["path"], kw["content"], cwd=cwd),
        "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"], cwd=cwd),
        "screenshot_url": _handle_screenshot_url,
        "view_image": _handle_view_image,
        "load_skill": lambda **kw: skill_registry.load_full_text(kw["name"]) if skill_registry else "Error: No skills loaded",
    }
