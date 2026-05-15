from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TextIO

from openai import OpenAI, BadRequestError

from .tools import BUILTIN_TOOLS, VISION_TOOLS, create_builtin_handlers
from .browser import create_image_content_block
from .skill import SkillRegistry
from .todo import TodoManager


@dataclass
class AgentResult:
    """Value object returned by AgentLoop.run()."""

    final_text: str
    messages: list = field(repr=False)
    exit_status: str = "stop"
    turn_count: int = 0


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-style tool defs to OpenAI function-calling format."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


class AgentLoop:
    """Core agent loop: call LLM -> execute tools -> feed results back.

    Uses OpenAI-compatible API (via fsm.config.yaml).

    Usage::

        loop = AgentLoop(
            api_key="sk-...",
            model="gpt-5.4",
            system_prompt="You are a coding agent.",
            cwd=Path("/my/project"),
        )
        result = loop.run("Create a hello.py that prints hello world")
        print(result.final_text)
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        system_prompt: str,
        cwd: Path,
        tools: list[dict] | None = None,
        tool_handlers: dict[str, Callable[..., str]] | None = None,
        max_turns: int = 50,
        max_tokens: int = 8000,
        base_url: str | None = None,
        cmd_timeout: int = 120,
        stream_output: bool = True,
        log_path: Path | str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.cwd = Path(cwd).resolve()
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.stream_output = stream_output

        # --- Log file ---
        self._log_file: TextIO | None = None
        if log_path:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_path, "w", encoding="utf-8")

        # --- Skills ---
        _agent_dir = Path(__file__).resolve().parent  # src/agent/
        skills_dir = _agent_dir / "skills"
        self.skill_registry = SkillRegistry(skills_dir)
        skill_catalog = self.skill_registry.describe_available()
        if skill_catalog != "(no skills available)":
            self.system_prompt += f"\n\nSkills available (use load_skill to load):\n{skill_catalog}"

        # --- OpenAI client ---
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            if not base_url.rstrip("/").endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        # --- Tools: builtins + any extras ---
        self.todo = TodoManager()
        builtin_handlers = create_builtin_handlers(self.cwd, cmd_timeout=cmd_timeout, skill_registry=self.skill_registry)
        builtin_handlers["todo"] = lambda **kw: self.todo.update(kw["items"])
        if tools is not None and tool_handlers is not None:
            self.tools = BUILTIN_TOOLS + tools
            self.tool_handlers = {**builtin_handlers, **tool_handlers}
        else:
            self.tools = list(BUILTIN_TOOLS)
            self.tool_handlers = builtin_handlers

        # Convert to OpenAI format
        self.openai_tools = _to_openai_tools(self.tools)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> AgentResult:
        """Execute the agent loop for a single user request."""
        self._log(f"=== Agent Loop Start ({self.model}) ===")
        self._log(f"CWD: {self.cwd}")
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        turn = 0
        exit_status = "stop"

        while True:
            self._log(f"\n--- Turn {turn + 1} ---")
            # --- LLM call ---
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.openai_tools,
                    max_tokens=self.max_tokens,
                )
            except BadRequestError as e:
                if "content_filter" in str(e) or "content management policy" in str(e):
                    print(f"\033[31m[loop] Content filter triggered, retrying...\033[0m")
                    self._log(f"[WARN] Content filter triggered, retrying...")
                    turn += 1
                    if turn >= self.max_turns:
                        exit_status = "content_filter"
                        break
                    continue
                raise
            choice = response.choices[0]
            assistant_msg: dict = {"role": "assistant"}
            if choice.message.content:
                assistant_msg["content"] = choice.message.content
                self._log(f"[assistant] {choice.message.content}")

            if choice.message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]
            messages.append(assistant_msg)

            # --- Check stop condition ---
            if choice.finish_reason != "tool_calls":
                if self.todo.has_incomplete() and turn + 1 < self.max_turns:
                    nudge = (
                        "You stopped but your session plan still has incomplete items. "
                        "Continue working on the remaining tasks. "
                        "Do NOT repeat what you already did — pick up where you left off."
                    )
                    messages.append({"role": "user", "content": nudge})
                    self._log(f"[hook] Nudged: todo incomplete, injecting continuation prompt")
                    turn += 1
                    continue
                exit_status = choice.finish_reason or "stop"
                break

            # --- Execute tools ---
            tool_messages, used_todo = self._execute_tools(choice.message.tool_calls)
            if not tool_messages:
                exit_status = "no_tool_results"
                break

            # --- Todo reminder ---
            if used_todo:
                self.todo.state.rounds_since_update = 0
            else:
                self.todo.note_round_without_update()
                reminder = self.todo.reminder()
                if reminder:
                    tool_messages.append({"role": "user", "content": reminder})

            messages.extend(tool_messages)

            turn += 1
            if turn >= self.max_turns:
                exit_status = "max_turns"
                break

        # --- Extract final text ---
        final_text = choice.message.content or ""

        self._log(f"\n=== Agent Loop End (exit_status={exit_status}, turns={turn}) ===")
        if self._log_file:
            self._log_file.close()
            self._log_file = None

        return AgentResult(
            final_text=final_text,
            messages=messages,
            exit_status=exit_status,
            turn_count=turn,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_tools(self, tool_calls) -> tuple[list[dict], bool]:
        """Execute tool calls and return (tool result messages, used_todo)."""
        results = []
        used_todo = False
        for tc in tool_calls:
            name = tc.function.name
            if name == "todo":
                used_todo = True
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            handler = self.tool_handlers.get(name)
            if handler is None:
                output = f"Error: Unknown tool '{name}'"
            else:
                try:
                    output = handler(**args)
                except Exception as e:
                    output = f"Error: {e}"

            if self.stream_output:
                self._print_tool(name, args, output)

            # For vision tools, we still get the text part only for OpenAI format
            if name in VISION_TOOLS and isinstance(output, dict):
                text = output.get("text", "")
                results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": text,
                })
            else:
                results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(output),
                })
        return results, used_todo

    def _log(self, text: str) -> None:
        """Write a line to the log file (if open)."""
        if self._log_file:
            self._log_file.write(text + "\n")
            self._log_file.flush()

    def _print_tool(self, name: str, inputs: dict, output) -> None:
        ts = time.strftime("%H:%M:%S")
        if name == "bash":
            header = f"[{ts}] $ {inputs.get('command', '')}"
        elif name == "screenshot_url":
            header = f"[{ts}] > screenshot_url: {inputs.get('url', '')}"
        elif name == "view_image":
            header = f"[{ts}] > view_image: {inputs.get('path', '')}"
        else:
            header = f"[{ts}] > {name}"
        display = output.get("text", str(output)) if isinstance(output, dict) else str(output)

        print(f"\033[33m{header}\033[0m")
        print(display[:1000])

        self._log(header)
        self._log(display)
