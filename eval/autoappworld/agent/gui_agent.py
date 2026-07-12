from __future__ import annotations

from typing import Any

from easyagent.agent.react_agent import ReactAgent
from easyagent.debug.log import Color
from easyagent.model.schema import content_to_text
from easyagent.memory.base import BaseMemory
from easyagent.model.base import BaseLLM

# ============================================================================
# System Prompt (default)
# ============================================================================
GUI_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the action assistant would take. The reasoning "
    "process and action are enclosed within <think> </think> and <action> </action> tags, respectively, i.e., "
    "<think> reasoning process here </think><action> action here </action>."
    "If history information is provided in the instruction, refer to it to decide the next action, but avoid repeating the same click actions that have already appeared in the history."
)


TONGUI_SYSTEM_PROMPT = (
    """You are an assistant trained to navigate the web screen.
Given a task instruction, a screen observation, and an action history sequence,
output the next action and wait for the next observation.
Here is the action space:
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required.
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required.
3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required.
4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
6. `ENTER`: Enter operation, value and position are not applicable.
7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.

Format the action as a JSON object with the following keys:
{"action": "ACTION_TYPE", "value": "element", "position": [x,y]}
You can only output one action each time.
If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1."""
)

# ============================================================================
# UI-TARS System Prompt (Computer Use)
# ============================================================================
# UI-TARS does not use a system prompt; leave it empty
UITARS_SYSTEM_PROMPT = ""

# ============================================================================
# OpenCUA System Prompt (L2 Format)
# ============================================================================
OPENCUA_SYSTEM_PROMPT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

For each step, provide your response in this format:

Thought:
  - Step by Step Progress Assessment:
    - Analyze completed task parts and their contribution to the overall goal
    - Reflect on potential errors, unexpected results, or obstacles
    - If previous action was incorrect, predict a logical recovery step
  - Next Action Analysis:
    - List possible next actions based on current state
    - Evaluate options considering current state and previous actions
    - Propose most logical next action
    - Anticipate consequences of the proposed action
  - For Text Input Actions:
    - Note current cursor position
    - Consolidate repetitive actions (specify count for multiple keypresses)
    - Describe expected final text outcome
    - Use first-person perspective in reasoning

Action:
  Provide clear, concise, and actionable instructions:
  - If the action involves interacting with a specific target:
    - Describe target explicitly without using coordinates
    - Specify element names when possible (use original language if non-English)
    - Describe features (shape, color, position) if name unavailable
    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")
  - if the action involves keyboard actions like 'press', 'write', 'hotkey':
    - Consolidate repetitive keypresses with count
    - Specify expected text outcome for typing actions

Finally, output the action as PyAutoGUI code or the following functions:
- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}
- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}}"""

# ============================================================================
# User Prompt Template
# ============================================================================
# This template will be formatted with task_prompt in the environment
USER_PROMPT_TEMPLATE = (
    "In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
    "If history information is provided, consider it when choosing the next action.\n\n"
    "Available actions:\n"
    "- click: {{'action': 'click', 'coordinate': [x, y]}}\n"
    "- hover: {{'action': 'hover', 'coordinate': [x, y]}}\n"
    "- drag: {{'action': 'drag', 'from': [x1, y1], 'to': [x2, y2]}}\n"
    "- type_text: {{'action': 'type_text', 'text': 'content'}}\n"
    "- press_enter: {{'action': 'press_enter'}}\n"
    "- scroll: {{'action': 'scroll', 'value': 'down/up'}}\n"
    "- wait: {{'action': 'wait', 'time': seconds}}\n"
    "- answer: {{'action': 'answer', 'text': 'content'}} (MUST use when current screenshot contains ALL the necessary information to answer the query)\n\n"
    ""
    "Output format: <think>..</think><action>..</action>"
)

# ============================================================================
# UI-Venus Style User Prompt Template
# ============================================================================
# Simplified template following UI-Venus action space format
USER_PROMPT_TEMPLATE_VENUS = (
    "In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
    "Refer to the previous conversation history when choosing the next action.\n\n"
    "### Available Actions\n"
    "You may execute one of the following functions:\n"
    "- Click(box=(x1, y1))\n"
    "- Drag(start=(x1, y1), end=(x2, y2))\n"
    "- Scroll(start=(x1, y1), end=(x2, y2), direction='down/up/right/left')\n"
    "- Type(content='')\n"
    "- Wait()\n"
    "- Finished(content='')\n\n"
    "Output format: <think>..</think><action>..</action>"
)

# ============================================================================
# TongUI Style User Prompt Template (Web-Single)
# ============================================================================
# Template following TongUI's Web-Single action space format
USER_PROMPT_TEMPLATE_TONGUI = (
    "Task: '{task_prompt}'.\n"
)

# ============================================================================
# UI-TARS Style User Prompt Template
# ============================================================================
# UI-TARS's USER_PROMPT_TEMPLATE actually acts as a system prompt
# This template will be filled with task_prompt in the environment, then moved to the front in _build_messages_uitars
USER_PROMPT_TEMPLATE_UITARS = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{task_prompt}
"""

# ============================================================================
# OpenCUA Style User Prompt Template (L2 Format)
# ============================================================================
# OpenCUA USER_PROMPT_TEMPLATE contains the task instruction
USER_PROMPT_TEMPLATE_OPENCUA = """
# Task Instruction:
{task_prompt}

Please generate the next move according to the screenshot, task instruction and previous steps (if provided).
"""

# ============================================================================


class GuiAgent(ReactAgent):
    """GUI agent for step-by-step web actions from screenshot + text input."""

    # Mapping from agent_type to message_mode
    AGENT_TYPE_TO_MESSAGE_MODE = {
        "gui_agent": "ours",
        "tongui": "tongui",
        "ui_venus": "ours",
        "showui": "tongui",
        "ui_tars": "ui_tars",
        "open_cua": "default"
    }

    def __init__(
        self,
        model: BaseLLM,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        memory: BaseMemory | None = None,
        agent_type: str = "gui_agent",
        message_mode: str | None = None,
    ) -> None:
        # Select default system prompt based on agent_type
        if system_prompt is None:
            if agent_type in ["tongui", "showui"]:
                prompt = TONGUI_SYSTEM_PROMPT
            elif agent_type == "ui_tars":
                # UI-TARS does not use a system prompt; content goes in the first user message
                prompt = ""
            elif agent_type == "open_cua":
                prompt = OPENCUA_SYSTEM_PROMPT
            else:
                prompt = GUI_SYSTEM_PROMPT
        else:
            prompt = system_prompt

        # Auto-select message_mode from agent_type if not explicitly provided
        if message_mode is None:
            self.message_mode = self.AGENT_TYPE_TO_MESSAGE_MODE.get(agent_type, "ours")
        else:
            self.message_mode = message_mode

        super().__init__(model, system_prompt=prompt, tools=tools, memory=memory)

    def _build_system_prompt(self, user_prompt: str) -> str:
        return user_prompt

    def _build_messages(self) -> list[dict[str, Any]]:
        """
        Select message-building strategy based on message_mode.

        Supported modes:
        1. "default": Use the parent class default implementation
        2. "ours": Keep only the system message and the last user message; embed history inside the user message
        3. "tongui": TongUI style - place system prompt, task, and action history in a single user message
        4. "ui_tars": UI-TARS style - place system prompt content in the first user message
        """
        if self.message_mode == "default":
            return self._build_messages_default()
        elif self.message_mode == "tongui":
            return self._build_messages_tongui()
        elif self.message_mode == "ui_tars":
            return self._build_messages_uitars()
        else:  # "ours" or any other value
            return self._build_messages_ours()

    def _build_messages_default(self) -> list[dict[str, Any]]:
        """Default mode: delegate to the parent class implementation."""
        return super()._build_messages()

    def _build_messages_ours(self) -> list[dict[str, Any]]:
        """
        Ours mode: keep only the system message and the last user message.
        History from previous steps is embedded directly into the last user message.
        """
        # Get the base message list from the parent class
        msgs = super()._build_messages()

        # Collect assistant message history
        history_entries = []
        step_count = 0

        for msg in msgs:
            if msg.get("role") == "assistant":
                content_str = str(msg.get("content", ""))
                # Use the raw model output directly without re-parsing
                step_count += 1
                history_entries.append(f"Step {step_count}: {content_str}")

        # Embed history into the last user message
        if history_entries and len(msgs) > 0:
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    history_text = "\n\nHISTORY:\n" + "\n".join(history_entries) + "\n"

                    content = msgs[i].get("content", "")

                    # Insert history before the marker sentence if present
                    if isinstance(content, str):
                        insert_marker = "If history information is provided,"
                        if insert_marker in content:
                            msgs[i]["content"] = content.replace(
                                insert_marker,
                                history_text + insert_marker
                            )
                        else:
                            msgs[i]["content"] = content + history_text
                    # For multimodal content (list), find the text part and insert there
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                insert_marker = "If history information is provided,"
                                if insert_marker in text:
                                    item["text"] = text.replace(
                                        insert_marker,
                                        history_text + insert_marker
                                    )
                                else:
                                    item["text"] = text + history_text
                                break
                    break

        # Keep only system messages and the last user message to limit context length
        filtered_msgs = []

        # 1. Retain all system messages
        for msg in msgs:
            if msg.get("role") == "system":
                filtered_msgs.append(msg)

        # 2. Retain the last user message
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                filtered_msgs.append(msgs[i])
                break

        return filtered_msgs

    def _build_messages_tongui(self) -> list[dict[str, Any]]:
        """
        TongUI mode: pack system prompt, task, and action history into a single user message.
        Follows the pattern from TongUI-agent/tongui/data/template/shared_navigation.py.

        Message structure:
        1. System prompt
        2. Task instruction
        3. Action history (text + historical screenshots)
        4. Current screenshot (last image)
        """
        # Get the base message list from the parent class
        msgs = super()._build_messages()

        system_prompt = self._system_prompt or ""

        # Extract the last user message (contains task text and current screenshot)
        last_user_content = None
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                last_user_content = msgs[i].get("content", "")
                break

        if last_user_content is None:
            return []

        # Parse task text and current screenshot from the last user message
        task_text = ""
        current_image = None

        if isinstance(last_user_content, str):
            task_text = last_user_content
        elif isinstance(last_user_content, list):
            for item in last_user_content:
                if item.get("type") == "text":
                    task_text = item.get("text", "")
                elif item.get("type") == "image_url":
                    current_image = item  # Keep the most recent image

        # Collect action history (text responses + historical images)
        action_history = []
        for msg in msgs[:-1]:  # Exclude the last user message
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "assistant":
                content_str = str(content).strip()
                if content_str:
                    action_history.append({"type": "text", "text": content_str + "\n"})
            elif role == "user" and content:
                # Extract images from previous user messages
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image_url":
                            action_history.append(item)

        # Assemble the unified user message content
        user_content = []

        # 1. System prompt
        user_content.append({"type": "text", "text": system_prompt + "\n"})

        # 2. Task (avoid duplicating the "Task:" prefix if already present)
        if task_text.strip().startswith("Task:"):
            user_content.append({"type": "text", "text": task_text + "\n"})
        else:
            user_content.append({"type": "text", "text": f"Task: {task_text}\n"})

        # 3. Action history
        if action_history:
            user_content.append({"type": "text", "text": "Action History:\n"})
            user_content.extend(action_history)

        # 4. Current screenshot (placed last)
        if current_image:
            user_content.append(current_image)

        return [{"role": "user", "content": user_content}]

    def _build_messages_uitars(self) -> list[dict[str, Any]]:
        """
        UI-TARS mode: restructure messages so that:
        - The first user message contains only text (acts as the system prompt)
        - Intermediate assistant messages remain as-is
        - The last user message contains only the current screenshot

        Processing steps:
        1. Get the base message list (includes system + user/assistant messages)
        2. Remove system messages
        3. Split the last user message into text part and image parts
        4. Reconstruct the message list:
           - User: [system prompt text]
           - Assistant: Thought: ... Action: ...
           - ...
           - User: [current screenshot image]
        """
        # Get the base message list from the parent class
        msgs = super()._build_messages()

        # Drop system messages
        filtered_msgs = [msg for msg in msgs if msg.get("role") != "system"]

        if not filtered_msgs:
            return []

        # Find the last user message
        last_user_msg = None
        for i in range(len(filtered_msgs) - 1, -1, -1):
            if filtered_msgs[i].get("role") == "user":
                last_user_msg = filtered_msgs[i]
                break

        if last_user_msg is None:
            return filtered_msgs

        # Split the last user message into text and image parts
        last_user_content = last_user_msg.get("content", "")

        text_content = ""
        image_parts = []

        if isinstance(last_user_content, str):
            text_content = last_user_content
        elif isinstance(last_user_content, list):
            for item in last_user_content:
                if item.get("type") == "text":
                    text_content = item.get("text", "")
                elif item.get("type") == "image_url":
                    image_parts.append(item)

        # Assemble the new message list
        new_msgs = []

        # 1. First user message: text only (system prompt)
        if text_content:
            new_msgs.append({
                "role": "user",
                "content": [{"type": "text", "text": text_content}]
            })

        # 2. All assistant messages
        for msg in filtered_msgs:
            if msg.get("role") == "assistant":
                new_msgs.append(msg)

        # 3. Last user message: image only (current screenshot)
        if image_parts:
            new_msgs.append({
                "role": "user",
                "content": image_parts
            })

        return new_msgs

    async def run(self, user_input: str | dict[str, Any] | list[dict[str, Any]]) -> str:
        action_text, _ = await self.run_with_response(user_input)
        return action_text

    async def run_with_response(
        self,
        user_input: str | dict[str, Any] | list[dict[str, Any]],
    ) -> tuple[str, Any]:
        content = self._build_user_content(user_input)
        self.add_message(self._message_from_content(content))

        msgs = self._build_messages()
        kwargs: dict[str, Any] = {}
        if schema := self._get_tools_schema():
            kwargs["tools"] = schema

        # Use a low temperature for more deterministic outputs
        kwargs["temperature"] = 0.3

        if self._debug:
            self._log.info(_format_model_input(msgs), color=Color.GRAY)
        response = await self._model.call_with_history(msgs, **kwargs)
        if self._debug:
            self._log.info(f"Model response: {response.content}", color=Color.GRAY)
        action_text = _extract_action_block(response.content)
        self.add_message(self._assistant_message(action_text, response))
        return action_text, response


    def _message_from_content(self, content: Any):
        from easyagent.model.schema import Message

        return Message.user(content)

    def _assistant_message(self, content: str, response: Any):
        from easyagent.model.schema import Message

        tool_calls = None
        if getattr(response, "tool_calls", None):
            tool_calls = self._format_tool_calls(response.tool_calls)
        return Message.assistant(content, tool_calls)


def _extract_action_block(content: str) -> str:
    """
    Extract action block from LLM response.

    Supports both formats:
    - New format: <think>...</think><action>...</action>
    - Old format: <reason>...</reason><action>...</action>
    """
    # Try new format first (<think>...</think><action>...</action>)
    think_start = content.find("<think>")
    action_end = content.find("</action>")
    if think_start != -1 and action_end != -1:
        return content[think_start : action_end + len("</action>")].strip()

    # Fallback to old format (<reason>...</reason><action>...</action>)
    reason_start = content.find("<reason>")
    if reason_start != -1 and action_end != -1:
        return content[reason_start : action_end + len("</action>")].strip()

    return content.strip()


def _format_model_input(msgs: list[dict[str, Any]]) -> str:
    """Format model input messages for logging with clear separators."""
    lines = []
    for idx, msg in enumerate(msgs):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        text = content_to_text(content)
        # Add a blank line before each message (except the first)
        if idx > 0:
            lines.append("")
        lines.append(f"{'-'*12}{role}{'-'*12}")
        lines.append(text)
    return "\n".join(lines)
