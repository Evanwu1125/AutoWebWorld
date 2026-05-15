from __future__ import annotations


def normalize_messages(messages: list) -> list:
    """Clean up messages before sending to the OpenAI-compatible API.

    For OpenAI format, messages use:
    - {"role": "assistant", "content": "...", "tool_calls": [...]}
    - {"role": "tool", "tool_call_id": "...", "content": "..."}

    This function ensures:
    1. Every tool_call has a matching tool result message
    2. Messages are well-formed
    """
    # Collect existing tool result IDs
    existing_results: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            existing_results.add(msg.get("tool_call_id", ""))

    # Find orphaned tool_calls and insert placeholder results
    patched = list(messages)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            if tc_id and tc_id not in existing_results:
                patched.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": "(cancelled)",
                })
                existing_results.add(tc_id)

    return patched
