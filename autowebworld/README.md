# AutoWebWorld Web Synthesis

This folder is dedicated to FSM-to-Web synthesis code.

## Current entrypoint

- `fsm_to_web_agent.py`: use `openai-agents-python` as coding agent to generate a React/Vite frontend from a perfect FSM.
  - By default, it first copies `autowebworld/template/react_template` into the target project dir, then lets the agent modify it.
- `react_coding_agent.py`: iterative coding-agent framework with persistent memory (`memory.json`) for multi-step FSM-to-React implementation.
  - It saves full per-run context into `autowebworld/react_context/<run_id>/` (memory/todo/style/fsm snapshots/turn logs).

## Run

```bash
cd /Users/evanwu/Desktop/autoguiworld/AutoWebWorld
export OPENAI_API_KEY=YOUR_KEY

python autowebworld/fsm_to_web_agent.py \
  --agents-src /Users/evanwu/Desktop/autoguiworld/openai-agents-python/src \
  --fsm trajectory/fsm/generator/fsm_perfect_outputs/team_chat-slack/perfect_fsm_team_chat-slack_100.json \
  --project-dir autowebworld/web_outputs/team_chat-slack-web \
  --model gpt-5 \
  --max-turns 80

python autowebworld/react_coding_agent.py \
  --mini-src /Users/evanwu/Desktop/autoguiworld/mini-swe-agent/src \
  --fsm trajectory/fsm/generator/fsm_perfect_outputs/team_chat-slack/perfect_fsm_team_chat-slack_100.json \
  --project-dir autowebworld/web_outputs/team_chat-slack-web \
  --context-root autowebworld/react_context \
  --model openai/gpt-4o-mini \
  --max-turns 50
```

## Notes

- Keep all future FSM->Web generation scripts under this folder.
- Use `--skip-init` when iterating on an existing frontend project.
