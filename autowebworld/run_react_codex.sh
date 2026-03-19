#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

FSM_PATH="${1:-${REPO_DIR}/trajectory/fsm/generator/fsm_perfect_outputs/team_chat-slack/perfect_fsm_team_chat-slack_100.json}"
PROJECT_BASE_DIR="${REPO_DIR}/autowebworld/web_outputs"
CONTEXT_ROOT="${REPO_DIR}/autowebworld/react_context"
TEMPLATE_DIR="${REPO_DIR}/autowebworld/template/react_template"
INSTRUCTION_PROMPT="${REPO_DIR}/autowebworld/prompts/instruction_prompt.txt"
SYSTEM_PROMPT="${REPO_DIR}/autowebworld/prompts/react_system_prompt.txt"

# Theme rule: use the token after the last '-' in FSM parent directory name.
FSM_PARENT="$(basename "$(dirname "${FSM_PATH}")")"
FSM_THEME="${FSM_PARENT##*-}"
FSM_THEME="$(echo "${FSM_THEME}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_')"
FSM_THEME="${FSM_THEME:-web}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PROJECT_THEME_ROOT="${PROJECT_BASE_DIR}/${FSM_THEME}-web"
PROJECT_DIR="${PROJECT_THEME_ROOT}/${STAMP}"
RUN_ID="react_codex_${STAMP}_$(date +%6N)"
RUN_CONTEXT_DIR="${CONTEXT_ROOT}/${RUN_ID}"
TURN_DIR="${RUN_CONTEXT_DIR}/turns"
SESSION="codex_${FSM_THEME}_$(date +%H%M%S)"

mkdir -p "${PROJECT_THEME_ROOT}" "${TURN_DIR}" "${PROJECT_DIR}"
cp -R "${TEMPLATE_DIR}/." "${PROJECT_DIR}/"
cp "${FSM_PATH}" "${PROJECT_DIR}/fsm.json"

PROMPT_FILE="${TURN_DIR}/turn_001_model_input.txt"
python - <<'PY' "${SYSTEM_PROMPT}" "${INSTRUCTION_PROMPT}" "${PROJECT_DIR}/fsm.json" "${FSM_THEME}" "${PROMPT_FILE}"
from pathlib import Path
import sys

system_path, instruction_path, fsm_path, theme, out_path = sys.argv[1:]
system_prompt = Path(system_path).read_text(encoding="utf-8").strip()
instruction = Path(instruction_path).read_text(encoding="utf-8")
fsm_json = Path(fsm_path).read_text(encoding="utf-8")

user_prompt = instruction.replace("{{REAL_WEB_THEME}}", theme)
if "{{FSM_JSON}}" in user_prompt:
    user_prompt = user_prompt.replace("{{FSM_JSON}}", fsm_json)
else:
    user_prompt = f"{user_prompt.rstrip()}\n\nFSM JSON:\n{fsm_json}"

final_prompt = (
    "System policy (must follow):\n"
    f"{system_prompt}\n\n"
    "Task:\n"
    f"{user_prompt}\n"
)
Path(out_path).write_text(final_prompt, encoding="utf-8")
PY

tmux new-session -d -s "${SESSION}" "cd '${PROJECT_DIR}' && exec codex --full-auto --no-alt-screen"
PANE="${SESSION}:0.0"

# Wait for codex pane to become responsive before pasting prompt.
for _ in $(seq 1 30); do
  if tmux capture-pane -p -J -t "${PANE}" -S -30 2>/dev/null | grep -q .; then
    break
  fi
  sleep 0.2
done
tmux load-buffer "${PROMPT_FILE}"
tmux paste-buffer -t "${PANE}"
tmux send-keys -t "${PANE}" C-m
sleep 0.2
tmux send-keys -t "${PANE}" C-m
sleep 2
tmux capture-pane -p -J -t "${PANE}" -S -2000 > "${TURN_DIR}/turn_001_agent_output.txt"

cat <<MSG
Started local codex session.
- session: ${SESSION}
- theme: ${FSM_THEME}
- project_dir: ${PROJECT_DIR}
- context_dir: ${RUN_CONTEXT_DIR}

Useful commands:
  tmux attach -t ${SESSION}
  tmux capture-pane -p -J -t ${SESSION} -S -2000 | sed -n '1,200p'
  (cd ${PROJECT_DIR} && npm run build)
MSG
