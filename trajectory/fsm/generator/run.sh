#!/usr/bin/env bash
set -euo pipefail

# 固定约束（可通过同名环境变量覆盖）
DEFAULT_OPENAI_API_KEY=""
DEFAULT_MODEL="gpt-5"
DEFAULT_THEME="blog_platform-medium"

# 用法:
#   bash trajectory/fsm/generator/run.sh "trajectory/fsm/generator/outputs/my_fsm" 16 "trajectory/fsm/generator/profiles/medium.json"
#   FSM_THEME 可选覆盖固定主题（默认使用 DEFAULT_THEME）
THEME="${FSM_THEME:-${DEFAULT_THEME}}"
OUTPUT_DIR="${1:-trajectory/fsm/generator/fsm_perfect_outputs}"
CONCURRENT_COUNT="${2:-8}"
PROFILE_JSON="${3:-trajectory/fsm/generator/profiles/medium.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-${DEFAULT_OPENAI_API_KEY}}"
MODEL="${FSM_MODEL:-${DEFAULT_MODEL}}"

# 说明：base_url 使用 trajectory/fsm/generator/base_agent.py 中默认值
# 当前默认是 https://newapi.deepwisdom.ai/v1
python -m trajectory.fsm.generator.fsm \
  --theme "${THEME}" \
  --model "${MODEL}" \
  --concurrent_count "${CONCURRENT_COUNT}" \
  --output_dir "${OUTPUT_DIR}" \
  --profile_json "${PROFILE_JSON}"
