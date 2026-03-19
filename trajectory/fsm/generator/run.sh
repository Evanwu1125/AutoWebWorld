#!/usr/bin/env bash
set -euo pipefail

# Fixed constraints (can be overridden via environment variables with the same name)
DEFAULT_OPENAI_API_KEY=""
DEFAULT_MODEL="gpt-5.4"
DEFAULT_THEME="blog_platform-medium"

# Usage:
#   bash trajectory/fsm/generator/run.sh "trajectory/fsm/generator/outputs/my_fsm" 16 "trajectory/fsm/generator/profiles/medium.json"
#   FSM_THEME optionally overrides the fixed theme (defaults to DEFAULT_THEME)
THEME="${FSM_THEME:-${DEFAULT_THEME}}"
OUTPUT_DIR="${1:-trajectory/fsm/generator/fsm_perfect_outputs}"
CONCURRENT_COUNT="${2:-8}"
PROFILE_JSON="${3:-trajectory/fsm/generator/profiles/medium.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-${DEFAULT_OPENAI_API_KEY}}"
MODEL="${FSM_MODEL:-${DEFAULT_MODEL}}"

# Note: base_url uses the default value from trajectory/fsm/generator/base_agent.py
# Set OPENAI_BASE_URL environment variable to override
python -m trajectory.fsm.generator.fsm \
  --theme "${THEME}" \
  --model "${MODEL}" \
  --concurrent_count "${CONCURRENT_COUNT}" \
  --output_dir "${OUTPUT_DIR}" \
  --profile_json "${PROFILE_JSON}"
