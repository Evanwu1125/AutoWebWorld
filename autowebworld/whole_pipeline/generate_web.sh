#!/bin/bash
# generate_web.sh — Full pipeline: FSM → Frontend Codegen
#
# Usage:
#   ./generate_web.sh                           # use defaults
#   ./generate_web.sh --theme "team_chat-slack"  # override theme
#   ./generate_web.sh --skip-fsm                # skip FSM generation (reuse existing)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export https_proxy=http://127.0.0.1:1087
export http_proxy=http://127.0.0.1:1087

# --- Defaults ---
THEME="social_platform-medium"
FSM_MODEL="gpt-5.4"
CODEGEN_MODEL="gemini-3.1-pro-preview"
PAGES_MIN=10
PAGES_MAX=15
TERMINALS=5
SKIP_FSM=false
SKIP_FRONTEND=false

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --theme) THEME="$2"; shift 2 ;;
        --fsm-model) FSM_MODEL="$2"; shift 2 ;;
        --codegen-model) CODEGEN_MODEL="$2"; shift 2 ;;
        --pages-min) PAGES_MIN="$2"; shift 2 ;;
        --pages-max) PAGES_MAX="$2"; shift 2 ;;
        --terminals) TERMINALS="$2"; shift 2 ;;
        --skip-fsm) SKIP_FSM=true; shift ;;
        --skip-frontend) SKIP_FRONTEND=true; shift ;;
        --*) echo "Unknown arg: $1"; exit 1 ;;
        *) THEME="$1"; shift ;;
    esac
done

echo "========================================="
echo "  Theme:     $THEME"
echo "  Pages:     $PAGES_MIN - $PAGES_MAX"
echo "  Terminals: $TERMINALS"
echo "  FSM model: $FSM_MODEL"
echo "  Frontend:  $CODEGEN_MODEL"
echo "========================================="

# --- Step 1: FSM Generation ---
if [ "$SKIP_FSM" = false ]; then
    echo ""
    echo "[1/2] FSM Generation..."
    echo "-----------------------------------------"
    python -m src.agent.fsm --theme "$THEME" --model "$FSM_MODEL" --pages-min "$PAGES_MIN" --pages-max "$PAGES_MAX" --terminals "$TERMINALS"
else
    echo ""
    echo "[1/2] FSM Generation — SKIPPED"
fi

# --- Step 2: Frontend Codegen ---
if [ "$SKIP_FRONTEND" = false ]; then
    echo ""
    echo "[2/2] Frontend Codegen..."
    echo "-----------------------------------------"
    python -m src.agent.frontend_codegen --theme "$THEME" --model "$CODEGEN_MODEL"
else
    echo ""
    echo "[2/2] Frontend Codegen — SKIPPED"
fi

echo ""
echo "========================================="
echo "  Done!"
echo "  Output: fsm_outputs/$(echo "$THEME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_-]/_/g')/"
echo "========================================="
